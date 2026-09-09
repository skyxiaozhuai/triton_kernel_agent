"""NCU（Nsight Compute）剖析工具 —— 用真实硬件指标生成优化反馈。

可行性（本机已验证，GTX1650/sm_75 消费卡也能采）：
    Triton kernel 在 NCU 里的名字 = jit 函数名（无 triton 前缀），
    用 --kernel-name regex:<fn名> 可精确锁；dram/sm throughput 等核心
    roofline 计数器在消费卡上可用。

用法（供 run_opt / profile CLI 调用）：
    prof = ncu_profiler.profile(op_name, code=None)   # code=None 用 reference_triton
    # 返回 {kernel_name, block, grid, metrics:{...}}，另可生成 roofline 反馈文本：
    text = ncu_profiler.format_feedback(prof)
"""
from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import uuid

from .executor import PROJECT_ROOT, SCRATCH_DIR

# 每 kernel 采的指标（可加 occupancy/stall，消费卡部分受限）
METRICS = [
    "gpu__time_duration.sum",                                   # kernel 耗时 (nsecond)
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",       # DRAM 利用率 %（带宽饱和度）
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",         # SM 吞吐 %
    "sm__warps_active.avg.pct_of_peak_sustained_active",        # 占用率 %
    # cache 命中率（消费卡通常也支持；若某卡不支持需移出，见 _CACHE_KEYS）
    "l1tex__t_sector_hit_rate.pct",   # L1/TEX sector 命中率 %
    "lts__t_sector_hit_rate.pct",     # L2 sector 命中率 %
]

# cache 指标键（便于诊断/可选移除）
_CACHE_KEYS = ("l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct")

# torch 内部 kernel 名特征（CPU→cuda 构造输入后应几乎没有，兜底过滤）
_TORCH_NAME_HINTS = ("at::", "templates::", "vectorized", "elementwise_kernel",
                     "distribution", "Philox", "lambda", "normal_kernel",
                     "reduce_kernel", "TensorIterator")

# 常见 ncu 安装位置（服务器/容器：CUDA toolkit 或 Nsight Compute 独立包）
_NCU_CANDIDATES = (
    "/usr/local/cuda/bin/ncu",
    "/usr/local/cuda-12/bin/ncu",
    "/opt/nvidia/nsight-compute/ncu",
    "/opt/nvidia/nsight-compute/2024.1.1/ncu",
    "/opt/nvidia/nsight-compute/2023.3.0/ncu",
)


def find_ncu() -> str | None:
    """定位 ncu：env NCU_BIN 优先 → PATH → 常见安装路径。找不到返回 None。

    服务器/容器里 ncu 常不在 PATH（如 /usr/local/cuda/bin），用 NCU_BIN 指定最稳。
    """
    env = os.environ.get("NCU_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("ncu")
    if found:
        return found
    for p in _NCU_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _diag_ncu_error(stderr: str) -> None:
    """把常见 ncu 失败翻译成可操作提示（权限 / 版本 / 其它）。"""
    err = (stderr or "")[-2000:]
    if "ERR_NVGPUCTRPERM" in err:
        print("[ncu] 无性能计数器权限(ERR_NVGPUCTRPERM)：用 root/sudo 运行，"
              "或设驱动参数 NVreg_RestrictProfilingToAdminUsers=0 后重启")
    elif "ERR_NVGPUCTRTOOL" in err:
        print("[ncu] ncu 与驱动版本不匹配(ERR_NVGPUCTRTOOL)：装与驱动匹配的 "
              "Nsight Compute")
    elif err.strip():
        print("[ncu] 运行失败: " + err.strip().splitlines()[-1][:160])


def _build_target(op_name: str, code: str | None) -> str:
    if code is not None:
        body = f"""\
def _main():
    a0 = op.generate_inputs(device="cpu")   # CPU 生成 → .cuda()，避免 CUDA randn kernel 干扰
    call = {{k: (v.to("cuda") if torch.is_tensor(v) else v)
           for k, v in a0.items() if k != "meta"}}
    call.update(a0.get("meta", {{}}) or {{}})
    for _ in range(4):
        launch(**call)
        torch.cuda.synchronize()
"""
    else:  # reference_triton 路径
        body = f"""\
def _main():
    a0 = op.generate_inputs(device="cpu")
    args = {{k: v.to("cuda") if torch.is_tensor(v) else v
           for k, v in a0.items() if k != "meta"}}
    for _ in range(4):
        op.reference_triton(**args)
        torch.cuda.synchronize()
"""
    return (f"""\
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import torch
from benchmarks import ops_registry
op = ops_registry.get_op({op_name!r})

{code if code is not None else ''}

{body}
if __name__ == "__main__":
    _main()
""")


def _looks_torch(name: str) -> bool:
    return any(h in name for h in _TORCH_NAME_HINTS)


def profile(op_name: str, code: str | None = None, timeout: float = 300.0) -> dict | None:
    """对 op 的一个 Triton kernel 跑 ncu，返回 {kernel_name, block, grid, metrics}。

    code=None 时剖析模块的 reference_triton（自检用）；
    code 提供时剖析生成代码（含 def launch）里的 triton kernel（优化端用）。
    """
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    ncu_bin = find_ncu()
    if ncu_bin is None:
        print("[ncu] 未找到 ncu：请装 Nsight Compute（Ubuntu: apt install nsight-compute，"
              "或 CUDA toolkit 自带 /usr/local/cuda/bin/ncu），或设 env NCU_BIN=/path/to/ncu")
        return None
    script = _build_target(op_name, code)
    script_path = os.path.join(SCRATCH_DIR, f"ncu_{uuid.uuid4().hex[:6]}.py")
    with open(script_path, "w") as f:
        f.write(script)

    cmd = [ncu_bin, "--csv", "--launch-count", "6",
           "--metrics", ",".join(METRICS),
           sys.executable, script_path]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        _diag_ncu_error(proc.stderr)
        return None
    csv_out = proc.stdout

    # 找第一个非 torch 的 kernel（= Triton），取其首组指标
    # 注意：ncu 把 ==PROF== 日志也打到 stdout，解析前先只保留以 " 开头的 CSV 行
    csv_lines = [ln for ln in csv_out.splitlines() if ln.startswith('"')]
    chosen = None
    rows: list[dict] = []
    try:
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        for r in reader:
            if "Kernel Name" not in r or not r["Kernel Name"]:
                continue
            rows.append(r)
    except Exception:  # noqa: BLE001
        return None
    for r in rows:
        name = r["Kernel Name"]
        if _looks_torch(name):
            continue
        if chosen is None:
            chosen = {"kernel_name": name, "block": r.get("Block Size"),
                      "grid": r.get("Grid Size"), "metrics": {}}
        if name == chosen["kernel_name"] and r["Metric Name"] not in chosen["metrics"]:
            try:
                val = float(r["Metric Value"])
            except ValueError:
                continue
            chosen["metrics"][r["Metric Name"]] = {"value": val,
                                                   "unit": r.get("Metric Unit", "")}
    return chosen


def _get(prof: dict, key: str) -> float | None:
    m = prof["metrics"].get(key)
    return m["value"] if m else None


def format_feedback(prof: dict) -> str:
    """把 ncu 指标格式化成给 LLM 的 roofline 优化反馈。"""
    dur_ns = _get(prof, "gpu__time_duration.sum")   # metric unit: nsecond
    dram = _get(prof, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    sm = _get(prof, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    warps = _get(prof, "sm__warps_active.avg.pct_of_peak_sustained_active")
    ns = dur_ns if dur_ns is not None else None   # 单位 nsecond

    lines = [
        f"<硬件剖析(NCU)> kernel={prof['kernel_name']} "
        f"grid={prof.get('grid')} block={prof.get('block')}",
    ]
    if ns is not None:
        lines.append(f"耗时: {ns / 1000:.2f} us")   # nsecond -> 微秒
    if dram is not None:
        lines.append(f"DRAM 带宽利用率: {dram:.1f}% of peak")
    if sm is not None:
        lines.append(f"SM 利用率: {sm:.1f}% of peak")
    if warps is not None:
        lines.append(f"占用率(active warps): {warps:.1f}%")
    l1 = _get(prof, "l1tex__t_sector_hit_rate.pct")
    l2 = _get(prof, "lts__t_sector_hit_rate.pct")
    if l1 is not None and l2 is not None:
        lines.append(f"cache 命中率: L1={l1:.1f}%  L2={l2:.1f}%")

    # —— roofline 诊断 ——
    diag = []
    if l2 is not None and l2 >= 80:
        diag.append(f"L2 命中率高({l2:.0f}%)：数据复用较好，瓶颈在必须读主存的流量")
    elif l2 is not None and l2 < 30 and sm is not None and sm >= 70:
        diag.append(f"L2 命中率低({l2:.0f}%)且 compute 忙：可尝试改善数据复用/tile 调度")
    if dram is not None and sm is not None:
        if dram >= 80 and sm < 60:
            diag.append("诊断: memory-bound（DRAM 近峰值而 SM 空闲）—— 收益主要靠减内存流量，"
                        "不是靠加运算")
        elif sm >= 70 and dram < 60:
            diag.append("诊断: compute-bound（SM 忙而带宽有余）—— 考虑提升计算效率/占用/流水线")
        elif dram < 80 and sm < 60 and warps is not None and warps < 60:
            diag.append("诊断: under-utilized（占用率低、硬件没喂饱）—— 检查 grid/block/分支/"
                        "是否有足够并行度")
        else:
            diag.append("诊断: 已较饱和或混合 bound，优化空间有限")
    if ns is not None and dram is not None and dram >= 90:
        diag.append("提示: DRAM 利用率已 ≥90%，大概率已接近该 op 的带宽极限，可尝试方向有限")
    lines.append("；".join(diag))
    lines.append("</硬件剖析>")
    return "\n".join(lines)


if __name__ == "__main__":
    p = profile("vector_add")
    if p:
        print(format_feedback(p))
    else:
        print("profile 失败：确认 ncu 可用且 GPU 空闲。")
