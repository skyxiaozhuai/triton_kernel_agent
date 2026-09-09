#!/usr/bin/env python3
"""云端一键跑批 —— 在租的 GPU 服务器上按顺序执行完整验证 / 评测 / KernelBench。

把零散的 CLI 串成一次跑批，阶段产物各自落 results/，最后汇总 markdown：
  env    环境自检（GPU / torch / ncu 探测）
  tests  一把梭 run_all_tests（core/agent/gpu，含新 opt_beam/report_html）
  perf   全算子 run_all --perf（agent 端到端生成 + 性能 critic；花 LLM 时间，可限 --perf-ops）
  kb     KernelBench L1(默认) 批量跑（agent 每题；耗时长，可 --kb-ids 限子集）

用法（服务器上，用 conda env 的 python）：
  python scripts/cloud_run.py --dry                      # 只打印计划不执行
  python scripts/cloud_run.py --only env tests           # 只环境自检 + 一把梭
  python scripts/cloud_run.py --only perf --perf-ops vector_add,softmax,matmul
  python scripts/cloud_run.py --only kb --kb-ids 1-10    # KernelBench level1 前 10 题
  python scripts/cloud_run.py                            # 全阶段（很耗时，建议分段挂跑）

产物：results/cloud_<ts>.md 汇总 + 各脚本自带 results/summary_*.json / 轨迹。
先本地冒烟：python scripts/cloud_run.py --dry
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

STAGES = ("env", "tests", "perf", "kb")


def _log_run(label: str, cmd_args: list[str], report: list[str]) -> int:
    print(f"\n===== [{label}] python {' '.join(cmd_args)} =====", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + cmd_args, cwd=ROOT,
                          capture_output=True, text=True, timeout=3600 * 6)
    sec = time.time() - t0
    stdout = proc.stdout or ""
    tail = "\n".join(stdout.strip().splitlines()[-8:])
    print(tail, flush=True)
    print(f"[{label}] exit={proc.returncode} 用时 {sec:.0f}s", flush=True)
    report.append(f"\n## {label}  (exit={proc.returncode}, {sec:.0f}s)\n")
    report.append("```\n" + tail + "\n```")
    return proc.returncode


def _env_check(report: list[str]) -> None:
    print("===== [env] 环境自检 =====", flush=True)
    import torch  # noqa: F401
    ok_gpu = torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0) if ok_gpu else '无'}"
          f"  | torch {torch.__version__}  | sm={torch.cuda.get_device_capability(0) if ok_gpu else '-'}", flush=True)
    try:
        from agent.tools import ncu_profiler
        print("NCU →", ncu_profiler.find_ncu(), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("NCU 探测异常:", exc, flush=True)
    report.append("## env\n"
                  + f"- GPU: {torch.cuda.get_device_name(0) if ok_gpu else '无'}  torch {torch.__version__}")
    if not ok_gpu:
        print("[env] 无 GPU，无法继续；先装驱动/CUDA。")
        sys.exit(2)


def _parse_ids(spec: str) -> list[int]:
    """'1-10' → [1..10]；'1,3,5' → [1,3,5]；默认 None → 全量(由调用方处理)。"""
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def _kb_available_ids(level: int, root: str | None) -> list[int]:
    from benchmarks.kernelbench import problem as kbp
    base = kbp.default_kernelbench_root() if root is None else os.path.abspath(root)
    lvl = os.path.join(base, "KernelBench", f"level{level}")
    if not os.path.isdir(lvl):
        lvl = os.path.join(base, f"level{level}")
    if not os.path.isdir(lvl):
        raise SystemExit(f"找不到 KernelBench level{level}: {lvl}")
    ids = []
    for f in sorted(os.listdir(lvl)):
        if f.endswith(".py"):
            stem = f[:-3]
            num = "".join(ch for ch in stem.split("_")[0] if ch.isdigit())
            if num.isdigit():
                ids.append(int(num))
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=STAGES,
                    help="只跑指定阶段（默认全跑；注意全跑非常耗时）")
    ap.add_argument("--dry", action="store_true", help="只打印计划，不执行")
    ap.add_argument("--perf-ops", default=None,
                    help="perf 阶段限算子（逗号分隔，默认全部）")
    ap.add_argument("--kb-level", type=int, default=1)
    ap.add_argument("--kb-ids", default=None,
                    help="KernelBench 限题号：'1-10' 或 '1,3,5'（默认全量）")
    ap.add_argument("--kb-root", default=None, help="KernelBench 根目录")
    ap.add_argument("--shape", default=None,
                    help="全算子主 case shape 覆盖(env OP_SHAPE)，如 4096,4096,4096")
    args = ap.parse_args()

    stages = args.only or list(STAGES)
    if args.shape:
        os.environ["OP_SHAPE"] = args.shape
        os.environ["MATMUL_SHAPE"] = args.shape
    print(f"云端跑批计划: 阶段={stages}" + (f"  shape={args.shape}" if args.shape else "")
          + (f"  perf-ops={args.perf_ops}" if args.perf_ops else "")
          + (f"  kb=level{args.kb_level} ids={args.kb_ids or 'ALL'}" if "kb" in stages else ""))
    if args.dry:
        print("[dry] 以上为计划，未执行任何阶段。")
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report: list[str] = [f"# 云端跑批 {ts}", f"阶段: {stages}"]
    exit_codes: dict[str, int] = {}

    if "env" in stages:
        _env_check(report)
        exit_codes["env"] = 0

    if "tests" in stages:
        exit_codes["tests"] = _log_run("tests", ["scripts/run_all_tests.py"], report)

    if "perf" in stages:
        ops = [o.strip() for o in args.perf_ops.split(",")] if args.perf_ops else []
        exit_codes["perf"] = _log_run("perf", ["scripts/run_all.py", "--perf"] + ops, report)

    if "kb" in stages:
        if args.kb_ids:
            ids = _parse_ids(args.kb_ids)
        else:
            ids = _kb_available_ids(args.kb_level, args.kb_root)
            print(f"[kb] level{args.kb_level} 共 {len(ids)} 题", flush=True)
        report.append(f"\n## KernelBench level{args.kb_level}: {len(ids)} 题")
        n_ok = 0
        for pid in ids:
            root_args = ["--root", args.kb_root] if args.kb_root else []
            code = _log_run(f"kb-{pid}",
                            ["scripts/run_kernelbench.py", "--level", str(args.kb_level),
                             "--id", str(pid), "--quiet"] + root_args, report)
            if code == 0:
                n_ok += 1
            exit_codes[f"kb-{pid}"] = code
        report.append(f"\nKernelBench 通过 {n_ok}/{len(ids)}")

    md_path = os.path.join(RESULTS, f"cloud_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    failed = [k for k, v in exit_codes.items() if v]
    print("\n" + "=" * 60)
    print(f"跑批完成: 阶段失败 {failed if failed else '无'} | 汇总: {md_path}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
