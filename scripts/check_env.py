#!/usr/bin/env python3
"""环境自检：GPU / torch / triton 版本 + 最小 Triton kernel 冒烟测试。

用法：
    python scripts/check_env.py

输出结论解读：
    - 全部 OK  : 本机可做【小 shape 正确性冒烟】开发（GTX1650 4G 也够）。
    - 性能数据 : 仍需在服务器（大 shape / do_bench / GEMM）上跑。
"""
import platform
import sys

# 注意：triton/tl 必须在【模块全局】导入。
# triton.jit 编译 kernel 时按函数源码重新解析、从全局命名空间查名字(如 tl)，
# 若放在函数内 import，会报 NameError('tl is not defined')。
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # triton 未随 torch 安装
    triton = None
    tl = None
    _HAS_TRITON = False


def main():
    print("=" * 62)
    print("Triton Kernel Agent — 环境自检")
    print("=" * 62)
    print(f"Python  : {platform.python_version()} ({platform.platform()})")

    # ---- torch ----
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 未安装 torch: {e}")
        print("       CPU 开发: pip install torch  (仅写代码)")
        sys.exit(1)
    print(f"torch   : {torch.__version__}  (cuda={torch.version.cuda})")

    # ---- GPU ----
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用 —— 本机只能做 CPU 开发，无法冒烟跑 Triton。")
        print("       请到有 NVIDIA GPU 的机器(或服务器)上运行本脚本。")
        sys.exit(2)

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)  # (major, minor)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU     : {dev}")
    print(f"  sm    : sm_{cap[0]}{cap[1]}   memory={mem:.1f} GiB")
    if cap[0] < 7:
        print("[FAIL] Triton 需要 Volta(sm_70) 及以上，当前 GPU 过旧。")
        sys.exit(3)
    if cap < (8, 0):
        print("[INFO] sm_75(Turing)：基础 kernel 可跑；"
              "tl.dot/tensor-core 高级特性受限，性能以服务器为准。")

    # ---- triton (bundled) ----
    if _HAS_TRITON:
        print(f"triton  : {triton.__version__} (bundled with torch)")
    else:
        print("[WARN] 未随 torch 找到 triton。建议使用 torch 自带 bundled triton。")

    # ---- 最小 kernel 冒烟 ----
    if not _HAS_TRITON:
        print("[FAIL] triton 不可用，无法冒烟")
        sys.exit(5)
    try:
        @triton.jit
        def _add_kernel(x1, x2, y, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)
            tl.store(y + offs, v, mask=mask)

        n = 1 << 16
        x1 = torch.randn(n, device="cuda", dtype=torch.float32)
        x2 = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x1)
        _add_kernel[(triton.cdiv(n, 1024),)](x1, x2, y, n, BLOCK=1024)
        torch.cuda.synchronize()
        ok = torch.allclose(y, x1 + x2, atol=1e-5, rtol=1e-4)
        print(f"[{'OK  ' if ok else 'FAIL'}] 最小 Triton vector-add 冒烟 (n=2^16)")
        if not ok:
            sys.exit(4)
    except Exception:  # noqa: BLE001
        import traceback
        print("[FAIL] Triton 冒烟测试抛异常：")
        traceback.print_exc()
        sys.exit(5)

    print("=" * 62)
    print("结论: 本机可做【小 shape 正确性冒烟】开发。")
    print("      性能 / 大 shape / GEMM 请部署到服务器后跑。")
    print("=" * 62)


if __name__ == "__main__":
    main()
