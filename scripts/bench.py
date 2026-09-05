#!/usr/bin/env python3
"""性能基准 CLI：对算子跑 do_bench 对比表（triton vs eager vs torch.compile）。

用法（项目根目录）：
    python scripts/bench.py                     # 全部算子（含 torch.compile，较慢）
    python scripts/bench.py vector_add softmax  # 指定算子
    python scripts/bench.py --no-compile matmul # 跳过 torch.compile（快）

说明：GTX1650 + 小 shape 的数字仅作趋势参考；正式性能在服务器大 shape 跑。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton kernel 性能基准")
    parser.add_argument("ops", nargs="*", help="算子名；缺省跑全部")
    parser.add_argument("--no-compile", action="store_true", help="跳过 torch.compile")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA")
        return 2

    from benchmarks import ops_registry
    from agent.tools.benchmark import bench_op, format_table

    ops = args.ops or ops_registry.list_ops()
    print(f"基准算子: {ops} | torch.compile: {'跳过' if args.no_compile else '开启'}")
    print("(小 shape / sm_75，数字仅供参考；预热+多次取最小)\n")
    results = []
    for name in ops:
        print(f"[bench] {name} ...", flush=True)
        results.append(bench_op(name, with_compile=not args.no_compile))
    print("\n" + format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
