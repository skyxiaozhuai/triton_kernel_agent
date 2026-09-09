#!/usr/bin/env python3
"""NCU 剖析 CLI：对某个 op 的真实 Triton kernel 采 roofline 指标并生成优化反馈。

用法（需 GPU + ncu）：
    python scripts/profile_kernel.py --op vector_add --from-memory   # 经验库里的真实生成 kernel
    python scripts/profile_kernel.py --op matmul --from-memory
    python scripts/profile_kernel.py --op softmax --ref             # 剖析 reference_triton
    python scripts/profile_kernel.py --op matmul --code-file gen.py  # 指定代码文件
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tools import ncu_profiler  # noqa: E402
from benchmarks import ops_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-memory", action="store_true",
                     help="用 results/memory/<op>.json 里的真实生成代码")
    src.add_argument("--ref", action="store_true", help="剖析 reference_triton")
    src.add_argument("--code-file", default=None, help="从文件读生成代码")
    args = ap.parse_args()

    if args.op not in ops_registry.list_ops():
        print(f"未知算子: {args.op}。可用: {ops_registry.list_ops()}")
        return 2

    code = None
    source_note = "reference_triton"
    if args.code_file:
        with open(args.code_file, encoding="utf-8") as f:
            code = f.read()
        source_note = os.path.basename(args.code_file)
    elif args.from_memory or not args.ref:
        mem_path = os.path.join(ROOT, "results", "memory", f"{args.op}.json")
        if os.path.exists(mem_path):
            with open(mem_path, encoding="utf-8") as f:
                code = json.load(f)["code"]
            source_note = "results/memory"
        elif args.from_memory:
            print(f"results/memory/{args.op}.json 不存在（先跑 agent 成功入库）")
            return 2

    print(f"剖析 {args.op} ({source_note}) ...（ncu 每次约 20-40s）")
    prof = ncu_profiler.profile(args.op, code=code)
    if not prof:
        print("❌ profile 失败：确认 ncu 可用、GPU 空闲、代码可运行。")
        return 1
    print("=" * 60)
    print(ncu_profiler.format_feedback(prof))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
