#!/usr/bin/env python3
"""CLI：跑一个算子的 agent 生成闭环（真实调用 LLM + GPU）。

用法（在项目根目录）：
    python scripts/run_agent.py                    # 默认 vector_add
    python scripts/run_agent.py softmax --rounds 5
    python scripts/run_agent.py matmul --quiet

需要：GPU + .env 里配好 DEEPSEEK_API_KEY。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton Kernel Agent — 单算子闭环")
    parser.add_argument("op", nargs="?", default="vector_add")
    parser.add_argument("--rounds", type=int, default=6, help="最大迭代轮数")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--perf", action="store_true",
                        help="开启性能 critic(do_bench vs eager，达标才停)")
    parser.add_argument("--perf-min-speedup", type=float, default=0.9,
                        help="性能门槛：speedup_vs_eager 低于此值进入优化轮")
    parser.add_argument("--memory", action="store_true",
                        help="RAG：检索同类历史成功 kernel 作参考")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA 才能跑 agent 闭环。")
        return 2

    from benchmarks import ops_registry
    if args.op not in ops_registry.list_ops():
        print(f"未知算子: {args.op}。可用: {ops_registry.list_ops()}")
        return 2

    from agent.loop import KernelAgent
    agent = KernelAgent(max_rounds=args.rounds, max_tokens=args.max_tokens,
                        perf_mode=args.perf, perf_min_speedup=args.perf_min_speedup,
                        memory_mode=args.memory, verbose=not args.quiet)
    summary, _steps = agent.run(args.op)

    print("=" * 56)
    print(f"结果: {'✔ 通过' if summary['success'] else '✗ 未通过'}")
    print(f"  算子       : {summary['op']}")
    print(f"  使用轮数   : {summary['rounds_used']}")
    print(f"  总耗时     : {summary['wall_s']} s")
    print(f"  token      : {summary['total_tokens']}")
    print(f"  末轮状态   : {summary['final_status']}"
          + (f" (err={summary['final_max_abs_err']:.3e})"
             if summary['final_max_abs_err'] is not None else ""))
    if summary.get("final_speedup_vs_eager") is not None:
        print(f"  末轮性能   : speedup_vs_eager="
              f"{summary['final_speedup_vs_eager']}x")
    if summary.get("memory_used"):
        print(f"  RAG 参考   : {summary['memory_used']}")
    print("=" * 56)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
