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
    parser.add_argument("--seeds", type=int, default=None,
                        help="并行 seed 数(默认按难度自动: easy=1/medium=2/hard=3)；"
                             "任一判卷通过即停其它(竞速)")
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

    # 难度路由：按 OP_META.difficulty 自动分配 seed 预算（借鉴 KernelAgent auto_agent）
    _SEED_BY_DIFF = {"easy": 1, "medium": 2, "hard": 3}
    if args.seeds is None:
        diff = ops_registry.get_op(args.op).OP_META.get("difficulty", "easy")
        args.seeds = _SEED_BY_DIFF.get(diff, 1)
        if not args.quiet:
            print(f"[router] op={args.op} difficulty={diff} -> seeds={args.seeds}")
    seeds = max(1, args.seeds)
    if seeds == 1:
        agent = KernelAgent(max_rounds=args.rounds, max_tokens=args.max_tokens,
                            perf_mode=args.perf,
                            perf_min_speedup=args.perf_min_speedup,
                            memory_mode=args.memory, verbose=not args.quiet)
        summary, _steps = agent.run(args.op)
    else:
        # —— 多 seed 竞速：任一判卷通过即置位早停其余（借鉴 KernelAgent 多 worker 竞速）——
        import threading
        stop = threading.Event()
        cache = None if args.perf else {}   # perf 模式禁缓存(性能测量不走缓存)
        outcomes: list[tuple[bool, int, dict]] = []

        def _worker(idx: int) -> None:
            a = KernelAgent(max_rounds=args.rounds, max_tokens=args.max_tokens,
                            perf_mode=args.perf,
                            perf_min_speedup=args.perf_min_speedup,
                            memory_mode=args.memory, verbose=not args.quiet,
                            log_prefix=f"[seed{idx}] ")
            s, _st = a.run(args.op, stop_event=stop, code_cache=cache)
            outcomes.append((s["success"], idx, s))
            if s["success"]:
                stop.set()

        threads = [threading.Thread(target=_worker, args=(i,), daemon=True)
                   for i in range(seeds)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for succ, idx, s in sorted(outcomes, key=lambda o: o[1]):
            print(f"[seed{idx}] {'✔' if succ else '✗'} success={succ} "
                  f"rounds={s['rounds_used']} status={s['final_status']}")
        winners = [s for succ, _i, s in outcomes if succ]
        if winners:
            summary = min(winners, key=lambda s: s["rounds_used"])  # 取轮数最少的成功者
        else:
            summary = max(outcomes, key=lambda o: o[2]["rounds_used"])[2]

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
