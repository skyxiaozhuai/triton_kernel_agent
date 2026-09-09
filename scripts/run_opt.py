#!/usr/bin/env python3
"""优化端闭环 CLI —— 对一个"已正确"kernel 做 hardware-guided 持续优化。

流程: 基线(经验库/文件/先生成) → 每轮 [NCU 剖析 → LLM 优化 → 验证+do_bench]
      → 更快则接受 → 连续无改进收敛 → best + 收敛曲线(存 results/opt_*)

用法（需 GPU；ncu 剖析可选 --no-ncu 关闭）：
    python scripts/run_opt.py --op vector_add                      # 用经验库成功代码起步
    python scripts/run_opt.py --op matmul --code-file gen.py
    python scripts/run_opt.py --op matmul --generate --gen-rounds 4  # 先让 agent 生成正确版
    python scripts/run_opt.py --op softmax --opt-rounds 5 --stall 2 --improve-min 0.01
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from benchmarks import ops_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--code-file", default=None, help="用文件里的正确代码起步")
    src.add_argument("--generate", action="store_true",
                     help="先让 agent 生成正确版本再优化（花 API）")
    ap.add_argument("--opt-rounds", type=int, default=6)
    ap.add_argument("--stall", type=int, default=2, help="连续无改进即收敛")
    ap.add_argument("--improve-min", type=float, default=0.02,
                    help="相对 ms 改进 ≥ 此比例才接受(默认2%%，滤 do_bench 噪声)")
    ap.add_argument("--shape", default=None,
                    help="matmul 主 case 形状覆盖，如 4096,4096,4096（设 env MATMUL_SHAPE，判卷/剖析子进程继承）")
    ap.add_argument("--no-ncu", action="store_true", help="关闭 NCU 剖析")
    ap.add_argument("--gen-rounds", type=int, default=6, help="--generate 时生成的最大轮数")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="优化模式 prompt 较长，默认给足避免推理截断")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.shape:
        os.environ["MATMUL_SHAPE"] = args.shape   # executor/ncu 子进程会继承该 env
    if args.op not in ops_registry.list_ops():
        print(f"未知算子: {args.op}。可用: {ops_registry.list_ops()}")
        return 2
    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA（本机小 shape 出趋势；服务器大 shape 出硬数字）。")
        return 2

    init_code = None
    source_note = "results/memory"
    if args.code_file:
        with open(args.code_file, encoding="utf-8") as f:
            init_code = f.read()
        source_note = os.path.basename(args.code_file)
    elif args.generate:
        print(f"[gen] 先生成 {args.op} 的正确 kernel ...")
        from agent.loop import KernelAgent
        from agent import memory
        summary, steps = KernelAgent(max_rounds=args.gen_rounds,
                                     verbose=not args.quiet).run(args.op)
        if not summary["success"]:
            print("✗ 生成未成功，无法进入优化。")
            return 1
        init_code = steps[-1].code
        source_note = f"刚生成的正确代码(第{summary['rounds_used']}轮)"
    # 否则：KernelOptimizer 自动从经验库取

    from agent.opt_loop import KernelOptimizer
    opt = KernelOptimizer(max_tokens=args.max_tokens, verbose=not args.quiet)
    print(f"优化 {args.op}（起点: {source_note}，opt_rounds={args.opt_rounds}，"
          f"stall={args.stall}，ncu={'on' if not args.no_ncu else 'off'}）...")
    res = opt.optimize(args.op, init_code=init_code,
                       opt_rounds=args.opt_rounds, stall_limit=args.stall,
                       improve_min=args.improve_min,
                       use_ncu=not args.no_ncu)
    if not res.get("ok"):
        print(f"✗ 优化失败: {res.get('error')}")
        return 1

    print("\n" + "=" * 58)
    print(f"{'round':>5}{'best_ms':>11}{'speedup':>9}  note")
    for c in res["curve"]:
        print(f"{c['round']:>5}{c['ms']:>11.4f}{c['speedup']:>9.3f}  {c['note']}")
    print("-" * 58)
    print(f"基线 {res['baseline_ms']:.4f}ms → best {res['best_ms']:.4f}ms"
          f"  提升 {res['improve_pct']:.1f}%"
          + (" ✔" if res["improved"] else "（未提升：可能已接近极限）"))
    print(f"总耗时 {res['wall_s']}s | token {res['total_tokens']}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
