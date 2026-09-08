#!/usr/bin/env python3
"""KernelBench 适配层 CLI —— 让我们的 agent 跑官方 KernelBench 题目。

需要 GPU + 显存（多数 L1 默认 shape 是 A100 级），正式跑放在服务器。
用法（在项目根，KernelBench 根目录默认 /home/claude/agent_project/KernelBench，
可用 env KERNELBENCH_ROOT 覆盖）：
    python scripts/run_kernelbench.py --level 1 --list          # 列出 level1 题目
    python scripts/run_kernelbench.py --level 1 --id 19 --dry   # 只加载+打印规格(不跑)
    python scripts/run_kernelbench.py --level 1 --id 19         # agent 生成+判卷(真 GPU)
    python scripts/run_kernelbench.py --path /abs/KernelBench/KernelBench/level1/19_ReLU.py
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from benchmarks.kernelbench import problem as kbp  # noqa: E402


def _list_level(level: int, root) -> None:
    base = kbp.default_kernelbench_root() if root is None else root
    lvl = base / "KernelBench" / f"level{level}"
    if not lvl.is_dir():
        lvl = base / f"level{level}"
    if not lvl.is_dir():
        print(f"找不到 level{level} 目录: {lvl}")
        sys.exit(2)
    files = sorted(lvl.glob("*.py"))
    print(f"level{level} 共 {len(files)} 题：")
    for p in files:
        print(f"  {p.stem}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", help="题目 .py 绝对路径")
    ap.add_argument("--level", type=int, help="级别 1/2/3")
    ap.add_argument("--id", type=int, help="题号(如 19)")
    ap.add_argument("--root", default=None, help="KernelBench 根目录(默认 env KERNELBENCH_ROOT 或 /home/claude/agent_project/KernelBench)")
    ap.add_argument("--list", action="store_true", help="列出某 level 题目")
    ap.add_argument("--dry", action="store_true", help="只加载+打印规格预览，不跑 agent")
    ap.add_argument("--cases", type=int, default=2, help="判卷随机输入 case 数(默认2)")
    ap.add_argument("--rounds", type=int, default=6, help="最大轮数")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        if args.level is None:
            print("--list 需要 --level N")
            return 2
        _list_level(args.level, args.root)
        return 0

    if args.path:
        problem_path = args.path
    elif args.level and args.id:
        problem_path = str(kbp.resolve_problem(args.level, args.id, args.root))
    else:
        print("需要 --path 或 (--level + --id)")
        return 2
    if not os.path.exists(problem_path):
        print(f"题目不存在: {problem_path}")
        return 2

    prob = kbp.load_problem(problem_path)
    print(f"题目: {prob.name} (level {prob.level}) | {prob.path}")

    if args.dry:
        print("\n===== 规格预览（给 LLM 的内容）=====")
        spec = prob.spec_text()
        print(spec[:5000] + ("\n…(截断)" if len(spec) > 5000 else ""))
        print("\n[dry] 预检通过：题目可加载、规格可生成。真跑需 GPU/显存(服务器)。")
        return 0

    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA 才能判卷(服务器)。用 --dry 先做规格预检。")
        return 2

    from agent.kb_loop import KernelBenchAgent
    agent = KernelBenchAgent(max_rounds=args.rounds, max_tokens=args.max_tokens,
                             verbose=not args.quiet)
    summary, _steps = agent.run_problem(problem_path, num_cases=args.cases)

    print("=" * 56)
    print(f"结果: {'✔ 通过' if summary['success'] else '✗ 未通过'}")
    print(f"  题目       : {summary['op']}")
    print(f"  使用轮数   : {summary['rounds_used']}")
    print(f"  总耗时     : {summary['wall_s']} s")
    print(f"  token      : {summary['total_tokens']}")
    print(f"  末轮状态   : {summary['final_status']}"
          + (f" (err={summary['final_max_abs_err']:.3e})"
             if summary.get("final_max_abs_err") is not None else ""))
    print("=" * 56)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
