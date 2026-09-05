#!/usr/bin/env python3
"""批量评测：对多个算子跑 agent 闭环，汇总成结果表（支持多轮取成功率 / 性能 critic）。

用法（项目根目录）：
    python scripts/run_all.py                     # 全部算子，1 次，正确性
    python scripts/run_all.py --perf              # 正确性 + 性能 critic
    python scripts/run_all.py softmax matmul      # 指定算子
    python scripts/run_all.py --repeat 3 --perf   # 每算子 3 次(成功率) + 性能

汇总打印 markdown 表并保存 results/summary_<ts>.json。
注意：真实调用 LLM + GPU，会消耗少量费用与时间。
"""
import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton Kernel Agent 批量评测")
    parser.add_argument("ops", nargs="*", help="算子名；缺省跑全部")
    parser.add_argument("--repeat", type=int, default=1, help="每个算子重复次数(算成功率)")
    parser.add_argument("--rounds", type=int, default=6, help="每个 agent 最大轮数")
    parser.add_argument("--perf", action="store_true", help="开启性能 critic")
    parser.add_argument("--save-traj", action="store_true", help="同时保留每条轨迹 jsonl")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA")
        return 2

    from agent.loop import KernelAgent
    from benchmarks import ops_registry

    ops = args.ops or ops_registry.list_ops()
    print(f"评测: ops={ops} | repeat={args.repeat} | perf={args.perf} | "
          f"rounds={args.rounds}\n", flush=True)

    rows = []
    for op in ops:
        for trial in range(1, args.repeat + 1):
            agent = KernelAgent(max_rounds=args.rounds, perf_mode=args.perf,
                                verbose=False)
            summary, _steps = agent.run(op, save=args.save_traj)
            row = {"op": op, "trial": trial,
                   "success": summary["success"],
                   "rounds_used": summary["rounds_used"],
                   "wall_s": summary["wall_s"],
                   "tokens": summary["total_tokens"],
                   "final_status": summary["final_status"],
                   "final_max_abs_err": summary["final_max_abs_err"],
                   "final_speedup_vs_eager": summary.get("final_speedup_vs_eager")}
            rows.append(row)
            flag = "✔" if row["success"] else "✗"
            print(f"  [{op} #{trial}] {flag} rounds={row['rounds_used']} "
                  f"wall={row['wall_s']}s "
                  + (f"speedup={row['final_speedup_vs_eager']}x "
                     if row.get("final_speedup_vs_eager") else ""), flush=True)

    agg = defaultdict(list)
    for r in rows:
        agg[r["op"]].append(r)

    lines = ["| op | 次数 | 通过 | 平均轮数 | 平均token | 末轮err(中位) | 末轮speedup |",
             "|---|---|---|---|---|---|---|"]
    for op in ops:
        rs = agg[op]
        n = len(rs)
        ok = sum(r["success"] for r in rs)
        avg_round = sum(r["rounds_used"] for r in rs) / n
        avg_tok = sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rs) / n
        errs = sorted(r["final_max_abs_err"] for r in rs
                      if r["final_max_abs_err"] is not None)
        err = f"{errs[len(errs) // 2]:.2e}" if errs else "-"
        spds = [r["final_speedup_vs_eager"] for r in rs
                if r.get("final_speedup_vs_eager") is not None]
        spd = f"{max(spds)}x" if spds else "-"
        lines.append(f"| {op} | {n} | {ok}/{n} | {avg_round:.1f} | {avg_tok:.0f} "
                     f"| {err} | {spd} |")
    table = "\n".join(lines)
    print("\n" + table)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"timestamp": ts, "perf": args.perf, "rows": rows, "table": table}
    path = os.path.join(ROOT, "results", f"summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {path}")

    total_ok = sum(r["success"] for r in rows)
    print(f"总体: {total_ok}/{len(rows)} 通过")
    return 0 if total_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
