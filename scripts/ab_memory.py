#!/usr/bin/env python3
"""RAG/经验库 A/B：同一算子在有/无 memory 检索下的收敛对比（受控实验）。

对指定 op，分别以 memory_mode on / off 各跑 repeats 次（并行），
对比 成功率 / 平均轮数 / token 成本 / 耗时，输出 markdown 表并存 json。

注意（诚实边界）：
- 简单 op（elementwise，通常 1 轮过）memory 增益≈0，属预期；
  "记忆有效"的证据需要 medium/hard 同族 op 才明显 —— 这类在服务器跑
  （KernelBench 难算子 / 融合归约等）。
- 本脚本本地用于验证流程 + 留复现；正式 A/B 建议在服务器。

用法：
    python scripts/ab_memory.py --op relu --repeats 2 --rounds 3
    python scripts/ab_memory.py --op matmul --repeats 3 --rounds 5 --out results/ab_matmul.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from agent.loop import KernelAgent  # noqa: E402
from benchmarks import ops_registry  # noqa: E402


def run_once(op: str, memory_on: bool, rounds: int) -> dict:
    a = KernelAgent(max_rounds=rounds, verbose=False, memory_mode=memory_on)
    s, _ = a.run(op, save=False)
    s["_memory_on"] = memory_on
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="relu")
    ap.add_argument("--repeats", type=int, default=2, help="每组重复次数")
    ap.add_argument("--rounds", type=int, default=3, help="每次 max_rounds")
    ap.add_argument("--out", default=None, help="结果 json 输出路径")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA（服务器）。本机 4G 只适合小 shape elementwise。")
        if args.op not in ("vector_add", "relu", "add_relu"):
            print("本机建议用 elementwise 小 op 试流程；难算子 A/B 放服务器。")
    if args.op not in ops_registry.list_ops():
        print(f"未知算子: {args.op}。可用: {ops_registry.list_ops()}")
        return 2

    jobs = ([("off", i) for i in range(args.repeats)]
            + [("on", i) for i in range(args.repeats)])
    t0 = time.time()
    results = []

    def _worker(tag_idx):
        tag, _i = tag_idx
        return run_once(args.op, memory_on=(tag == "on"), rounds=args.rounds)

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for s in ex.map(_worker, jobs):
            results.append(s)

    wall = time.time() - t0

    def _agg(tag: str) -> dict:
        ss = [s for s in results if s["_memory_on"] == (tag == "on")]
        n = len(ss)
        succ = sum(1 for s in ss if s["success"])
        avg_r = sum(s.get("rounds_used", 0) for s in ss) / max(n, 1)
        avg_tok = sum((s.get("total_tokens") or {}).get("completion", 0)
                      for s in ss) / max(n, 1)
        statuses = [s.get("final_status", "?") for s in ss]
        return {"tag": f"memory {tag}", "n": n, "succ": succ,
                "rate": succ / max(n, 1), "avg_rounds": avg_r,
                "avg_tok": round(avg_tok), "statuses": statuses,
                "runs": ss}

    off = _agg("off")
    on = _agg("on")
    print(f"\n== RAG A/B: op={args.op} | 每组 repeats={args.repeats} | "
          f"max_rounds={args.rounds} | 墙钟 {wall:.0f}s ==")
    print(f"{'组':<12}{'n':>3}{'通过':>5}{'成功率':>9}{'均轮':>7}{'均token':>10}")
    for g in (off, on):
        print(f"{g['tag']:<12}{g['n']:>3}{g['succ']:>5}"
              f"{g['rate'] * 100:>8.0f}%{g['avg_rounds']:>7.2f}{g['avg_tok']:>10}")
    for g in (off, on):
        print(f"  {g['tag']}: 末状态 {g['statuses']}")
    if off["rate"] and on["rate"] and off["avg_rounds"] != on["avg_rounds"]:
        print(f"  均轮差: off={off['avg_rounds']:.2f} vs on={on['avg_rounds']:.2f} "
              f"(记忆 {'更省' if on['avg_rounds'] < off['avg_rounds'] else '不省'} 轮)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"op": args.op, "repeats": args.repeats, "rounds": args.rounds,
                   "wall_s": round(wall, 1),
                   "off": {k: v for k, v in off.items() if k != "runs"},
                   "on": {k: v for k, v in on.items() if k != "runs"},
                   "detail_runs": results}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"结果已存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
