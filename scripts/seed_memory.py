#!/usr/bin/env python3
"""把历史成功轨迹灌入 RAG 经验库（memory）作为种子数据。

扫描 results/traj_*.jsonl：对 summary.success=True 的轨迹，
取最后一轮通过(pass)的代码 → memory.add_success(op, code)。
同 op 多条只保留最近一条（add_success 覆盖）。

用法: python scripts/seed_memory.py
纯本地，无 GPU / LLM / 网络。
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import memory  # noqa: E402


def _last_pass_code(steps: list[dict]) -> str | None:
    for st in steps:
        if st.get("status") == "pass" and st.get("code"):
            return st["code"]
    return steps[-1].get("code") if steps else None


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "results", "traj_*.jsonl")))
    if not files:
        print("results/ 下没有轨迹文件（先跑过 agent 才有）。")
        return 1

    seen: dict[str, int] = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            summary = json.loads(lines[0]).get("summary", {})
            steps = [json.loads(l) for l in lines[1:]]
        except Exception:  # noqa: BLE001
            continue
        if not summary.get("success") or not steps:
            continue
        code = _last_pass_code(steps)
        if not code:
            continue
        op = summary["op"]
        memory.add_success(op, code)   # 同 op 覆盖为最近一次
        seen[op] = seen.get(op, 0) + 1

    print(f"灌入成功记录（按 op 覆盖后剩 {len(memory.list_all())} 条）:")
    for r in memory.list_all():
        refs = memory.retrieve(r["op"], k=10)
        print(f"  {r['op']} [{r['category']}] added={r['added_at']} "
              f"| 同 category 可参考(非自身): {len(refs)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
