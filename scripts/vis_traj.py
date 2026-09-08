#!/usr/bin/env python3
"""轨迹可视化：把 results/traj_*.jsonl 变成可读的复盘报告。

单个轨迹：总览(成功/轮数/token/耗时/RAG/回灌/竞速命中) + 逐轮表(status/err/token)
           + 中途状态分布 + 可选显示每轮代码/反馈
聚合统计：扫全部轨迹 → 按 op 的成功率/平均轮数/状态分布（复盘"为什么绕 N 轮"）

用法：
    python scripts/vis_traj.py --latest            # 最新一条轨迹
    python scripts/vis_traj.py results/traj_x.jsonl   # 指定
    python scripts/vis_traj.py --op matmul          # 某算子最新
    python scripts/vis_traj.py --agg                # 全部聚合统计
    python scripts/vis_traj.py <p> --show-code --show-feedback
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_traj(path: str) -> tuple[dict, list[dict]]:
    summary, steps = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "summary" in obj:
                summary = obj["summary"]
            else:
                steps.append(obj)
    return summary, steps


def _round_status_ok(status: str) -> bool:
    return status == "pass"


def print_one(summary: dict, steps: list[dict], path: str,
              show_code: bool = False, show_feedback: bool = False) -> None:
    op = summary.get("op", "?")
    success = summary.get("success")
    print("=" * 64)
    print(f"轨迹: {os.path.basename(path)}")
    print(f"算子: {op:<14} 结果: {'✔ 通过' if success else '✗ 未通过'}"
          f"  轮数: {summary.get('rounds_used')}  "
          f"耗时: {summary.get('wall_s')}s")
    tok = summary.get("total_tokens") or {}
    print(f"token: prompt={tok.get('prompt', 0)} completion={tok.get('completion', 0)}"
          + (f"  | RAG={summary.get('memory_used')}" if summary.get("memory_used") else "")
          + (f"  | fix_refs={summary.get('fix_refs_used')}"
             if summary.get("fix_refs_used") else "")
          + (f"  | cache_hits={summary.get('cache_hits')}"
             if summary.get("cache_hits") else "")
          + (f"  | interrupted={summary.get('interrupted')}"
             if summary.get("interrupted") else ""))
    if summary.get("final_max_abs_err") is not None:
        print(f"末轮 err: {summary['final_max_abs_err']:.3e}"
              + (f"  | perf: {summary.get('final_speedup_vs_eager')}x"
                 if summary.get("final_speedup_vs_eager") else ""))
    print("-" * 64)
    print(f"{'r':>2} {'status':<18}{'err':>11}{'tok':>7}{'t(s)':>6}  reply")
    counter = collections.Counter()
    for st in steps:
        status = st.get("status", "?")
        counter[status] += 1
        ok = "✔" if _round_status_ok(status) else "✗"
        err = st.get("max_abs_err")
        err_s = f"{err:.2e}" if err is not None else "-"
        tok_s = (st.get("prompt_tokens", 0) + st.get("completion_tokens", 0))
        print(f"{ok} {st.get('round', 0):>2} {status:<18}{err_s:>11}"
              f"{tok_s:>7}{st.get('wall_s', 0):>6.1f}  reply={st.get('reply_len', 0)}")
        if show_feedback and st.get("feedback"):
            fb = st["feedback"].replace("\n", " ⏎ ")
            print(f"      └ feedback: {fb[:150]}{'…' if len(fb) > 150 else ''}")
        if show_code and st.get("code"):
            lines = st["code"].strip().splitlines()
            shown = lines[:60]
            print("      └ code:")
            for ln in shown:
                print(f"        | {ln}")
            if len(lines) > 60:
                print(f"        | …({len(lines) - 60} 行略)")
    print("-" * 64)
    print("中途状态分布: " + ", ".join(f"{k}={v}" for k, v in counter.most_common()))
    print("=" * 64)


def agg_print(paths: list[str]) -> None:
    by_op: dict[str, list[dict]] = collections.defaultdict(list)
    for p in paths:
        summary, _ = load_traj(p)
        if summary:
            by_op.setdefault(summary.get("op", "?"), []).append(summary)
    if not by_op:
        print("没有可聚合的轨迹。")
        return
    print(f"扫描 {len(paths)} 个轨迹文件，有效 {sum(len(v) for v in by_op.values())} 条\n")
    print(f"{'op':<18}{'n':>3}{'通过':>6}{'成功率':>9}{'均轮':>7}{'均token':>10}"
          f"{'末状态top':>14}")
    for op in sorted(by_op):
        ss = by_op[op]
        n = len(ss)
        succ = sum(1 for s in ss if s.get("success"))
        avg_rounds = sum(s.get("rounds_used", 0) for s in ss) / max(n, 1)
        avg_tok = sum((s.get("total_tokens") or {}).get("completion", 0)
                      for s in ss) / max(n, 1)
        last = collections.Counter(s.get("final_status", "?") for s in ss).most_common(1)
        print(f"{op:<18}{n:>3}{succ:>6}{succ / n * 100:>8.0f}%{avg_rounds:>7.2f}"
              f"{avg_tok:>10.0f}{last[0][0] if last else '-':>14}")
    total = sum(len(v) for v in by_op.values())
    all_s = [s for v in by_op.values() for s in v]
    n_ok = sum(1 for s in all_s if s.get("success"))
    print("-" * 64)
    print(f"总计: {n_ok}/{total} 通过 ({n_ok / max(total, 1) * 100:.0f}%)")

    # 全程状态分布（跨所有 step）
    dist = collections.Counter()
    for p in paths:
        _, steps = load_traj(p)
        for st in steps:
            dist[st.get("status", "?")] += 1
    if dist:
        print("全部 step 状态分布: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))


def find_latest(op: str | None = None) -> str | None:
    cands = []
    if os.path.isdir(RESULTS):
        for fn in sorted(os.listdir(RESULTS)):
            if fn.startswith("traj_") and fn.endswith(".jsonl"):
                if op and not fn.startswith(f"traj_{op}_"):
                    continue
                cands.append(os.path.join(RESULTS, fn))
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="轨迹 jsonl 路径；缺省用 --latest")
    ap.add_argument("--latest", action="store_true", help="用最新一条轨迹")
    ap.add_argument("--op", help="按算子过滤(--latest/--agg 时)")
    ap.add_argument("--agg", action="store_true", help="聚合全部轨迹统计")
    ap.add_argument("--show-code", action="store_true")
    ap.add_argument("--show-feedback", action="store_true")
    args = ap.parse_args()

    if args.agg:
        paths = []
        if os.path.isdir(RESULTS):
            paths = [os.path.join(RESULTS, fn)
                     for fn in sorted(os.listdir(RESULTS))
                     if fn.startswith("traj_") and fn.endswith(".jsonl")]
            if args.op:
                paths = [p for p in paths
                         if os.path.basename(p).startswith(f"traj_{args.op}_")]
        agg_print(paths)
        return 0

    path = args.path
    if not path:
        path = find_latest(args.op)
        if not path:
            print("没有找到轨迹（results/traj_*.jsonl）。先跑一次 run_agent。")
            return 1
        print(f"(默认: 最新轨迹 {path})\n")
    summary, steps = load_traj(path)
    print_one(summary, steps, path,
              show_code=args.show_code, show_feedback=args.show_feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
