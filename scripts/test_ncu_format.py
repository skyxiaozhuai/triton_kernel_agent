#!/usr/bin/env python3
"""ncu_profiler.format_feedback 纯逻辑单测（不调用 ncu/不需要 GPU）。

验证三类 roofline 诊断分支：memory-bound / compute-bound / under-utilized，
以及耗时纳秒→微秒换算。
用法: python scripts/test_ncu_format.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tools import ncu_profiler as N  # noqa: E402


def prof(dram, sm, warps, dur_ns, l1=None, l2=None):
    metrics = {
        "gpu__time_duration.sum": {"value": dur_ns, "unit": "nsecond"},
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": {"value": dram, "unit": "%"},
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": {"value": sm, "unit": "%"},
        "sm__warps_active.avg.pct_of_peak_sustained_active": {"value": warps, "unit": "%"},
    }
    if l1 is not None:
        metrics["l1tex__t_sector_hit_rate.pct"] = {"value": l1, "unit": "%"}
    if l2 is not None:
        metrics["lts__t_sector_hit_rate.pct"] = {"value": l2, "unit": "%"}
    return {"kernel_name": "test_kernel", "block": "(128, 1, 1)",
            "grid": "(1, 1, 1)", "metrics": metrics}


def main() -> int:
    failed = 0

    def check(label, cond):
        nonlocal failed
        print(f"{'✔' if cond else '✗'} {label}")
        if not cond:
            failed += 1

    # memory-bound：DRAM 高、SM 低
    txt = N.format_feedback(prof(91, 4, 83, 71420))
    check("memory-bound 诊断", "memory-bound" in txt)
    check("耗时换算 ns->us (71420ns=71.42us)", "71.42 us" in txt)
    # compute-bound：SM 高、DRAM 低
    txt2 = N.format_feedback(prof(40, 75, 90, 100_000))
    check("compute-bound 诊断", "compute-bound" in txt2)
    # under-utilized：都低 + 占用率低
    txt3 = N.format_feedback(prof(5, 13, 12, 13_500))
    check("under-utilized 诊断", "under-utilized" in txt3)
    # 接近带宽上限提示
    check("≥90% DRAM 提示", "≥90%" in txt)
    # cache 命中率行 + L2 高诊断
    txt4 = N.format_feedback(prof(85, 8, 80, 70_000, l1=0, l2=92))
    check("cache 命中率行展示", "L1=0.0%  L2=92.0%" in txt4)
    check("L2 高命中诊断", "L2 命中率高(92%)：数据复用较好" in txt4)
    print("-" * 50)
    print("ncu format 单测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
