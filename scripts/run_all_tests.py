#!/usr/bin/env python3
"""一把梭测试：汇总运行全部单元/自测脚本，打印通过率并返回退出码。

测试按是否需要 GPU / torch 分三组：
  core  —— 纯 stdlib（不 import torch）：error_parser / static_check / memory
  agent —— import torch+triton 但**不跑 kernel**（fake executor）：race / failure_memory
  gpu   —— 真实 GPU 沙箱：executor（会真跑 vector_add + do_bench）

用法：
    python scripts/run_all_tests.py            # 默认全跑（含 GPU 的 executor）
    python scripts/run_all_tests.py --ci       # 非 GPU：core + agent（供 CI）
    python scripts/run_all_tests.py --core     # 只 core（连 torch 都不用装）
    python scripts/run_all_tests.py --skip gpu # 跳过某组
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (script, group)
TESTS = {
    # core：纯 stdlib，无需 torch/triton
    "error_parser": ("scripts/test_error_parser.py", "core"),
    "static_check": ("scripts/test_static_check.py", "core"),
    "ncu_format": ("scripts/test_ncu_format.py", "core"),
    "prompts": ("scripts/test_prompts.py", "core"),
    # agent：import torch(+triton)，但不跑 kernel(fake executor / temp memory / KB 适配)
    "memory": ("scripts/test_memory.py", "agent"),
    "race": ("scripts/test_race.py", "agent"),
    "failure_memory": ("scripts/test_failure_memory.py", "agent"),
    "kernelbench": ("scripts/test_kernelbench.py", "agent"),
    "opt_format": ("scripts/test_opt_format.py", "agent"),
    "opt_beam": ("scripts/test_opt_beam.py", "agent"),
    "report_html": ("scripts/test_report_html.py", "agent"),
    # gpu：真实 GPU 沙箱
    "executor": ("scripts/test_executor.py", "gpu"),
}
GROUPS = ("core", "agent", "gpu")


def run_one(name: str, script: str) -> tuple[bool, float, str]:
    path = os.path.join(ROOT, script)
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=600, cwd=ROOT)
        ok = proc.returncode == 0
        tail = (proc.stdout or proc.stderr).strip().splitlines()
        note = tail[-1][:70] if tail else ""
        return ok, time.time() - t0, note
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "TIMEOUT>600s"
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - t0, f"ERR {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci", action="store_true", help="非 GPU：core+agent")
    ap.add_argument("--core", action="store_true", help="只 core（无需 torch）")
    ap.add_argument("--agent", action="store_true", help="只 agent")
    ap.add_argument("--gpu", action="store_true", help="只 gpu")
    ap.add_argument("--skip", choices=GROUPS, action="append", default=[],
                    help="跳过的组（可多次）")
    ap.add_argument("--verbose", action="store_true", help="失败时打印完整输出")
    args = ap.parse_args()

    if args.core:
        groups = ["core"]
    elif args.agent:
        groups = ["agent"]
    elif args.gpu:
        groups = ["gpu"]
    elif args.ci:
        groups = ["core", "agent"]
    else:
        groups = list(GROUPS)
    groups = [g for g in groups if g not in args.skip]

    selected = {n: s for n, (s, g) in TESTS.items() if g in groups}
    if not selected:
        print("没有要跑的测试。")
        return 0

    print(f"== 一把梭测试（{', '.join(groups)} 组，共 {len(selected)} 项）==")
    results = []
    for name, script in selected.items():
        ok, sec, note = run_one(name, script)
        results.append((name, ok, sec, note))
        mark = "✔" if ok else "✗"
        print(f"[{mark}] {name:<16} {sec:6.1f}s  {note}")

    n_ok = sum(1 for _, ok, *_ in results if ok)
    print("-" * 60)
    print(f"通过 {n_ok}/{len(results)}"
          + (" ✔" if n_ok == len(results) else " ✗"))
    if n_ok != len(results) and args.verbose:
        for name, ok, sec, note in results:
            if not ok:
                path = os.path.join(ROOT, TESTS[name][0])
                proc = subprocess.run([sys.executable, path], capture_output=True,
                                      text=True, cwd=ROOT)
                print(f"\n===== {name} 完整输出 =====")
                print((proc.stdout or "") + (proc.stderr or ""))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
