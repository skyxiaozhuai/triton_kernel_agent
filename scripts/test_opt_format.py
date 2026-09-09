#!/usr/bin/env python3
"""opt_loop 纯逻辑单测：attempt 窗口渲染 / AVOID 教训生成（不需要 LLM/GPU）。

验证对齐官方的两个机制：
  - _avoid_for('not_faster') 生成确定性 AVOID 教训
  - _fmt_attempts 把被拒尝试渲染成含(状态/ms/截断代码/AVOID)的文本块
用法: python scripts/test_opt_format.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.opt_loop import KernelOptimizer as O  # noqa: E402


def main() -> int:
    failed = 0

    def check(label, cond):
        nonlocal failed
        print(f"{'✔' if cond else '✗'} {label}")
        if not cond:
            failed += 1

    # --- _avoid_for：按 status 生成教训 ---
    a_nf = O._avoid_for("not_faster", "未更快")
    check("not_faster → 避免微调教训", "避免" in a_nf and "num_warps" in a_nf)
    a_err = O._avoid_for("error", "CUDA error: 越界")
    check("error → 取原因", "CUDA error" in a_err)

    # --- _fmt_attempts：渲染窗口（含截断代码 + AVOID）---
    # 长代码：头部正常、末尾放唯一标记 TAIL_MARK（应被 [:600] 截掉）
    long_code = "def launch(x):\n    return x\n" + ("#" * 800) + "\nTAIL_MARK"
    attempts = [{"round": 1, "status": "not_faster", "ms": 0.0752,
                 "code": long_code, "note": "未更快: 0.0752 ≥ best 0.0755",
                 "avoid": a_nf}]
    txt = O._fmt_attempts(attempts)
    check("渲染含 status/ms", "not_faster" in txt and "0.0752" in txt)
    check("渲染含代码块", "```python" in txt)
    check("代码已截断(尾部标记被切掉 + 有省略号)",
          "TAIL_MARK" not in txt and "..." in txt)
    check("渲染含 AVOID", "AVOID:" in txt)
    check("空窗口 → 空串", O._fmt_attempts([]) == "")

    print("-" * 50)
    print("opt format 单测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
