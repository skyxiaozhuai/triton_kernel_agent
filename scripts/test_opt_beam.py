#!/usr/bin/env python3
"""opt_loop 的 beam + prescribe 纯逻辑单测（不需要 LLM/GPU）。

验证对齐官方 beam / BottleneckAnalyzer 的轻量实现：
  - _fallback_directions 按 roofline 文本给确定性方向
  - _parse_directions 从 LLM 诊断输出解析候选方向
  - _opt_user 带 direction 时渲染"本候选方向"
  - 默认构造保持 beam=1 / prescribe=False（向后兼容 optimize()）
用法: python scripts/test_opt_beam.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.opt_loop import KernelOptimizer as O  # noqa: E402


class _Dummy:
    OP_META = {"name": "vector_add", "signature": "y=f(x)", "dtype": "fp32",
               "notes": None, "launch_sig": None}


def main() -> int:
    failed = 0

    def check(label, cond):
        nonlocal failed
        print(f"{'✔' if cond else '✗'} {label}")
        if not cond:
            failed += 1

    # --- 默认向后兼容 ---
    opt = O(verbose=False)
    check("默认 beam=1/prescribe=False(兼容旧行为)",
          opt.beam_width == 1 and opt.prescribe is False)
    check("可开 beam/prescribe", O(verbose=False, beam_width=3,
                                   prescribe=True).prescribe is True)

    # --- _fallback_directions：按 roofline 文本 ---
    mem = O._fallback_directions("诊断: memory-bound, DRAM 91% 高")
    check("memory-bound → 访存方向优先", mem[0].startswith("优化访存") and len(mem) >= 2)
    comp = O._fallback_directions("compute-bound, SM 75%")
    check("compute-bound → tile 方向优先", comp[0].startswith("增大/重排") and len(comp) >= 2)
    gen = O._fallback_directions(None)
    check("无剖析 → 通用 3 方向", len(gen) == 3)

    # --- _parse_directions：LLM 诊断 → 方向列表 ---
    t = ("瓶颈: 访存受限。\n1) 增大 BLOCK tile 并配 num_warps\n"
         "2) 加深 num_stages 流水\n- 优化 load 向量化\n3) 其它")
    p = O._parse_directions(t)
    check("去序号/标题，限 3 条", len(p) == 3 and p[0].startswith("增大")
          and p[2].startswith("优化"))
    check("空输出 → fallback 兜底", len(O._parse_directions("")) == 3)

    # --- _opt_user：direction 强化注入 ---
    u = opt._opt_user(_Dummy(), "import torch\ndef launch(): ...", 0.1, 1.0, [],
                      None, direction="增大 tile")
    check("_opt_user 渲染本候选方向", "[本候选方向]" in u and "增大 tile" in u)
    u0 = opt._opt_user(_Dummy(), "code", 0.1, 1.0, [], None, direction=None)
    check("无 direction 不注入", "[本候选方向]" not in u0)

    print("-" * 50)
    print("opt beam 单测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
