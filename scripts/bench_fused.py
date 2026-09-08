#!/usr/bin/env python3
"""融合算子 vs 分离算子性能对比（借鉴 KernelAgent "融合省内存流量" 理念的量化）。

对每个 fused op，对比：
  融合版 = 该 op 的 reference_triton（单轮，中间结果不落全局内存）
  分离版 = 用两个基础 kernel 串行（如 vector_add 写中间 + relu 读中间）
用 do_bench 测两版耗时，输出 fused/separate/加速比 表，并做数值一致性副检。

用法（在项目根，需 GPU）：
    python scripts/bench_fused.py            # 全部 fused op
    python scripts/bench_fused.py add_relu   # 指定
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

from benchmarks import ops_registry as R  # noqa: E402
from benchmarks.ops import relu, relu_sum, sum_1d, vector_add  # noqa: E402


def _strip(args: dict) -> dict:
    return {k: v for k, v in args.items() if k != "meta"}


# 每个 fused op 的"分离基线"：用基础 kernel 两步串行（含一次整张中间读写）
def _sep_add_relu(args: dict):
    m = _strip(args)
    return relu.reference_triton(vector_add.reference_triton(m["x1"], m["x2"]))


def _sep_relu_sum(args: dict):
    m = _strip(args)
    return sum_1d.reference_triton(relu.reference_triton(m["x"]))


SEPARATE = {
    "add_relu": _sep_add_relu,
    "relu_sum": _sep_relu_sum,
}


def bench_one(op_name: str, warmup: int = 20, rep: int = 100) -> dict:
    mod = R.get_op(op_name)
    args = mod.generate_inputs()          # fp32 主 case
    fused_fn = lambda: mod.reference_triton(**_strip(args))
    sep_fn = lambda: SEPARATE[op_name](args)

    # 数值一致性副检（融合版 == 分离版）
    y_fused = mod.reference_triton(**_strip(args))
    y_sep = SEPARATE[op_name](args)
    agree = bool(torch.allclose(y_fused.float(), y_sep.float(),
                                rtol=1e-2, atol=1e-2))

    fused_ms = float(do_bench(fused_fn, warmup=warmup, rep=rep))
    sep_ms = float(do_bench(sep_fn, warmup=warmup, rep=rep))
    return {"op": op_name,
            "fused_ms": round(fused_ms, 4),
            "separate_ms": round(sep_ms, 4),
            "speedup": round(sep_ms / fused_ms, 3),
            "agree": agree}


def main() -> int:
    if not torch.cuda.is_available():
        print("[WARN] 需要 CUDA。")
        return 2
    fused_ops = [n for n in R.list_ops() if R.get_op(n).OP_META.get("fused")]
    targets = [a for a in sys.argv[1:] if a in SEPARATE] or fused_ops
    targets = [t for t in targets if t in SEPARATE]
    if not targets:
        print(f"可用 fused op: {fused_ops}")
        return 2

    print(f"{'op':<10}{'fused(ms)':>12}{'separate(ms)':>14}{'加速比':>8}  数值一致")
    rows = []
    for op in targets:
        r = bench_one(op)
        rows.append(r)
        mark = "✔" if r["agree"] else "✗"
        print(f"{r['op']:<10}{r['fused_ms']:>12.4f}{r['separate_ms']:>14.4f}"
              f"{r['speedup']:>7.2f}x   {mark}")
    # 简洁结论
    for r in rows:
        print(f"\n[{r['op']}] 融合 {'快' if r['speedup'] > 1 else '慢'} "
              f"{abs(r['speedup'] - 1) * 100:.0f}% "
              f"({'快' if r['agree'] else '数值不一致!'}) "
              f"(separate={r['separate_ms']}ms -> fused={r['fused_ms']}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
