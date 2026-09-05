"""性能基准工具 —— 性能闭环第一块。

对注册表里的 op，用 triton.testing.do_bench 测三者耗时(ms，取多次最小)：
  - reference_triton : 手写/agent 生成的 Triton kernel
  - eager            : op.golden（等价 torch 原生实现）
  - torch.compile    : 对 golden 做 graph 编译后的版本
产出加速比：triton vs eager、triton vs torch.compile。

注意：本机是 GTX1650(sm_75) + 小 shape，数字仅供"链路验证/趋势"；
真实性能对比建议在服务器大 shape 上跑（shape 参数化是后续升级项）。
"""
from __future__ import annotations

import torch

from benchmarks import ops_registry


def _bench_fn_ms(fn, warmup_ms: int = 50, rep_ms: int = 200) -> float:
    from triton.testing import do_bench
    return do_bench(fn, warmup=warmup_ms, rep=rep_ms)


def bench_op(op_name: str, with_compile: bool = True,
             reps: int = 3) -> dict:
    """对一个 op 跑基准，返回含各实现耗时的 dict。"""
    op = ops_registry.get_op(op_name)
    args = op.generate_inputs()
    ins = {k: v for k, v in args.items() if k != "meta"}

    def eager_fn():
        return op.golden(**ins)

    def triton_fn():
        return op.reference_triton(**ins)

    eager_ms = min(_bench_fn_ms(eager_fn) for _ in range(reps))
    triton_ms = min(_bench_fn_ms(triton_fn) for _ in range(reps))
    res: dict = {"op": op_name, "meta": args.get("meta"),
                 "eager_ms": round(eager_ms, 4),
                 "triton_ms": round(triton_ms, 4),
                 "compile_ms": None}

    if with_compile:
        try:
            compiled = torch.compile(op.golden)
            compiled(**ins)          # 首次触发编译
            torch.cuda.synchronize()
            res["compile_ms"] = round(
                min(_bench_fn_ms(lambda: compiled(**ins)) for _ in range(reps)), 4)
        except Exception as e:  # noqa: BLE001
            res["compile_error"] = str(e)[:200]
    return res


def speedups(res: dict) -> dict:
    """由 bench_op 结果计算加速比。"""
    sp: dict = {}
    if res.get("triton_ms"):
        sp["vs_eager"] = round(res["eager_ms"] / res["triton_ms"], 3)
        if res.get("compile_ms"):
            sp["vs_compile"] = round(res["compile_ms"] / res["triton_ms"], 3)
    return sp


def format_table(results: list[dict]) -> str:
    """把多个 bench 结果格式化成 markdown 表格。"""
    lines = ["| op | eager(ms) | triton(ms) | compile(ms) | vs eager | vs compile |",
             "|---|---|---|---|---|---|"]
    for r in results:
        sp = speedups(r)
        cms = r.get("compile_ms")
        if cms is None:
            cms = r.get("compile_error", "NA")
        lines.append(
            f"| {r['op']} | {r['eager_ms']} | {r['triton_ms']} | {cms} "
            f"| {sp.get('vs_eager', '-')}x | {sp.get('vs_compile', '-')}x |")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自测：vector_add 快速跑一遍
    r = bench_op("vector_add", with_compile=False)
    print(r)
    print("speedups:", speedups(r))
