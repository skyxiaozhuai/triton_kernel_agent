"""sum_1d —— 一维全量求和（跨 block 归约，两阶段）。

y = sum_i x[i]   for x:[N]

考察点：单 block 只能归约自己那段，跨 block 归约需要
第一阶段 partial 部分和 + 第二阶段对 partial 再归约。
冒烟: python -m benchmarks.ops.sum_1d
"""
import torch
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "sum_1d"

# fp32 求和顺序差异随 N 累积，容差放宽到相对 1e-3；fp16 另设
TOL32 = {"rtol": 1e-3, "atol": 1e-2}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}

OP_META = {
    "name": OP_NAME,
    "category": "reduction (cross-block)",
    "difficulty": "medium",
    "dtype": "float32 / float16",
    "signature": "y = sum_1d(x)   # x: float32 [N] -> y: float32 [1]",
    "description": (
        "Sum of all elements of a 1-D tensor x of length N: y = sum_i x[i]. "
        "A single program can only reduce the elements it loads, so a full "
        "reduction needs TWO stages: (1) each program reduces its BLOCK-sized "
        "chunk into a partial result; (2) one final program reduces the "
        "partial results. Use masking since N may not divide by BLOCK."
    ),
    "notes": "输出是长度为 1 的 tensor。",
    "launch_sig": "launch(x: Tensor, N: int) -> Tensor   # N 是元素总数，由 meta 提供",
}

BLOCK = 1024
DEFAULT_N = 1 << 20


def current_shape() -> int:
    """主 case N：默认 2^20；可用 env OP_SHAPE=N 调大。"""
    return get_op_shape(OP_NAME, (DEFAULT_N,))[0]


def _make_case(n, device, dtype):
    x = torch.randn(n, device=device, dtype=dtype)
    return {"x": x, "meta": {"N": n}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(current_shape(), device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 多组 N（fp16 用中规模避免误差放大）。dtype=None 都测。"""
    specs = [(torch.float32, (current_shape(), 1_000_003, 1000)),
             (torch.float16, (1 << 18, 1_000_003))]
    return [_make_case(n, device, dt)
            for dt, ns in specs if dtype is None or dt == dtype
            for n in ns]


def golden(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x).reshape(1)


@triton.jit
def _sum_stage1(x, partial, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)  # fp16 也提升 fp32 累加
    tl.store(partial + pid, tl.sum(vals, axis=0))


@triton.jit
def _sum_stage2(partial, out, nparts, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < nparts
    vals = tl.load(partial + offs, mask=mask, other=0.0)
    tl.store(out, tl.sum(vals, axis=0))


def reference_triton(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    grid1 = (triton.cdiv(n, BLOCK),)
    partial = torch.empty(grid1[0], device=x.device, dtype=torch.float32)
    _sum_stage1[grid1](x, partial, n, BLOCK=BLOCK)

    out = torch.empty(1, device=x.device, dtype=torch.float32)
    nparts = grid1[0]
    BLOCK2 = triton.next_power_of_2(nparts)
    _sum_stage2[(1,)](partial, out, nparts, BLOCK=BLOCK2)
    return out.to(x.dtype)   # fp16 输入 → 内部 fp32 算完再截断回 fp16


def check(out: torch.Tensor, ref: torch.Tensor,
          rtol: float | None = None, atol: float | None = None) -> bool:
    is16 = (out.dtype == torch.float16) or (ref.dtype == torch.float16)
    base = TOL16 if is16 else TOL32
    rtol = base["rtol"] if rtol is None else rtol
    atol = base["atol"] if atol is None else atol
    return bool(torch.allclose(out, ref, rtol=rtol, atol=atol))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA 才能冒烟"
    args = generate_inputs()
    x = args["x"]
    y = reference_triton(x)
    g = golden(x)
    print(f"[{OP_NAME}] N={args['meta']['N']}")
    print(f"  reference_triton vs golden  allclose: {check(y, g)}")
    print("冒烟通过 ✔")
