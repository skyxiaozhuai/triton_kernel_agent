"""softmax —— 二维逐行 softmax（经典 reduce + 数值稳定）。

y[m, n] = exp(x[m, n] - max_n(x[m, :])) / sum_n(exp(x[m, :] - max_n(x[m, :])))

考察点：axis=1 归约、为数值稳定先减 row max、mask 处理。
冒烟: python -m benchmarks.ops.softmax
"""
import torch
import triton
import triton.language as tl

OP_NAME = "softmax"

# 数值对齐容差（softmax 输出在 [0,1]）：fp32 精确、fp16 放宽
TOL32 = {"rtol": 1e-4, "atol": 1e-5}
TOL16 = {"rtol": 1e-2, "atol": 1e-3}

OP_META = {
    "name": OP_NAME,
    "category": "softmax / row-reduce",
    "difficulty": "medium",
    "dtype": "float32 / float16",
    "signature": "y = softmax(x)   # x, y: float32 [M, N]",
    "description": (
        "Row-wise softmax of a 2-D tensor x of shape [M, N]: "
        "y[m, n] = exp(x[m, n] - r_m) / sum_n(exp(x[m, :] - r_m)), "
        "where r_m = max over n of x[m, :]. "
        "Subtracting the row max first is REQUIRED for numerical stability. "
        "N is the contiguous (last) dimension. "
        "One program may handle one or several rows; N may not divide evenly, "
        "so use an offset mask `offs < N`."
    ),
    "notes": "输出与输入同 shape；务必先减 row max 保证稳定。",
    "launch_sig": "launch(x: Tensor, M: int, N: int) -> Tensor   # M/N 由 meta 提供",
}

# 默认规模：够小，GTX1650 冒烟无压力；服务器可用更大 shape
DEFAULT_M, DEFAULT_N = 1024, 1024


def _make_case(m, n, device, dtype):
    x = torch.randn(m, n, device=device, dtype=dtype)
    return {"x": x, "meta": {"M": m, "N": n}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(DEFAULT_M, DEFAULT_N, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非整除/极小）。dtype=None 表示都测。"""
    specs = [(torch.float32, ((DEFAULT_M, DEFAULT_N), (1024, 1000), (31, 127))),
             (torch.float16, ((1024, 1024), (32, 128)))]
    return [_make_case(m, n, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for m, n in shapes]


def golden(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


@triton.jit
def _softmax_kernel(x, y, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)          # 每 program 处理一行
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    row_ptr = x + row * stride_m
    xrow = tl.load(row_ptr + offs, mask=mask, other=-float("inf"))
    xf = xrow.to(tl.float32)                 # 内部提升 fp32：数值稳定/精度

    xmax = tl.max(xf, axis=0)                # 减 row max 保数值稳定
    num = tl.exp(xf - xmax)
    denom = tl.sum(num, axis=0)
    res = (num / denom).to(y.dtype.element_ty)   # 按输出 dtype 截断

    out_ptr = y + row * stride_m
    tl.store(out_ptr + offs, res, mask=mask)


def reference_triton(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(n)
    _softmax_kernel[(m,)](x, y, m, n, x.stride(0), BLOCK_N=BLOCK_N)
    return y


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
    print(f"[{OP_NAME}] shape={tuple(x.shape)}  meta={args['meta']}")
    print(f"  reference_triton vs golden  allclose: {check(y, g)}")
    print("冒烟通过 ✔")
