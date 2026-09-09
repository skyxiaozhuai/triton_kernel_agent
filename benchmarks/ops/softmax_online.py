"""softmax_online —— 行 softmax 的 ONLINE 单遍实现（分块流式 running max）。

y[m, n] = exp(x[m, n] - M_m) / L_m,  M_m = max_n(x[m,:]), L_m = sum_n(exp(x - M_m))

与"朴素两遍"(先整行求 max，再整行求 sum)不同，online 版在**单遍**里维护
running max M 与 running sum L：每读一块就
  M_new = max(M, block_max)
  L     = L * exp(M - M_new) + sum(exp(xb - M_new))
  M     = M_new
第一遍结束即得精确 (M, L)，再第二遍归一写出。数值上与朴素版等价，但可分块流式、
不要求先知道全局 max —— 这正是 flash attention 里 online softmax 的核心思想。

冒烟: python -m benchmarks.ops.softmax_online
"""
import torch
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "softmax_online"

# online 分块 rescaled 累加与 torch 微小顺序差；fp32 紧、fp16 放宽
TOL32 = {"rtol": 1e-4, "atol": 1e-5}
TOL16 = {"rtol": 1e-2, "atol": 1e-3}

OP_META = {
    "name": OP_NAME,
    "category": "row-softmax / online single-pass",
    "difficulty": "hard",
    "dtype": "float32 / float16",
    "signature": "y = softmax_online(x)   # x, y: [M, N]，逐行 softmax",
    "description": (
        "Row-wise softmax with the ONLINE (single-pass) formulation — the same "
        "idea as flash attention's online softmax. Each program handles one row "
        "(or a chunk of rows); process the row over N in BLOCK-sized steps "
        "(loop `for k in range(tl.cdiv(N, BLOCK_N))`), maintaining a running "
        "max M and running sum L in fp32 scalars:\n"
        "  M_new = max(M, max(xb));\n"
        "  L = L * exp(M - M_new) + sum(exp(xb - M_new));  M = M_new;\n"
        "After that first pass M/L are EXACT (rescaled online, so no precision "
        "loss vs two-pass). Then do a SECOND pass loading the row again to store "
        "y = exp(x - M) / L. Do NOT compute the full-row max before summing "
        "(that would be two reductions, not online). Mask with `offs < N`; "
        "load masked lanes as -inf (exp(-inf)=0 keeps them out of max/sum)."
    ),
    "notes": "数值稳定 = 减 running max；fp32 累加；N 分块循环、无需一次装整行。",
    "launch_sig": "launch(x: Tensor, M: int, N: int) -> Tensor  # x:[M,N]，M/N 由 meta 提供",
}

# 主 shape：N 偏大让分块循环真正生效
DEFAULT_M, DEFAULT_N = 512, 2048


def current_shape() -> tuple[int, int]:
    """主 case (M, N)：默认 512×2048；可用 env OP_SHAPE=M,N 覆盖。"""
    return get_op_shape(OP_NAME, (DEFAULT_M, DEFAULT_N))


def _make_case(m, n, device, dtype):
    x = torch.randn(m, n, device=device, dtype=dtype)
    return {"x": x, "meta": {"M": m, "N": n}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    m, n = current_shape()
    return _make_case(m, n, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非整除/极小）。dtype=None 表示都测。"""
    m, n = current_shape()
    specs = [(torch.float32, ((m, n), (257, 1000), (31, 17))),
             (torch.float16, ((128, 512), (17, 128)))]
    return [_make_case(mm, nn, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for mm, nn in shapes]


def golden(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)              # 每 program 处理一行
    base = x_ptr + row * stride_m
    n_blk = tl.cdiv(N, BLOCK_N)

    # pass 1: online 单遍求精确 (M, L)
    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)
    for k in range(0, n_blk):
        offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        xb = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(xb, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(xb - m_new), axis=0)
        m = m_new

    # pass 2: 归一写出
    for k in range(0, n_blk):
        offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        xb = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        res = tl.exp(xb - m) / l
        tl.store(y_ptr + row * stride_m + offs,
                 res.to(y_ptr.dtype.element_ty), mask=mask)


def reference_triton(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK_N = 512
    _softmax_online_kernel[(m,)](x, y, m, n, x.stride(0), BLOCK_N=BLOCK_N)
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
    err = (y - g).abs().max().item()
    print(f"[{OP_NAME}] x={tuple(x.shape)}")
    print(f"  reference(online) vs golden  allclose: {check(y, g)}  "
          f"max_abs_err={err:.3e}")
    print("冒烟通过 ✔")
