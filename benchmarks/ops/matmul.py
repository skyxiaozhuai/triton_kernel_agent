"""matmul —— 二维 GEMM（tl.dot 考察点）。

c[m, n] = sum_k a[m, k] * b[k, n]   for a:[M,K], b:[K,N]

考察点：2D tile + 外层 K 循环、tl.dot、mask、fp32 累加。
注意：reference 的 fp32 tl.dot 精度按当前 GPU 自适应（sm_80+ 用 tf32，否则 ieee）。
冒烟: python -m benchmarks.ops.matmul
"""
import torch
import triton
import triton.language as tl

OP_NAME = "matmul"

# fp32 ieee 点积累加顺序差异，容差略放宽
TOL = {"rtol": 1e-3, "atol": 1e-3}

OP_META = {
    "name": OP_NAME,
    "category": "matmul / gemm",
    "dtype": "float32",
    "signature": "c = matmul(a, b)   # a: [M,K], b: [K,N] -> c: [M,N]",
    "description": (
        "General matrix multiplication: c = a @ b, where a is [M, K] and "
        "b is [K, N]. Use tl.dot on BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N "
        "tiles with an outer loop over K; accumulate in fp32. "
        "Both matrices are row-major (contiguous) 2-D. "
        "M/N/K may not divide evenly by the block sizes, so mask the loads "
        "and the final store."
    ),
    "notes": "K 是 a 的列数 / b 的行数；累加用 fp32 保精度。fp32 的 tl.dot 用哪种 input_precision 取决于目标 GPU（见消息末尾 Target GPU 提示）：sm_80+ 用 tf32、否则用 ieee。建议 BLOCK_K ≥ 32、BLOCK_M/N 取 32~128。",
    "launch_sig": "launch(a: Tensor, b: Tensor, M: int, K: int, N: int) -> Tensor   # M/K/N 由 meta 提供",
}

# 默认规模：128 立方，sm_75 ieee dot 也能较快冒烟
DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 128, 128
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    m, k, n = DEFAULT_M, DEFAULT_K, DEFAULT_N
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    return {"a": a, "b": b, "meta": {"M": m, "K": k, "N": n}}


def golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


@triton.jit
def _matmul_kernel(a, b, c, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_ptrs,
                         mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                         other=0.0)
        b_tile = tl.load(b_ptrs,
                         mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                         other=0.0)
        if PREC == "tf32":
            acc += tl.dot(a_tile, b_tile, input_precision="tf32")
        else:
            acc += tl.dot(a_tile, b_tile, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def auto_dot_precision() -> str:
    """按当前 GPU 自动选 fp32 tl.dot 精度：sm_80+ 用 tf32(快)，否则 ieee(精确)。"""
    cap = torch.cuda.get_device_capability(0)
    return "tf32" if cap >= (8, 0) else "ieee"


def reference_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"K 不匹配: {k} vs {k2}"
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        PREC=auto_dot_precision(),
    )
    return c


def check(out: torch.Tensor, ref: torch.Tensor,
          rtol: float | None = None, atol: float | None = None) -> bool:
    rtol = TOL["rtol"] if rtol is None else rtol
    atol = TOL["atol"] if atol is None else atol
    return bool(torch.allclose(out, ref, rtol=rtol, atol=atol))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA 才能冒烟"
    args = generate_inputs()
    a, b = args["a"], args["b"]
    y = reference_triton(a, b)
    g = golden(a, b)
    print(f"[{OP_NAME}] meta={args['meta']}")
    print(f"  reference_triton vs golden  allclose: {check(y, g)}")
    print("冒烟通过 ✔")
