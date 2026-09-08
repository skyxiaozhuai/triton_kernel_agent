"""matmul_bias_relu —— 融合算子家族成员③（GEMM + epilogue 融合，最贴近工业级）。

程序级前向：c = relu(a @ b + bias)，a:[M,K], b:[K,N], bias:[N] -> c:[M,N]。
融合语义：GEMM 的 epilogue（加 bias + relu）在**同一 kernel、片上完成**——
中间结果 a@b 的 [M,N] 不写回全局内存。
分离版 = GEMM kernel 写 C + 独立 elementwise kernel 读 C 加 bias/relu 再写。
这正是 flash-attention / 工业 kernel 里 epilogue fusion 的简化演示。

冒烟: python -m benchmarks.ops.matmul_bias_relu
"""
import torch
import triton
import triton.language as tl

from .matmul import auto_dot_precision  # 与 matmul 同一精度自适应逻辑

OP_NAME = "matmul_bias_relu"

TOL32 = {"rtol": 1e-3, "atol": 1e-3}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}

OP_META = {
    "name": OP_NAME,
    "category": "matmul / gemm",        # 与 matmul 同类 → RAG 可互相参考
    "difficulty": "hard",
    "fused": True,                       # 融合算子家族成员③
    "dtype": "float32 / float16",
    "signature": "c = matmul_bias_relu(a, b, bias)   # c = relu(a@b + bias)",
    "description": (
        "FUSED GEMM with epilogue: c[m, n] = relu(sum_k a[m,k]*b[k,n] + bias[n]), "
        "a:[M,K], b:[K,N] row-major, bias:[N] broadcast along columns. "
        "Use tl.dot on BLOCK_M x BLOCK_K / BLOCK_K x BLOCK_N tiles with an "
        "outer loop over K, accumulate in fp32. "
        "REQUIREMENT (fusion): after the K loop, apply the epilogue (add bias "
        "then relu) IN THE SAME KERNEL in registers, then store once. "
        "The full a@b intermediate must NOT be written back to global memory "
        "(that would be unfused). Mask loads/stores since M/N/K may not "
        "divide evenly."
    ),
    "notes": "累加用 fp32 保精度；bias 沿 N 维广播。fp32 的 tl.dot 用哪种 input_precision 取决于目标 GPU（见消息末尾 Target GPU 提示）：sm_80+ 用 tf32、否则用 ieee。建议 BLOCK_K ≥ 32、BLOCK_M/N 32~128。",
    "launch_sig": "launch(a: Tensor, b: Tensor, bias: Tensor, M: int, K: int, N: int) -> Tensor   # M/K/N 由 meta 提供",
}

# 默认规模 128 立方（sm_75 ieee dot 冒烟）；服务器可大 shape
DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 128, 128
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32


def _make_case(m, k, n, device, dtype):
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    bias = torch.randn(n, device=device, dtype=dtype)
    return {"a": a, "b": b, "bias": bias, "meta": {"M": m, "K": k, "N": n}}


def generate_inputs(device: str = "cuda",
                    dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(DEFAULT_M, DEFAULT_K, DEFAULT_N, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非整除/极小）。"""
    specs = [(torch.float32, ((DEFAULT_M, DEFAULT_K, DEFAULT_N), (100, 130, 97), (16, 17, 19))),
             (torch.float16, ((64, 96, 80), (32, 33, 64)))]
    return [_make_case(m, k, n, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for m, k, n in shapes]


def golden(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(a @ b + bias)


@triton.jit
def _mmbr_kernel(a, b, bias, c, M, N, K,
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

    # —— epilogue 融合：bias + relu 在片上完成，中间 [M,N] 不落全局 ——
    bvec = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)      # [BLOCK_N]
    acc = acc + bvec[None, :]                                       # 广播加 bias
    acc = tl.maximum(acc, 0.0)                                      # relu

    c_ptrs = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def reference_triton(a: torch.Tensor, b: torch.Tensor,
                     bias: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2 and bias.numel() == n, "shape 不匹配"
    in_dtype = a.dtype
    if in_dtype == torch.float16:
        a, b, bias = a.float(), b.float(), bias.float()
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _mmbr_kernel[grid](
        a, b, bias, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        PREC=auto_dot_precision(),
    )
    return c.to(in_dtype)


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
    y = reference_triton(args["a"], args["b"], args["bias"])
    g = golden(args["a"], args["b"], args["bias"])
    print(f"[{OP_NAME}] meta={args['meta']}")
    print(f"  fused GEMM+epilogue vs golden  allclose: {check(y, g)}")
    print("冒烟通过 ✔")
