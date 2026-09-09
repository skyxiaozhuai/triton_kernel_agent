"""layer_norm —— 带仿射的逐行 LayerNorm（hard：两次行归约 + 归一）。

y[m, n] = (x[m, n] - mean_m) / sqrt(var_m + eps) * weight[n] + bias[n]
其中 mean_m = mean_n(x[m, :])，var_m = mean_n((x[m, :] - mean_m)^2)（沿最后一维 N）。

考察点：per-row reduce（axis=1）、两遍归约求 mean/var（避免 E[x²]-E[x]² 的
大 N 精度损失）、mask 处理、逐元素仿射(weight/bias)融合。
冒烟: python -m benchmarks.ops.layer_norm
"""
import torch
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "layer_norm"
EPS = 1e-5

# 数值对齐容差：LN 方差由归约顺序/算法差异引入微小误差；fp16 放宽
TOL32 = {"rtol": 1e-3, "atol": 1e-4}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}

OP_META = {
    "name": OP_NAME,
    "category": "layer-norm / two-pass row-reduce + affine",
    "difficulty": "hard",
    "dtype": "float32 / float16",
    "signature": "y = layer_norm(x, weight, bias, eps)  "
                 "# x, y: [M, N]; weight, bias: [N]",
    "description": (
        "Per-row (over the last dim N) LayerNorm with learnable affine: "
        "mean_m = mean over n of x[m, :]; "
        "var_m  = mean over n of (x[m, :] - mean_m)^2; "
        "y[m, n] = (x[m, n] - mean_m) / sqrt(var_m + eps) * weight[n] + bias[n]. "
        "eps = 1e-5. N is the contiguous (last) dimension; x, y are [M, N]; "
        "weight and bias are 1-D [N]. "
        "IMPORTANT two-pass guidance: compute the row mean first, then in a "
        "second pass accumulate (x - mean)^2 for the variance. Do NOT use "
        "E[x^2] - E[x]^2 (catastrophic cancellation on larger N). "
        "One program per row is fine; if N is large, loop over N in blocks "
        "inside the program and accumulate in fp32. Keep the variance in fp32. "
        "Use an offset mask `offs < N` since N may not divide evenly."
    ),
    "notes": "带仿射(weight/bias)；两遍归约求 mean/var 更稳；fp32 累加；禁止 "
             "torch.nn.functional.layer_norm 外包。",
    "launch_sig": "launch(x: Tensor[M,N], weight: Tensor[N], bias: Tensor[N], "
                  "eps: float, M: int, N: int) -> Tensor[M,N]  # M/N 由 meta 提供",
}

# 默认规模：行数多、特征 N 适中 —— GTX1650 冒烟无压力
DEFAULT_M, DEFAULT_N = 1024, 512


def current_shape() -> tuple[int, int]:
    """主 case shape (M, N)：默认 1024×512；可用 env OP_SHAPE=M,N 调大。"""
    return get_op_shape(OP_NAME, (DEFAULT_M, DEFAULT_N))


def _make_case(m, n, device, dtype):
    x = torch.randn(m, n, device=device, dtype=dtype)
    # 仿射参数与输入同 dtype；小范围初值让归一化输出可预期
    weight = torch.randn(n, device=device, dtype=dtype) * 0.1 + 1.0
    bias = torch.randn(n, device=device, dtype=dtype) * 0.1
    return {"x": x, "weight": weight, "bias": bias, "eps": EPS,
            "meta": {"M": m, "N": n}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    m, n = current_shape()
    return _make_case(m, n, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非整除/极小）。dtype=None 表示都测。"""
    m, n = current_shape()
    specs = [(torch.float32, ((m, n), (257, 130), (31, 17))),
             (torch.float16, ((512, 512), (17, 128)))]
    return [_make_case(m, n, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for m, n in shapes]


def golden(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                      M, N, eps, stride_m,
                      BLOCK_N: tl.constexpr):
    row = tl.program_id(0)            # 每 program 处理一行
    base = x_ptr + row * stride_m

    # pass 1: mean = sum(x) / N
    s = tl.zeros((), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        xb = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(xb, axis=0)
    mean = s / N

    # pass 2: var = mean((x - mean)^2)
    v = tl.zeros((), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        xb = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        d = xb - mean
        d = tl.where(mask, d, 0.0)      # 越界元素清零，避免 (0-mean)^2 污染方差
        v += tl.sum(d * d, axis=0)
    var = v / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # pass 3: normalize + affine(weight/bias) + store
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        xb = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        yn = (xb - mean) * rstd
        # affine: y = (x-mean)*rstd * weight + bias（weight/bias 与 N 对齐）
        wb = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        yn = yn * wb + bb
        tl.store(y_ptr + row * stride_m + offs,
                 yn.to(y_ptr.dtype.element_ty), mask=mask)


def reference_triton(x, weight, bias, eps):
    m, n = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(max(1, n))
    _layernorm_kernel[(m,)](x, weight, bias, y, m, n, eps, x.stride(0),
                            BLOCK_N=BLOCK_N)
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
    y = reference_triton(args["x"], args["weight"], args["bias"], args["eps"])
    g = golden(args["x"], args["weight"], args["bias"], args["eps"])
    print(f"[{OP_NAME}] shape={tuple(args['x'].shape)}  meta={args['meta']}")
    err = (y - g).abs().max().item()
    print(f"  reference_triton vs golden  allclose: {check(y, g)}  "
          f"max_abs_err={err:.3e}")
    print("冒烟通过 ✔")
