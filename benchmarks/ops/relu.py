"""relu —— elementwise 同族算子（category 与 vector_add 相同，用于 RAG 跨算子互参考）。

y[i] = max(x[i], 0)
冒烟: python -m benchmarks.ops.relu
"""
import torch
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "relu"

OP_META = {
    "name": OP_NAME,
    "category": "elementwise",          # 与 vector_add 同类 → RAG 可互相参考
    "difficulty": "easy",
    "dtype": "float32 / float16",
    "signature": "y = relu(x)   # x: [N] -> y: [N]，同 dtype",
    "description": (
        "Element-wise ReLU of a 1-D tensor x of length N: y[i] = max(x[i], 0). "
        "One program handles a contiguous BLOCK of elements; "
        "use an offset mask `offs < N` since N may not divide by BLOCK."
    ),
    "notes": "输出与输入同 shape、同 dtype。",
    "launch_sig": "launch(x: Tensor, n: int) -> Tensor   # n 是元素总数，由 meta 提供",
}


def default_n() -> int:
    return 1 << 20


def current_shape() -> int:
    """主 case N：默认 2^20；可用 env OP_SHAPE=N 调大。"""
    return get_op_shape(OP_NAME, (default_n(),))[0]


TOL32 = {"rtol": 1e-4, "atol": 1e-5}   # fp32
TOL16 = {"rtol": 1e-2, "atol": 1e-2}   # fp16


def _make_case(n, device, dtype):
    x = torch.randn(n, device=device, dtype=dtype)
    return {"x": x, "meta": {"n": n}}


def generate_inputs(device: str = "cuda",
                    dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(current_shape(), device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape。dtype=None 表示都测。"""
    specs = [(torch.float32, (current_shape(), 1_000_003, 1025)),
             (torch.float16, (1 << 20, 100_003))]
    return [_make_case(n, device, dt)
            for dt, ns in specs if dtype is None or dt == dtype
            for n in ns]


def golden(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@triton.jit
def _relu_kernel(x, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(y + offs, tl.maximum(v, 0), mask=mask)


def reference_triton(x: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _relu_kernel[grid](x, y, n, BLOCK=BLOCK)
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
    y_ref = reference_triton(x)
    y_gold = golden(x)
    print(f"[{OP_NAME}] n={args['meta']['n']}, shape={tuple(x.shape)}")
    print(f"  reference_triton vs golden  allclose: {check(y_ref, y_gold)}")
    print("冒烟通过 ✔")
