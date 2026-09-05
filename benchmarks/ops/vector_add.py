"""vector_add —— 第一个基准算子（模板范例）。

这个文件定义"一个 op 在 benchmark 集里长什么样"：
  - OP_META       : 给 LLM 看的语义/签名/约束
  - generate_inputs : 输入生成器
  - golden        : PyTorch eager 的期望结果 —— 机器可打分的 ground truth
  - reference_triton : 手写参考 kernel —— 仅供我们自检 runner，绝不喂给 agent
  - check         : 数值对齐断言

冒烟测试：python -m benchmarks.ops.vector_add
"""
import torch
import triton
import triton.language as tl

OP_NAME = "vector_add"

OP_META = {
    "name": OP_NAME,
    "category": "elementwise",
    "dtype": "float32",
    "signature": "y = vector_add(x1, x2)",
    "description": (
        "Element-wise addition of two 1-D tensors of the same length N: "
        "y[i] = x1[i] + x2[i] for i in [0, N). "
        "One program handles a contiguous BLOCK of elements; "
        "use an offset mask `offs < N` since N may not be divisible by BLOCK."
    ),
    "notes": "输出与输入同 shape、同 dtype。",
    "launch_sig": "launch(x1: Tensor, x2: Tensor, n: int) -> Tensor   # n 是元素总数，由 meta 提供",
}


def default_n() -> int:
    # 2^20 float32 = 4MB/张：1650 冒烟无压力；服务器可用更大 shape。
    return 1 << 20


def _make_case(n, device, dtype):
    x1 = torch.randn(n, device=device, dtype=dtype)
    x2 = torch.randn(n, device=device, dtype=dtype)
    return {"x1": x1, "x2": x2, "meta": {"n": n}}


def generate_inputs(n: int | None = None, device: str = "cuda",
                    dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(n or default_n(), device, dtype)


def generate_cases(device: str = "cuda",
                   dtype: torch.dtype = torch.float32) -> list[dict]:
    """多组 shape：主 / 非整除边界 / 小。全过才算正确。"""
    return [_make_case(n, device, dtype)
            for n in (default_n(), 1_000_003, 1025)]


def golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 + x2


@triton.jit
def _vector_add_kernel(x1, x2, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)
    tl.store(y + offs, v, mask=mask)


def reference_triton(x1: torch.Tensor, x2: torch.Tensor,
                     BLOCK: int = 1024) -> torch.Tensor:
    y = torch.empty_like(x1)
    n = x1.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _vector_add_kernel[grid](x1, x2, y, n, BLOCK=BLOCK)
    return y


def check(out: torch.Tensor, ref: torch.Tensor,
          rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    return bool(torch.allclose(out, ref, rtol=rtol, atol=atol))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA 才能冒烟"
    args = generate_inputs()
    x1, x2 = args["x1"], args["x2"]
    y_ref = reference_triton(x1, x2)
    y_gold = golden(x1, x2)
    print(f"[{OP_NAME}] n={args['meta']['n']}, shape={tuple(x1.shape)}")
    print(f"  reference_triton vs golden  allclose: {check(y_ref, y_gold)}")
    print("冒烟通过 ✔")
