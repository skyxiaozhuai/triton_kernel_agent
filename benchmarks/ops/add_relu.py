"""add_relu —— 第一个"融合(fused)复合算子"（对齐 KernelAgent Fuser 理念的最简版）。

把两个 elementwise 算子串成一个小程序级前向：y = relu(x1 + x2)。
关键约束：**必须在一个 @triton.jit kernel 内完成**（load → 相加 → relu → store），
中间结果 x1+x2 不得写回全局内存 —— 这就是"融合"：省一次全局读写。

golden 仍是 PyTorch eager：torch.relu(x1+x2)（机器可打分，融合与否只影响实现方式）。

冒烟: python -m benchmarks.ops.add_relu
"""
import torch
import triton
import triton.language as tl

OP_NAME = "add_relu"

OP_META = {
    "name": OP_NAME,
    "category": "elementwise",          # 与 vector_add/relu 同族 → RAG 可互相参考
    "difficulty": "easy",
    "dtype": "float32 / float16",
    "signature": "y = add_relu(x1, x2)   # y[i] = relu(x1[i] + x2[i])",
    "description": (
        "FUSED operator: given two 1-D tensors x1, x2 of length N, compute "
        "y[i] = relu(x1[i] + x2[i]). "
        "This is the fusion of two elementwise ops (add -> relu). "
        "REQUIREMENT: do it in ONE @triton.jit kernel: load x1 & x2, add in "
        "registers/on-chip, apply max(..., 0), store to y. "
        "The intermediate (x1+x2) must NOT be written back to global memory "
        "(that would be unfused and defeats the purpose). "
        "Use an offset mask `offs < N` since N may not divide by BLOCK."
    ),
    "notes": "输出与输入同 shape、同 dtype；禁止调用 torch 计算（静态闸门会拦）。",
    "launch_sig": "launch(x1: Tensor, x2: Tensor, n: int) -> Tensor   # n 是元素总数",
}


def default_n() -> int:
    return 1 << 20


TOL32 = {"rtol": 1e-4, "atol": 1e-5}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}   # fp16 逐元素两步，稍放宽


def _make_case(n, device, dtype):
    x1 = torch.randn(n, device=device, dtype=dtype)
    x2 = torch.randn(n, device=device, dtype=dtype)
    return {"x1": x1, "x2": x2, "meta": {"n": n}}


def generate_inputs(device: str = "cuda",
                    dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(default_n(), device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非整除/小）。"""
    specs = [(torch.float32, (default_n(), 1_000_003, 1025)),
             (torch.float16, (1 << 20, 100_003))]
    return [_make_case(n, device, dt)
            for dt, ns in specs if dtype is None or dt == dtype
            for n in ns]


def golden(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.relu(x1 + x2)


@triton.jit
def _add_relu_kernel(x1, x2, y, n, BLOCK: tl.constexpr):
    # 融合：一次 load 两输入 → 加 → relu → store，中间结果不落全局内存
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x1 + offs, mask=mask, other=0.0)
    b = tl.load(x2 + offs, mask=mask, other=0.0)
    tl.store(y + offs, tl.maximum(a + b, 0), mask=mask)


def reference_triton(x1: torch.Tensor, x2: torch.Tensor,
                     BLOCK: int = 1024) -> torch.Tensor:
    y = torch.empty_like(x1)
    n = x1.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _add_relu_kernel[grid](x1, x2, y, n, BLOCK=BLOCK)
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
    x1, x2 = args["x1"], args["x2"]
    y_ref = reference_triton(x1, x2)
    y_gold = golden(x1, x2)
    print(f"[{OP_NAME}] n={args['meta']['n']}, shape={tuple(x1.shape)}")
    print(f"  fused kernel vs golden  allclose: {check(y_ref, y_gold)}")
    print("冒烟通过 ✔（融合 kernel 与 eager 参考对齐）")
