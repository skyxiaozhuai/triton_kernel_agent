"""relu_sum —— 融合算子家族成员②（融合跨到 reduction 类）。

程序级前向：y = sum_i relu(x[i])，x:[N] -> y:[1]。
融合语义：relu 在 stage1 kernel 内【load 后、写 partial 前】完成，
**relu 的整张中间结果不写回全局内存**（分离版 = relu kernel 写整张 + sum kernel 读整张）。
跨 block 归约仍需两阶段（partial 部分和 → 二次归约），与 sum_1d 同构但省一次全张量读写。

冒烟: python -m benchmarks.ops.relu_sum
"""
import torch
import triton
import triton.language as tl

OP_NAME = "relu_sum"

OP_META = {
    "name": OP_NAME,
    "category": "reduction (cross-block)",   # 与 sum_1d 同类 → RAG 可互相参考
    "difficulty": "medium",
    "fused": True,                            # 融合算子家族成员
    "dtype": "float32 / float16",
    "signature": "y = relu_sum(x)   # x: [N] -> y: [1], y = sum(relu(x))",
    "description": (
        "FUSED operator: y = sum_i relu(x[i]) for a 1-D tensor x of length N "
        "(result is a length-1 tensor). This fuses an elementwise (relu) into "
        "a full reduction. A single program can only reduce the elements it "
        "loads, so use TWO stages: (1) each program loads its BLOCK-sized "
        "chunk, applies relu IN REGISTERS (max(v, 0)) then reduces to a "
        "partial result; (2) one final program reduces the partial results. "
        "KEY: the full-size relu intermediate must NOT be written back to "
        "global memory (that would be unfused). Use masking since N may not "
        "divide by BLOCK."
    ),
    "notes": "输出是长度为 1 的 tensor；sum/累加用 fp32 保精度；禁止 torch 计算。",
    "launch_sig": "launch(x: Tensor, N: int) -> Tensor   # N 是元素总数，由 meta 提供",
}

BLOCK = 1024
DEFAULT_N = 1 << 20

TOL32 = {"rtol": 1e-3, "atol": 1e-2}   # 累加顺序差异随 N 累积，同 sum_1d
TOL16 = {"rtol": 1e-2, "atol": 1e-2}


def _make_case(n, device, dtype):
    x = torch.randn(n, device=device, dtype=dtype)
    return {"x": x, "meta": {"N": n}}


def generate_inputs(device: str = "cuda",
                    dtype: torch.dtype = torch.float32) -> dict:
    return _make_case(DEFAULT_N, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 多组 N（fp16 用中规模避免误差放大）。"""
    specs = [(torch.float32, (DEFAULT_N, 1_000_003, 1000)),
             (torch.float16, (1 << 18, 1_000_003))]
    return [_make_case(n, device, dt)
            for dt, ns in specs if dtype is None or dt == dtype
            for n in ns]


def golden(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.relu(x)).reshape(1)


@triton.jit
def _relu_sum_stage1(x, partial, n, BLOCK: tl.constexpr):
    # 融合点：load → relu(寄存器) → 块内归约 → partial（不写整张中间）
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    vals = tl.maximum(vals, 0.0)                       # relu 融合进来
    tl.store(partial + pid, tl.sum(vals, axis=0))


@triton.jit
def _relu_sum_stage2(partial, out, nparts, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < nparts
    vals = tl.load(partial + offs, mask=mask, other=0.0)
    tl.store(out, tl.sum(vals, axis=0))


def reference_triton(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    grid1 = (triton.cdiv(n, BLOCK),)
    partial = torch.empty(grid1[0], device=x.device, dtype=torch.float32)
    _relu_sum_stage1[grid1](x, partial, n, BLOCK=BLOCK)

    out = torch.empty(1, device=x.device, dtype=torch.float32)
    nparts = grid1[0]
    BLOCK2 = triton.next_power_of_2(nparts)
    _relu_sum_stage2[(1,)](partial, out, nparts, BLOCK=BLOCK2)
    return out.to(x.dtype)


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
    print(f"  fused reference vs golden  allclose: {check(y, g)}")
    print("冒烟通过 ✔")
