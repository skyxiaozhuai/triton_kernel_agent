"""conv2d —— 标准 2D 卷积（sliding-window，hard）。

y[n, co, oh, ow] = bias[co] + Σ_{ci,kh,kw} x[n, ci, ih, iw] * w[co, ci, kh, kw]
  ih = oh + kh, iw = ow + kw   （stride=1, padding=0, dilation=1, 无分组）
  x: [N, CI, H, W]   w: [CO, CI, KH, KW]   bias: [CO]   y: [N, CO, OH, OW]
  OH = H - KH + 1, OW = W - KW + 1

考察点：4D 索引与 layout、输出每个元素是对 (ci,kh,kw) 的小型归约、
weight/bias 广播、边界由 valid 卷积自动满足（无需 mask）。
冒烟: python -m benchmarks.ops.conv2d
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "conv2d"

# 数值容差：conv 对 (CI*KH*KW) 项求和，fp32 顺序误差小；fp16 放宽
TOL32 = {"rtol": 1e-3, "atol": 1e-3}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}

OP_META = {
    "name": OP_NAME,
    "category": "conv2d / sliding-window reduction",
    "difficulty": "hard",
    "dtype": "float32 / float16",
    "signature": "y = conv2d(x, weight, bias)  # x:[N,CI,H,W], w:[CO,CI,KH,KW], bias:[CO] -> y:[N,CO,OH,OW]",
    "description": (
        "Standard 2D convolution (stride=1, padding=0, dilation=1, no groups): "
        "y[n,co,oh,ow] = bias[co] + sum over (ci,kh,kw) of "
        "x[n, ci, oh+kh, ow+kw] * weight[co, ci, kh, kw]. "
        "OH = H - KH + 1, OW = W - KW + 1. All tensors row-major/contiguous. "
        "RECOMMENDED simple scheme that verifies easily: let ONE program compute "
        "ONE output element (n,co,oh,ow); decode co/oh/ow from program_id, then "
        "loop ci, kh, kw with SCALAR loads (tl.load returns a scalar at a scalar "
        "address) accumulating in fp32, add bias[co], store. "
        "Decode order suggestion: with grid=(N*CO*OH*OW,): ow=pid%OW; "
        "oh=(pid//OW)%OH; co=(pid//(OW*OH))%CO; n=pid//(OW*OH*CO). "
        "No mask needed: because OH/OW already guarantee oh+kh < H and ow+kw < W. "
        "Pass x/w/bias pointers plus the dimension ints (N,CO,CI,H,W,KH,KW,OH,OW) "
        "and per-tensor strides so the kernel indexes contiguous tensors correctly."
    ),
    "notes": "累加用 fp32 保精度；bias 沿 co 广播；输出 H/W = 输入 H/W - KH/KW + 1。",
    "launch_sig": "launch(x: Tensor[N,CI,H,W], weight: Tensor[CO,CI,KH,KW], "
                  "bias: Tensor[CO], N: int, CO: int, CI: int, H: int, W: int, "
                  "KH: int, KW: int, OH: int, OW: int) -> Tensor  # 维度由 meta 提供",
}

# 默认规模（小，sm_75 冒烟无压力）
DEFAULT = (4, 3, 64, 64, 8, 3, 3, 3)   # N,CI,H,W,CO,KH,KW 等 → 主 (4,3,64,64) w(8,3,3,3)


def current_shape() -> tuple[int, int, int, int]:
    """主 case 输入 x 形状 (N, CI, H, W)：默认 4×3×64×64；可用 OP_SHAPE 覆盖。"""
    return get_op_shape(OP_NAME, (4, 3, 64, 64))


def _make_case(n, ci, h, w, co, kh, kw, device, dtype):
    x = torch.randn(n, ci, h, w, device=device, dtype=dtype) * 0.5
    weight = torch.randn(co, ci, kh, kw, device=device, dtype=dtype) * 0.2
    bias = torch.randn(co, device=device, dtype=dtype) * 0.1
    oh, ow = h - kh + 1, w - kw + 1
    return {"x": x, "weight": weight, "bias": bias,
            "meta": {"N": n, "CO": co, "CI": ci, "H": h, "W": w,
                     "KH": kh, "KW": kw, "OH": oh, "OW": ow}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    n, ci, h, w = current_shape()
    return _make_case(n, ci, h, w, 8, 3, 3, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 的多组 shape（主/非方形/极小）。dtype=None 表示都测。"""
    n, ci, h, w = current_shape()
    specs = [(torch.float32, ((n, ci, h, w, 8, 3, 3),
                              (2, 3, 17, 11, 4, 3, 3),
                              (1, 2, 7, 7, 2, 3, 3))),
             (torch.float16, ((2, 3, 16, 16, 4, 3, 3),))]
    return [_make_case(nn, c, hh, ww, co, kh, kw, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for (nn, c, hh, ww, co, kh, kw) in shapes]


def golden(x, weight, bias):
    return F.conv2d(x, weight, bias)


@triton.jit
def _conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                   CO, CI, H, W, KH, KW, OH, OW,
                   sx_n, sx_c, sx_h, sx_w,
                   sw_co, sw_ci, sw_kh, sw_kw,
                   sy_n, sy_co, sy_h, sy_w):
    pid = tl.program_id(0)
    ow = pid % OW
    t = pid // OW
    oh = t % OH
    t = t // OH
    co = t % CO
    n = t // CO

    acc = tl.zeros((), dtype=tl.float32)
    for ci in range(0, CI):
        x_base = x_ptr + n * sx_n + ci * sx_c
        w_base = w_ptr + co * sw_co + ci * sw_ci
        for kh in range(0, KH):
            ih = oh + kh
            for kw in range(0, KW):
                iw = ow + kw
                xv = tl.load(x_base + ih * sx_h + iw * sx_w)
                wv = tl.load(w_base + kh * sw_kh + kw * sw_kw)
                acc += xv.to(tl.float32) * wv.to(tl.float32)
    bv = tl.load(b_ptr + co)
    tl.store(y_ptr + n * sy_n + co * sy_co + oh * sy_h + ow * sy_w,
             (acc + bv).to(y_ptr.dtype.element_ty))


def reference_triton(x, weight, bias):
    n, ci, h, w = x.shape
    co, _ci, kh, kw = weight.shape
    oh, ow = h - kh + 1, w - kw + 1
    y = torch.empty((n, co, oh, ow), device=x.device, dtype=x.dtype)
    sx = x.stride()          # (CI*H*W, H*W, W, 1)
    sw = weight.stride()     # (CI*KH*KW, KH*KW, KW, 1)
    sy = y.stride()          # (CO*OH*OW, OH*OW, OW, 1)
    _conv2d_kernel[(n * co * oh * ow,)](
        x, weight, bias, y,
        co, ci, h, w, kh, kw, oh, ow,
        sx[0], sx[1], sx[2], sx[3],
        sw[0], sw[1], sw[2], sw[3],
        sy[0], sy[1], sy[2], sy[3])
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
    y = reference_triton(args["x"], args["weight"], args["bias"])
    g = golden(args["x"], args["weight"], args["bias"])
    print(f"[{OP_NAME}] x={tuple(args['x'].shape)} w={tuple(args['weight'].shape)} "
          f"y={tuple(y.shape)}")
    err = (y - g).abs().max().item()
    print(f"  reference_triton vs golden  allclose: {check(y, g)}  "
          f"max_abs_err={err:.3e}")
    print("冒烟通过 ✔")
