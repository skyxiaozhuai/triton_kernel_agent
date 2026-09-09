"""conv2d_pad —— 带 stride 与 padding 的 2D 卷积（conv2d 变体①）。

y[n,co,oh,ow] = bias[co] + Σ_{ci,kh,kw} x[n,ci,ih,iw]·w[co,ci,kh,kw]
  ih = oh*S - P + kh ; iw = ow*S - P + kw
  OH = floor((H + 2P - KH)/S) + 1 ; OW = 同
  x:[N,CI,H,W]  w:[CO,CI,KH,KW]  bias:[CO]  y:[N,CO,OH,OW]

与 conv2d(valid, S=1,P=0) 的差别 = 需要处理 pad/stride 越界：oh*S-P+kh 可能
<0 或 >=H，越界位置按 0 贡献（等价 padding-zeros）。每输出元素是标量归约，可对
越界做 clamp-load + 乘 valid 标志（避免 OOB 读）。
冒烟: python -m benchmarks.ops.conv2d_pad
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..shape_env import get_op_shape

OP_NAME = "conv2d_pad"
S, P = 2, 1   # 本 op 默认 stride=2 / padding=1

TOL32 = {"rtol": 1e-3, "atol": 1e-3}
TOL16 = {"rtol": 1e-2, "atol": 1e-2}

OP_META = {
    "name": OP_NAME,
    "category": "conv2d / strided-padded sliding-window",
    "difficulty": "hard",
    "dtype": "float32 / float16",
    "signature": "y = conv2d_pad(x, weight, bias)  # x:[N,CI,H,W], w:[CO,CI,KH,KW], "
                 "bias:[CO] -> y:[N,CO,OH,OW]",
    "description": (
        "2D convolution with STRIDE=S and PADDING=P (default S=2, P=1): "
        "y[n,co,oh,ow] = bias[co] + sum over (ci,kh,kw) of "
        "x[n, ci, oh*S-P+kh, ow*S-P+kw] * weight[co,ci,kh,kw]; "
        "OH = floor((H+2P-KH)/S)+1, OW likewise. All tensors row-major. "
        "RECOMMENDED (verifies easily): one program per output element "
        "(n,co,oh,ow) decoded from program_id; loop ci/kh/kw with SCALAR loads "
        "accumulating in fp32. Since oh*S-P+kh may be <0 or >=H (padding), you "
        "must handle out-of-range: clamp the row/col index to [0,H-1]/[0,W-1] "
        "before loading (so the address is always valid), compute a scalar "
        "`valid = (ih>=0)&(ih<H)&(iw>=0)&(iw<W)`, and multiply the loaded value "
        "by valid.to(fp32) so out-of-range contributes 0 (zero-padding). Then "
        "add bias[co] and store."
    ),
    "notes": "stride=2/padding=1；越界按 0 填充（clamp-load + valid 掩码乘）；fp32 累加。",
    "launch_sig": "launch(x, weight, bias, N, CO, CI, H, W, KH, KW, OH, OW, S, P) "
                  "-> Tensor  # 维度与 stride/pad 由 meta 提供",
}


def current_shape() -> tuple[int, int, int, int]:
    """主 case 输入 (N, CI, H, W)：默认 2×3×32×32。"""
    return get_op_shape(OP_NAME, (2, 3, 32, 32))


def _make_case(n, ci, h, w, co, kh, kw, device, dtype):
    x = torch.randn(n, ci, h, w, device=device, dtype=dtype) * 0.5
    weight = torch.randn(co, ci, kh, kw, device=device, dtype=dtype) * 0.2
    bias = torch.randn(co, device=device, dtype=dtype) * 0.1
    oh = (h + 2 * P - kh) // S + 1
    ow = (w + 2 * P - kw) // S + 1
    return {"x": x, "weight": weight, "bias": bias,
            "meta": {"N": n, "CO": co, "CI": ci, "H": h, "W": w,
                     "KH": kh, "KW": kw, "OH": oh, "OW": ow, "S": S, "P": P}}


def generate_inputs(device: str = "cuda", dtype: torch.dtype = torch.float32) -> dict:
    n, ci, h, w = current_shape()
    return _make_case(n, ci, h, w, 4, 3, 3, device, dtype)


def generate_cases(device: str = "cuda", dtype=None) -> list[dict]:
    """覆盖 fp32 + fp16 多组 shape（主/非整除/极小）。dtype=None 表示都测。"""
    n, ci, h, w = current_shape()
    specs = [(torch.float32, ((n, ci, h, w, 4, 3, 3),
                              (2, 2, 19, 17, 4, 3, 3),
                              (1, 1, 5, 5, 2, 3, 3))),
             (torch.float16, ((2, 3, 16, 16, 4, 3, 3),))]
    return [_make_case(nn, c, hh, ww, co, kh, kw, device, dt)
            for dt, shapes in specs if dtype is None or dt == dtype
            for (nn, c, hh, ww, co, kh, kw) in shapes]


def golden(x, weight, bias):
    return F.conv2d(x, weight, bias, stride=S, padding=P)


@triton.jit
def _conv2d_pad_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                       CO, CI, H, W, KH, KW, OH, OW, S, P,
                       sx_n, sx_c, sx_h, sx_w,
                       sw_co, sw_ci, sw_kh, sw_kw):
    pid = tl.program_id(0)
    ow = pid % OW
    t = pid // OW
    oh = t % OH
    t = t // OH
    co = t % CO
    n = t // CO

    acc = tl.zeros((), dtype=tl.float32)
    for ci in range(0, CI):
        x_c = x_ptr + n * sx_n + ci * sx_c
        w_c = w_ptr + co * sw_co + ci * sw_ci
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh * S - P + kh
                iw = ow * S - P + kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                ihs = tl.minimum(tl.maximum(ih, 0), H - 1)
                iws = tl.minimum(tl.maximum(iw, 0), W - 1)
                xv = tl.load(x_c + ihs * sx_h + iws * sx_w)
                wv = tl.load(w_c + kh * sw_kh + kw * sw_kw)
                acc += xv.to(tl.float32) * valid.to(tl.float32) * wv.to(tl.float32)
    bv = tl.load(b_ptr + co)
    tl.store(y_ptr + pid, (acc + bv).to(y_ptr.dtype.element_ty))


def reference_triton(x, weight, bias):
    n, ci, h, w = x.shape
    co, _ci, kh, kw = weight.shape
    oh, ow = (h + 2 * P - kh) // S + 1, (w + 2 * P - kw) // S + 1
    y = torch.empty((n, co, oh, ow), device=x.device, dtype=x.dtype)
    sx, sw = x.stride(), weight.stride()
    _conv2d_pad_kernel[(n * co * oh * ow,)](
        x, weight, bias, y, co, ci, h, w, kh, kw, oh, ow, S, P,
        sx[0], sx[1], sx[2], sx[3], sw[0], sw[1], sw[2], sw[3])
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
          f"S={S} P={P} y={tuple(y.shape)}")
    err = (y - g).abs().max().item()
    print(f"  reference vs golden  allclose: {check(y, g)}  max_abs_err={err:.3e}")
    print("冒烟通过 ✔")
