#!/usr/bin/env python3
"""matmul 参数扫描：对经验库/文件的 matmul kernel 扫 launch 参数找更快配置。

典型用法（剖析显示 compute-bound → 扫 num_warps / num_stages）：
    MATMUL_SHAPE=4096,4096,4096 python scripts/sweep_matmul.py --op matmul \
        --warps 4,8 --stages 2,3,4
参数网格：num_warps × num_stages（tile 结构默认不动，也可用 --blocks 扩）。
本地 do_bench，无 LLM / 无 API。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import triton  # noqa: E402
from triton.testing import do_bench  # noqa: E402

from benchmarks.ops.matmul import current_shape  # noqa: E402


def _load_kernel_module(code: str, name: str = "_mm_sweep_src"):
    """把代码写成临时文件再 import（triton.jit 需要真实源码文件，不能用 exec）。"""
    import uuid
    path = os.path.join(ROOT, "scratch", f"{name}_{uuid.uuid4().hex[:6]}.py")
    with open(path, "w") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="matmul")
    ap.add_argument("--tiles",
                    default="64x64x32,128x128x64,128x128x32,128x64x64",
                    help="逗号分隔 tile: BMxBNxBK（默认 64 + 128 族）")
    ap.add_argument("--warps", default="4,8", help="逗号分隔 num_warps")
    ap.add_argument("--stages", default="2,3", help="逗号分隔 num_stages")
    ap.add_argument("--rep", type=int, default=30, help="do_bench rep")
    args = ap.parse_args()

    M, K, N = current_shape()   # 读 env MATMUL_SHAPE（默认 128³）
    print(f"shape M,K,N = {M},{K},{N}")

    mem_path = os.path.join(ROOT, "results", "memory", f"{args.op}.json")
    with open(mem_path, encoding="utf-8") as f:
        code = json.load(f)["code"]
    mod = _load_kernel_module(code)
    kernel = mod._matmul_kernel

    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda", dtype=a.dtype)

    warps = [int(x) for x in args.warps.split(",")]
    stages = [int(x) for x in args.stages.split(",")]

    def run(BM, BN, BK, nw, ns):
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        c.zero_()
        kernel[grid](a, b, c, M, K, N,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1),
                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                     num_warps=nw, num_stages=ns)
        return c

    # 基线（与 launch 相同的 64/64/32, nw4, ns2）先算参考输出用于一致性
    ref = run(64, 64, 32, 4, 2).clone()

    tiles = [tuple(int(x) for x in t.split("x")) for t in args.tiles.split(",")]
    if (64, 64, 32) not in tiles:
        tiles = [(64, 64, 32)] + tiles   # 保证基线在内便于对比

    print(f"{'BM,BN,BK':>12}{'warps':>6}{'stages':>7}{'ms':>10}{'vs基线':>9}  ok")
    rows = []
    base_ms = None
    t0 = time.time()
    for (BM, BN, BK) in tiles:
        for nw in warps:
            for ns in stages:
                out = run(BM, BN, BK, nw, ns)
                ok = bool(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))
                ms = float(do_bench(lambda: run(BM, BN, BK, nw, ns),
                                    warmup=5, rep=args.rep))
                rows.append((BM, BN, BK, nw, ns, ms, ok))
                if base_ms is None:
                    base_ms = ms
                ratio = ms / base_ms
                print(f"{f'{BM}x{BN}x{BK}':>12}{nw:>6}{ns:>7}{ms:>10.3f}"
                      f"{ratio:>8.2f}x  {'✔' if ok else '✗'}")

    best = min(rows, key=lambda r: r[5])
    base = rows[0]
    print("-" * 56)
    print(f"基线: {base[0]}x{base[1]}x{base[2]} nw{base[3]} ns{base[4]} = {base[5]:.3f}ms")
    if best[5] < base[5] * (1 - 0.005):
        print(f"最佳: {best[0]}x{best[1]}x{best[2]} nw{best[3]} ns{best[4]}"
              f" = {best[5]:.3f}ms  ({base[5] / best[5]:.2f}x 提速) {'✔' if best[6] else '✗'}")
    else:
        print(f"扫描未找到 >0.5% 的更快配置 —— 当前 {base[5]:.3f}ms 接近该 tile/卡极限")
    print(f"耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
