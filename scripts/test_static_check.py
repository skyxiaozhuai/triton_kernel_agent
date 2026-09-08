"""static_check 静态闸门自测：合法代码放行 + 各类作弊/结构违规被拦。

- 回归：results/memory/*.json 里真实通过 GPU 判卷的成功 kernel 必须全部放行（不误伤）。
- 反作弊：torch 外包计算 / @ 矩阵乘 / kernel 内 torch / 反射 / 危险 import 必须被拦。
- 结构：缺 launch / 缺 kernel / launch 不调 kernel / launch 直接 return 输入运算 必须被拦。

运行（无需 GPU，纯 AST）：python scripts/test_static_check.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.tools import static_check  # noqa: E402

MEMORY_DIR = os.path.join(ROOT, "results", "memory")

# —— 内嵌合法样本（与 executor.__main__ 的自测样本同构） ——
GOOD_VECTOR_ADD = '''
import triton
import triton.language as tl

@triton.jit
def kernel(x1, x2, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)
    tl.store(y + offs, v, mask=mask)

def launch(x1, x2, n):
    y = torch.empty_like(x1)
    grid = (triton.cdiv(n, 1024),)
    kernel[grid](x1, x2, y, n, BLOCK=1024)
    return y
'''

GOOD_SOFTMAX = '''
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + offs)
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    tl.store(y + offs, e / s)

def launch(x, n):
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    softmax_kernel[grid](x, y, n, BLOCK=1024)
    return y
'''

# —— 作弊 / 违规样本 ——
CHEAT_TORCH_MATMUL = '''
import triton
import triton.language as tl
@triton.jit
def kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b + offs_k[:, None] + offs_n[None, :] * N
    av = tl.load(a_ptrs)
    bv = tl.load(b_ptrs)
    acc = tl.dot(av, bv)
    c_ptrs = c + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)
def launch(a, b, M, N, K):
    c = torch.matmul(a, b)   # ← 作弊：把矩阵乘交给 torch
    return c
'''

CHEAT_KERNEL_INNER_TORCH = GOOD_VECTOR_ADD.replace(
    "v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)",
    "v = torch.add(tl.load(x1 + offs, mask=mask), tl.load(x2 + offs, mask=mask))")

CHEAT_RETURN_BINOP = '''
import triton
import triton.language as tl
@triton.jit
def kernel(x1, x2, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)
    tl.store(y + offs, v, mask=mask)
def launch(x1, x2, n):
    y = torch.empty_like(x1)
    grid = (triton.cdiv(n, 1024),)
    kernel[grid](x1, x2, y, n, BLOCK=1024)
    return x1 + x2   # ← 作弊：绕过 kernel 直接用 torch 算答案返回
'''

CHEAT_MATMUL_AT = '''
import triton
import triton.language as tl
@triton.jit
def kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass
def launch(a, b, M, N, K):
    return a @ b   # ← 作弊：@ 交给 torch
'''

NO_LAUNCH = GOOD_VECTOR_ADD.split("def launch")[0]          # 去掉 launch
NO_KERNEL = '''
def launch(x1, x2, n):
    return x1 + x2   # 无 @triton.jit kernel，纯 torch
'''
LAUNCH_NO_KERNEL_CALL = '''
import triton
import triton.language as tl
@triton.jit
def unused(x, y, n, BLOCK: tl.constexpr):
    pass
def launch(x1, x2, n):
    return x1 + x2   # 定义了 kernel 但从不调用，直接 torch 算
'''
REFLECT_GLOBALS = '''
import triton
import triton.language as tl
@triton.jit
def kernel(x1, x2, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x1 + offs, mask=mask) + tl.load(x2 + offs, mask=mask)
    tl.store(y + offs, v, mask=mask)
def launch(x1, x2, n):
    g = globals()        # ← 反射
    y = torch.empty_like(x1)
    grid = (triton.cdiv(n, 1024),)
    kernel[grid](x1, x2, y, n, BLOCK=1024)
    return y
'''
IMPORT_OS = GOOD_VECTOR_ADD.replace("import triton", "import os\nimport triton", 1)
SYNTAX_BAD = '''
import triton
import triton.language as tl
@triton.jit
def kernel(   # ← 语法错误（缺冒号
'''


def _expect(cases: list[tuple[str, str, str]]) -> None:
    """cases: [(代码, 期望 ok, 期望 category)]；ok 用 'pass'/'block' 表示。"""
    for i, (code, want_ok, want_cat) in enumerate(cases, 1):
        res = static_check.check_generated_code(code, op_name="vector_add")
        got_ok, got_cat = res["ok"], res["category"]
        ok_flag = (got_ok is True and want_ok == "pass") or (got_ok is False and want_ok == "block")
        status = "ok " if ok_flag else "FAIL"
        print(f"[{status}] #{i:02d} want={want_ok}/{want_cat:<10} got={got_ok}/{got_cat:<10}  {res['reason'][:80]}")
        assert ok_flag, f"case #{i}: want {want_ok}/{want_cat}, got {got_ok}/{got_cat}: {res['reason']}"
        if want_ok == "block":
            assert got_cat == want_cat, f"case #{i}: category want {want_cat}, got {got_cat}"


def main() -> int:
    print("== 1) 合法代码必须放行 ==")
    cases_pass = [
        (GOOD_VECTOR_ADD, "pass", "pass"),
        (GOOD_SOFTMAX, "pass", "pass"),
    ]
    _expect(cases_pass)

    print("\n== 2) 作弊样本必须被拦 ==")
    cases_block = [
        (CHEAT_TORCH_MATMUL, "block", "cheat"),
        (CHEAT_KERNEL_INNER_TORCH, "block", "cheat"),
        (CHEAT_RETURN_BINOP, "block", "structure"),
        (CHEAT_MATMUL_AT, "block", "cheat"),
        (REFLECT_GLOBALS, "block", "cheat"),
        (IMPORT_OS, "block", "cheat"),
    ]
    _expect(cases_block)

    print("\n== 3) 结构违规必须被拦 ==")
    cases_struct = [
        (NO_LAUNCH, "block", "structure"),
        (NO_KERNEL, "block", "structure"),
        (LAUNCH_NO_KERNEL_CALL, "block", "structure"),
        (SYNTAX_BAD, "block", "syntax"),
        ("", "block", "structure"),
    ]
    _expect(cases_struct)

    print("\n== 4) 回归：memory 里真实通过判卷的成功 kernel 不得被误伤 ==")
    mem_files = sorted(f for f in os.listdir(MEMORY_DIR) if f.endswith(".json"))
    assert mem_files, "results/memory 为空？先 seed_memory"
    n_ok = 0
    for fn in mem_files:
        op = fn[:-5]
        with open(os.path.join(MEMORY_DIR, fn), encoding="utf-8") as f:
            rec = json.load(f)
        res = static_check.check_generated_code(rec["code"], op_name=op)
        ok = res["ok"] is True
        n_ok += ok
        print(f"[{'ok ' if ok else 'FAIL'}] {op}: {res['reason'][:100]}")
        assert ok, f"memory 成功样本 {op} 被静态闸门误伤: {res['reason']}"
    print(f"\n全部通过: {n_ok}/{len(mem_files)} 个 memory 成功样本未误伤")
    return 0


if __name__ == "__main__":
    sys.exit(main())
