#!/usr/bin/env python3
"""沙箱 executor 自测：验证 harness 判定链路的正确性。

用三段"伪生成代码"喂给 executor：
  1) 正确代码          -> 期望 status=pass
  2) 数值错误代码       -> 期望 status=correctness
  3) 语法错误代码       -> 期望 status=syntax
  4) 运行时异常代码     -> 期望 status=error

用法: python scripts/test_executor.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tools.executor import run  # noqa: E402

GOOD = '''
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

WRONG_VALUE = '''
def launch(x1, x2, n):
    return x1 - x2
'''

SYNTAX_ERR = '''
def launch(x1, x2, n):
    return x1 +   # 语法错误
'''

RUNTIME_ERR = '''
def launch(x1, x2, n):
    raise ValueError("故意的运行时错误")
'''


def main() -> int:
    if not __import__("torch").cuda.is_available():
        print("[WARN] 需要 CUDA")
        return 2
    cases = [
        ("good -> pass", GOOD, "pass"),
        ("wrong value -> correctness", WRONG_VALUE, "correctness"),
        ("syntax error -> syntax", SYNTAX_ERR, "syntax"),
        ("runtime error -> error", RUNTIME_ERR, "error"),
    ]
    failed = 0
    for label, code, expect in cases:
        rep = run("vector_add", code)
        tag = "✔" if rep.status == expect else "✗"
        print(f"{tag} [{label}] 实际={rep.status} (期望 {expect})  msg={rep.message[:60]!r}")
        if rep.status != expect:
            failed += 1
    print("-" * 60)
    print("executor 自测通过 ✔" if failed == 0 else f"{failed} 项不符合预期")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
