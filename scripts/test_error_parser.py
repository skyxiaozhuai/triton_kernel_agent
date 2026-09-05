#!/usr/bin/env python3
"""error_parser 自测：用伪造的 ExecReport 验证分类是否正确。

用法: python scripts/test_error_parser.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.tools.error_parser import classify, to_feedback_text  # noqa: E402
from agent.tools.executor import ExecReport  # noqa: E402


def mk(status, ok, message, stderr="", max_err=None):
    return ExecReport(status=status, ok=ok, message=message,
                      max_abs_err=max_err, stdout="", stderr=stderr, wall_s=0.0)


def main() -> int:
    cases = [
        ("correctness", mk("correctness", False, "数值不对齐: max_abs_err=1.2e-03"),
         "correctness"),
        ("compile(tl undefined)",
         mk("error", False, "异常...", "triton.compiler.errors.CompilationError: NameError('tl is not defined')"),
         "compile"),
        ("cuda illegal memory",
         mk("error", False, "异常...", "RuntimeError: CUDA error: an illegal memory access was encountered"),
         "cuda"),
        ("syntax", mk("syntax", False, "生成代码存在语法错误: SyntaxError: invalid syntax",
                      "  File \"x.py\", line 3\n    return x1 +\nSyntaxError: invalid syntax"),
         "syntax"),
        ("timeout", mk("timeout", False, "执行超时，已终止"), "timeout"),
        ("runtime ValueError",
         mk("error", False, "异常: ValueError: shape mismatch", "ValueError: shape mismatch"),
         "runtime"),
    ]
    failed = 0
    for label, rep, expect in cases:
        got = classify(rep)["category"]
        tag = "✔" if got == expect else "✗"
        print(f"{tag} [{label}] -> {got} (期望 {expect})")
        if got != expect:
            failed += 1

    print("-" * 60)
    print("反馈示例（会进 LLM 上下文的样子）：")
    print(to_feedback_text(cases[1][1]))
    print("-" * 60)
    print("error_parser 自测通过 ✔" if failed == 0 else f"{failed} 项不符合预期")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
