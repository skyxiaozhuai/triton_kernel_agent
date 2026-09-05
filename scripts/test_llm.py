#!/usr/bin/env python3
"""验证 .env 中 LLM 配置可用性（key/base/model），不显示 key 本体。

用法: python scripts/test_llm.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.llm.client import LLMClient  # noqa: E402


def main() -> int:
    print("=" * 50)
    print("LLM 连接测试")
    print("=" * 50)
    ok = LLMClient().test_connection()
    print("=" * 50)
    print("连接正常 ✔" if ok else "连接失败 ✗（检查 .env 的 model/key/base_url）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
