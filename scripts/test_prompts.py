#!/usr/bin/env python3
"""prompts/client 纯逻辑单测：全局 Triton guideline 加载 + LLM thinking 开关。

core 组：不 import torch、不联网（mock urlopen）。
覆盖 55f89f4(guideline) / 44b257d(thinking) 两个能力，防止后续改动破坏。
用法: python scripts/test_prompts.py
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest.mock as um

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.llm import client as client_mod  # noqa: E402
from agent.llm import prompts  # noqa: E402


def _chat_with_capture(c, messages):
    """mock urlopen 发起一次 chat，返回 (text, request_body)。"""
    captured: dict = {}

    @contextlib.contextmanager
    def fake_urlopen(req, timeout=180):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        payload = {"choices": [{"message": {"content": "ok"}}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 5}}

        class _Resp:
            def read(self):
                return json.dumps(payload).encode("utf-8")
        yield _Resp()

    with um.patch.object(client_mod.urllib.request, "urlopen", fake_urlopen):
        text, usage = c.chat(messages, temperature=0.0, max_tokens=8)
    return text, captured["body"]


def main() -> int:
    failed = 0

    def check(label, cond):
        nonlocal failed
        print(f"{'✔' if cond else '✗'} {label}")
        if not cond:
            failed += 1

    # --- 1. 全局 guideline 默认注入 ---
    sp = prompts.SYSTEM_PROMPT
    check("SYSTEM_PROMPT 含 guideline", "ADDITIONAL TRITON KERNEL GUIDELINES" in sp)
    check("guideline 模板文件存在", os.path.exists(prompts._GUIDELINES_DEFAULT))
    low = sp.lower()
    check("guideline 含家族坑(conv)+Reflexion", "conv" in low and "reflexion" in low)

    # --- 2. override env（TRITON_GUIDELINES_PATH）---
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("MY CUSTOM GUIDELINE")
        custom = f.name
    try:
        os.environ["TRITON_GUIDELINES_PATH"] = custom
        check("override env 生效", prompts._load_triton_guidelines() == "MY CUSTOM GUIDELINE")
        os.environ["TRITON_GUIDELINES_PATH"] = "/nonexistent/guidelines.txt"
        check("文件缺失返回空不崩", prompts._load_triton_guidelines() == "")
    finally:
        os.environ.pop("TRITON_GUIDELINES_PATH", None)
        os.unlink(custom)

    # --- 3. thinking env 解析（DEEPSEEK_THINKING）---
    for val, exp in [("off", False), ("0", False), ("disabled", False),
                     ("no", False), ("on", True), ("1", True), ("enabled", True)]:
        os.environ["DEEPSEEK_THINKING"] = val
        c = client_mod.LLMClient(api_key="k", base_url="http://x", model="m")
        check(f"thinking env={val} -> {exp}", c.thinking is exp)
    os.environ.pop("DEEPSEEK_THINKING", None)
    c = client_mod.LLMClient(api_key="k", base_url="http://x", model="m")
    check("无 env → thinking None(服务端默认)", c.thinking is None)

    # --- 4. chat body 注入 thinking 字段 + content/usage 解析 ---
    c0 = client_mod.LLMClient(api_key="k", base_url="http://x", model="m")  # None
    text, body = _chat_with_capture(c0, [{"role": "user", "content": "hi"}])
    check("thinking=None 不注入", "thinking" not in body)
    check("返回 content 与 usage", text == "ok" and body["max_tokens"] == 8)

    c1 = client_mod.LLMClient(api_key="k", base_url="http://x", model="m")
    c1.thinking = False
    _t, body = _chat_with_capture(c1, [{"role": "user", "content": "hi"}])
    check("thinking=False → thinking disabled", body.get("thinking") == {"type": "disabled"})

    c2 = client_mod.LLMClient(api_key="k", base_url="http://x", model="m")
    c2.thinking = True
    _t, body = _chat_with_capture(c2, [{"role": "user", "content": "hi"}])
    check("thinking=True → thinking enabled", body.get("thinking") == {"type": "enabled"})

    print("-" * 50)
    print("prompts 单测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
