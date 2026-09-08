"""失败样本回灌 v1 自测：memory 修复对存储/检索 + loop 端到端注入。

- 单元：normalize_status / record_fix_pair / retrieve_fix / format_fix_ref
- 端到端：agent 一次 compile 失败后，注入其它算子同类错误修复示范，再成功
所有测试把 memory 目录重定向到临时目录，不污染真实 results/memory。

运行：python scripts/test_failure_memory.py
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import memory as _mem  # noqa: E402
from agent.loop import AgentStep, KernelAgent  # noqa: E402
from agent.tools import executor as _ex  # noqa: E402
from agent.tools.executor import ExecReport  # noqa: E402

GOOD_CODE = '''
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


def _step(status, code, feedback=""):
    return AgentStep(round=1, status=status, code=code, feedback=feedback,
                     max_abs_err=None, prompt_tokens=0, completion_tokens=0,
                     wall_s=0.1)


def test_unit(tmp: str) -> None:
    print("== A) memory 单元：记录 / 归一化 / 检索 ==")
    _mem.MEMORY_DIR = tmp
    _mem.FAILURES_DIR = os.path.join(tmp, "failures")

    ok = _step("pass", "GOOD_A")
    # opA：一例 structure 类失败(invalid_code)，经 feedback 无细分类 → status
    pA = _mem.record_fix_pair("opA", [_step("invalid_code", "badA", "缺 launch"), ok])
    assert pA and os.path.exists(pA), "opA 修复对应已落盘"
    # opB：compile 类失败(从 feedback 提取细分类)
    okB = _step("pass", "GOOD_B")
    badB = _step("error", "badB",
                 "❌ 类别: compile\n摘要: CompilationError ...")
    pB = _mem.record_fix_pair("opB", [badB, okB])
    assert os.path.exists(pB)
    # 归一化
    assert _mem.normalize_status("invalid_code") == "structure"
    assert _mem.normalize_status("static_cheat") == "cheat"
    assert _mem.normalize_status("static_structure") == "structure"
    assert _mem.normalize_status("compile") == "compile"
    # 检索：compile 匹配 opB；structure 匹配 opA（此时不排除）
    assert (_mem.retrieve_fix("compile") or {}).get("op") == "opB"
    assert (_mem.retrieve_fix("invalid_code") or {}).get("op") == "opA"
    # 排除同 op + 无匹配 → None
    assert _mem.retrieve_fix("compile", exclude_op="opB") is None
    assert _mem.retrieve_fix("compile", exclude_op="opA") is not None
    txt = _mem.format_fix_ref({"op": "opB", "err_category": "compile",
                               "rounds": 3, "good_code": "GOOD_B"})
    assert "opB" in txt and "GOOD_B" in txt
    print("[ok ] 单元测试通过")


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.model = "fake"
        self.chat_calls = 0
        self.seen_messages: list = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.chat_calls += 1
        self.seen_messages.append(list(messages))   # 浅拷贝快照，供断言注入
        text = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        return text, {"prompt_tokens": 1, "completion_tokens": 1}


def test_loop_injection(tmp: str) -> None:
    print("== B) 端到端：agent compile 失败 → 注入历史同类修复示范 → 再成功 ==")
    _mem.MEMORY_DIR = tmp
    _mem.FAILURES_DIR = os.path.join(tmp, "failures")

    # 预置另一算子 other_compile 的 compile 修复对（其它 op 才能被检索注入）
    _mem.record_fix_pair(
        "other_compile",
        [_step("error", "badC", "❌ 类别: compile\nCompilationError"),
         _step("pass", "GOOD_FROM_OTHER")])

    calls: list[str] = []

    def fake_run(op_name, code, timeout=90.0, keep_script=True, perf=False):
        calls.append(code)
        if "FAIL_ME" in code:
            return ExecReport(status="error", ok=False,
                              message="CompilationError: FAIL_ME 编译失败",
                              max_abs_err=None, stdout="", stderr="",
                              wall_s=0.01)
        return ExecReport(status="pass", ok=True, message="PASS",
                          max_abs_err=0.0, stdout="", stderr="", wall_s=0.01)

    orig_run, _ex.run = _ex.run, fake_run
    try:
        code1 = "```python\n# FAIL_ME\n" + GOOD_CODE + "\n```"
        cli = FakeClient([code1, "```python\n" + GOOD_CODE + "\n```"])
        agent = KernelAgent(max_rounds=4, verbose=False, memory_mode=True,
                            client=cli)
        s, _st = agent.run("vector_add", save=False)
        assert s["success"] is True, "注入后应成功"
        assert cli.chat_calls == 2, f"应正好 2 轮 LLM，实际 {cli.chat_calls}"
        assert s["fix_refs_used"] >= 1, f"应注入 ≥1 次修复示范: {s['fix_refs_used']}"
        # 第二次调用传入的 messages 里应含“历史同类错误修复示范”且指向 other_compile
        second_msgs = cli.seen_messages[1]
        joined = "\n".join(m.get("content", "") for m in second_msgs
                           if m.get("role") == "user")
        assert "历史同类错误修复示范" in joined, "第二轮应注入修复示范"
        assert "other_compile" in joined
        print(f"[ok ] 端到端：compile 失败 → 注入 other_compile 修复示范，"
              f"fix_refs_used={s['fix_refs_used']}，2 轮成功")
    finally:
        _ex.run = orig_run


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_unit(d)
    with tempfile.TemporaryDirectory() as d:
        test_loop_injection(d)
    print("\n失败回灌自测全部通过 ✔")
