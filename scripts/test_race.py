"""多 seed 竞速 / digest 缓存 逻辑自测（不烧 GPU、不调真实 LLM）。

验证三条机制：
  T1 digest 缓存：两个 agent 生成相同代码 → 沙箱只跑一次（后者命中缓存）。
  T2 stop_event 早停：外部已置位 → agent 不发 LLM、立即 interrupted 退出。
  T3 早停后 summary 正确（success=False + interrupted=True，rounds=0）。

运行：python scripts/test_race.py   （需在 triton_env：import torch/triton 的模块）
"""
from __future__ import annotations

import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import memory as _mem  # noqa: E402
from agent.llm import prompts  # noqa: E402
from agent.loop import KernelAgent  # noqa: E402
from agent.tools import executor as _executor_mod  # noqa: E402
from agent.tools.executor import ExecReport  # noqa: E402

# —— 一份能过静态闸门的合法 kernel 代码（不真跑，executor 被 fake） ——
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


class FakeClient:
    """按序吐 code 的假 LLM。"""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.model = "fake"
        self.chat_calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):
        self.chat_calls += 1
        text = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        return text, {"prompt_tokens": 1, "completion_tokens": 1}


def _patch(monkey_memory: bool = True):
    """替换 executor.run / memory.add_success 为无害 fake。返回 (记录器, 恢复函数)。"""
    calls: list[str] = []

    def fake_run(op_name, code, timeout=90.0, keep_script=True, perf=False):
        calls.append(code)
        # 默认判 correctness 失败（避免碰 GPU）；测试可按需覆盖为 pass
        return ExecReport(status="correctness", ok=False,
                          message="数值不对齐 (fake)", max_abs_err=1.0,
                          stdout="", stderr="", wall_s=0.01)

    orig_run, _executor_mod.run = _executor_mod.run, fake_run
    orig_add = _mem.add_success
    if monkey_memory:
        _mem.add_success = lambda *a, **k: ""   # 不污染真实经验库
    def _restore():
        _executor_mod.run = orig_run
        if monkey_memory:
            _mem.add_success = orig_add
    return calls, _restore


def _check_static_ok(code: str) -> None:
    from agent.tools import static_check
    assert static_check.check_generated_code(code, op_name="vector_add")["ok"], \
        "GOOD_CODE 应能过静态闸门（否则用例无效）"


def test_cache_hit():
    """T1：两 agent 首轮生成相同代码 → 沙箱只跑一次。"""
    calls, restore = _patch()
    try:
        cache: dict = {}
        code = "```python\n" + GOOD_CODE + "\n```"   # 经 extract 后 == GOOD_CODE.strip()
        # 两个 agent 各 2 轮：首轮同失败代码，第二轮各自不同成功代码
        cli = FakeClient([code, code])
        a1 = KernelAgent(max_rounds=3, verbose=False, client=cli)
        a1.run("vector_add", save=False, code_cache=cache)     # 首轮 miss → 跑沙箱
        calls_before = len(calls)
        a2 = KernelAgent(max_rounds=3, verbose=False, client=cli)
        s2, _ = a2.run("vector_add", save=False, code_cache=cache)  # 首轮相同 code → 命中缓存
        assert len(calls) == calls_before, \
            f"相同代码应命中缓存不重复跑沙箱: calls={len(calls)}"
        assert s2["cache_hits"] >= 1, f"cache_hits={s2['cache_hits']} 应 ≥1"
        print(f"[ok ] T1 digest 缓存：相同代码第二次沙箱调用数为 0，cache_hits={s2['cache_hits']}")
    finally:
        restore()


def test_stop_preset():
    """T2：stop_event 预置 → 不发 LLM、立即 interrupted。"""
    calls, restore = _patch()
    try:
        stop = threading.Event()
        stop.set()
        cli = FakeClient(["```python\n" + GOOD_CODE + "\n```"])
        agent = KernelAgent(max_rounds=5, verbose=False, client=cli)
        s, steps = agent.run("vector_add", save=False, stop_event=stop)
        assert cli.chat_calls == 0, "stop 预置时应一发 LLM 都不调"
        assert len(steps) == 0 and s["rounds_used"] == 0
        assert s["interrupted"] is True and s["success"] is False
        print("[ok ] T2 stop_event 早停：不调 LLM、rounds=0、interrupted=True")
    finally:
        restore()


def test_first_success_stops_others():
    """T3：线程级竞速 —— seedA 立即成功置位，seedB 被早停、不再耗尽预算。"""
    calls, restore = _patch()
    # 让 executor 对 seedA 的成功代码返回 pass
    from agent.tools.executor import ExecReport as ER
    done: dict[str, list[str]] = {}
    def fake_run(op_name, code, timeout=90.0, keep_script=True, perf=False):
        calls.append(code)
        if op_name == "vector_add" and "SUCCESS_SEED_A" in code:
            return ER(status="pass", ok=True, message="PASS", max_abs_err=0.0,
                      stdout="", stderr="", wall_s=0.01)
        return ER(status="correctness", ok=False, message="fail(fake)",
                  max_abs_err=1.0, stdout="", stderr="", wall_s=0.01)
    orig_run, _executor_mod.run = _executor_mod.run, fake_run
    try:
        stop = threading.Event()
        cache: dict = {}
        a_code = "```python\n# SUCCESS_SEED_A\n" + GOOD_CODE + "\n```"
        b_code = "```python\n" + GOOD_CODE + "\n```"
        out: dict[str, dict] = {}

        def worker(idx: int, resp: list[str]) -> None:
            cli = FakeClient(resp)
            ag = KernelAgent(max_rounds=6, verbose=False, client=cli)
            s, _st = ag.run("vector_add", save=False, stop_event=stop, code_cache=cache)
            out[idx] = {"s": s, "chat": cli.chat_calls}
            if s["success"]:
                stop.set()

        ta = threading.Thread(target=worker, args=(0, [a_code]), daemon=True)   # A: 一轮就成功
        tb = threading.Thread(target=worker, args=(1, [b_code] * 10), daemon=True)  # B: 一直失败
        ta.start()
        # 给 A 一点先手，避免两个都抢第一轮（不影响结论，只影响确定性）
        import time
        time.sleep(0.05)
        tb.start()
        ta.join(); tb.join()

        assert out[0]["s"]["success"] is True, "seedA 应成功"
        # B 若没被早停会一直跑到 6 轮；被停后 chat 应显著 < 6
        b_chat = out[1]["chat"]
        assert b_chat < 6, f"seedB 应被早停，却调了 {b_chat} 次 LLM"
        assert out[1]["s"]["interrupted"] is True, "seedB 应标记 interrupted"
        assert out[1]["s"]["success"] is False
        print(f"[ok ] T3 竞速早停：seedA 成功，seedB 仅 {b_chat} 次 LLM 即被停")
    finally:
        _executor_mod.run = orig_run


if __name__ == "__main__":
    _check_static_ok(GOOD_CODE)
    test_cache_hit()
    test_stop_preset()
    test_first_success_stops_others()
    print("\n竞速/缓存自测全部通过 ✔")
