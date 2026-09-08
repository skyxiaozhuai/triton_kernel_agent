"""orchestrator 主循环 —— 最小闭环 v1（自写，不套 LangGraph）。

流程：
  Planner 暂并入 Coder（直接给完整算子规格让 LLM 写）—— D6 再拆成独立角色。
  每轮：LLM 生成代码 -> 沙箱执行 -> 结构化反馈(error_parser)
        -> 未通过则把(上一版代码+反馈)回填给 LLM 再试 —— Reflexion。
  终止：数值通过(pass) 或 达到 max_rounds。

可观测：每步(轮)连同 code/status/feedback/token 落 results/*.jsonl。
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import time

from agent import memory
from agent.llm import prompts
from agent.llm.client import LLMClient
from agent.tools import error_parser, executor, static_check
from benchmarks import ops_registry

RESULTS_DIR = os.path.join(executor.PROJECT_ROOT, "results")


@dataclasses.dataclass
class AgentStep:
    round: int
    status: str
    code: str
    feedback: str
    max_abs_err: float | None
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    reply_len: int = 0
    perf: dict | None = None     # 性能测量(仅 perf_mode 跑过才有)


class KernelAgent:
    def __init__(self, max_rounds: int = 6, max_tokens: int = 8192,
                 temperature: float = 0.2, verbose: bool = True,
                 perf_mode: bool = False, perf_min_speedup: float = 0.9,
                 perf_retry: int = 2, memory_mode: bool = False,
                 client: LLMClient | None = None, log_prefix: str = ""):
        self.client = client if client is not None else LLMClient()
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.log_prefix = log_prefix        # 多 seed 竞速时区分日志（如 [seed0]）
        self.perf_mode = perf_mode          # 性能 critic 开关
        self.perf_min_speedup = perf_min_speedup
        self.perf_retry = perf_retry
        self.memory_mode = memory_mode      # RAG 经验库检索开关
        self._memory_used: list[str] = []
        self._fix_used = 0                  # 失败回灌：注入历史同类修复示范次数

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"{self.log_prefix}{msg}", flush=True)

    def run(self, op_name: str, save: bool = True,
            stop_event=None, code_cache: dict | None = None
            ) -> tuple[dict, list[AgentStep]]:
        """运行一次闭环。

        多 seed 竞速扩展：
          stop_event: threading.Event —— 其它 seed 已成功时置位，本 run 每轮检查并早停；
          code_cache: dict[(op_name, sha256(code))] -> {status,max_err,feedback,ok}
                      —— 共享验证结果，避免多个 seed 生成同一代码重复烧 GPU。
                      （perf_mode 下禁用缓存，因为性能测量不走缓存。）
        """
        op = ops_registry.get_op(op_name)
        refs = []
        if self.memory_mode:
            refs = memory.retrieve(op_name, k=2)
            self._memory_used = [r["op"] for r in refs]
            if refs:
                self._log(f"[agent] RAG: 注入 {len(refs)} 个同类参考 "
                          f"({self._memory_used})")
        messages = prompts.build_initial_messages(op.OP_META, refs=refs)
        steps: list[AgentStep] = []
        t_start = time.time()
        tokens = {"prompt": 0, "completion": 0}

        self._log(f"[agent] 任务: {op_name} | 模型: {self.client.model} | "
                  f"max_rounds: {self.max_rounds}")

        last_ok = False
        step_perf_tries = 0
        cache_hits = 0
        interrupted = False
        for rnd in range(1, self.max_rounds + 1):
            # 竞速早停：其它 seed 已成功 → 本 seed 立即退出（不浪费预算）
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                self._log("[agent] 收到外部停止信号（其它 seed 已成功），提前退出")
                break
            t0 = time.time()
            rep = None   # 沙箱报告(仅 executor 分支赋值；用于 error 细分类失败回灌)
            self._log(f"[round {rnd}/{self.max_rounds}] 调用 LLM 生成 ...")
            text, usage = self.client.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens)
            tokens["prompt"] += usage.get("prompt_tokens", 0)
            tokens["completion"] += usage.get("completion_tokens", 0)

            code = prompts.extract_python_code(text)

            # 有效性快速检查：空代码 / 缺 launch 直接反馈，不浪费一轮沙箱
            valid, invalid_reason = prompts.check_code_valid(code)
            gate_category = None
            if valid:
                # 静态闸门：结构 + 反作弊（借鉴 PyTorch KernelAgent，AST 精确分析，不烧 GPU）
                sc = static_check.check_generated_code(code, op_name=op_name)
                if not sc["ok"]:
                    valid, invalid_reason = False, sc["reason"]
                    gate_category = sc["category"]
            if valid:
                if code_cache is not None and not self.perf_mode:
                    # 多 seed 竞速：按代码 digest 共享验证结果，避免重复烧 GPU
                    key = (op_name, hashlib.sha256(code.encode("utf-8")).hexdigest())
                    hit = code_cache.get(key)
                    if hit is not None:
                        cache_hits += 1
                        status = hit["status"]
                        max_err = hit["max_err"]
                        feedback = hit["feedback"]
                        ok = hit["ok"]
                        self._log(f"[round {rnd}] 命中共享代码缓存，跳过沙箱 "
                                  f"(status={status})")
                    else:
                        self._log(f"[round {rnd}] 沙箱执行 ...")
                        rep = executor.run(op_name, code)
                        status = rep.status
                        max_err = rep.max_abs_err
                        feedback = error_parser.to_feedback_text(rep)
                        ok = rep.ok
                        code_cache[key] = {"status": status, "max_err": max_err,
                                           "feedback": feedback, "ok": ok}
                else:
                    self._log(f"[round {rnd}] 沙箱执行 ...")
                    rep = executor.run(op_name, code)
                    status = rep.status
                    max_err = rep.max_abs_err
                    feedback = error_parser.to_feedback_text(rep)
                    ok = rep.ok
            else:
                self._log(f"[round {rnd}] 代码未通过检查，跳过沙箱")
                status = ("invalid_code" if gate_category is None
                          else f"static_{gate_category}")
                max_err = None
                feedback = (invalid_reason if gate_category is None
                            else f"❌ 静态闸门未通过：{invalid_reason}")
                ok = False

            step = AgentStep(
                round=rnd, status=status, code=code, feedback=feedback,
                max_abs_err=max_err,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                wall_s=round(time.time() - t0, 2),
                reply_len=len(text))
            steps.append(step)

            tag = "✔" if ok else "✗"
            err = f"  max_abs_err={max_err:.3e}" if max_err is not None else ""
            self._log(f"[round {rnd}] {tag} status={status}{err}")

            # 失败回灌：本步失败类别键(供检索其它算子同类错误的历史修复示范)
            if ok:
                fix_cat = None
            elif status == "error" and rep is not None:
                fix_cat = error_parser.classify(rep)["category"]
            else:
                fix_cat = status

            if ok:
                # —— 性能 critic（可选，仅 perf_mode）——
                if self.perf_mode and status == "pass":
                    p = executor.run(op_name, code, perf=True)
                    steps[-1].perf = p.perf
                    spd = (p.perf or {}).get("speedup_vs_eager")
                    self._log(f"[round {rnd}] perf: speedup_vs_eager={spd} "
                              f"(min={self.perf_min_speedup})")
                    if (spd is not None and spd < self.perf_min_speedup
                            and step_perf_tries < self.perf_retry):
                        step_perf_tries += 1
                        fb = prompts.perf_feedback_user_message(
                            launch_ms=p.perf.get("launch_ms"),
                            eager_ms=p.perf.get("eager_ms"),
                            speedup=spd, min_speedup=self.perf_min_speedup)
                        self._log(f"[round {rnd}] 性能未达标，进入优化轮 ...")
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": fb})
                        continue
                last_ok = True
                self._log(f"[agent] ✔ {op_name} 在第 {rnd} 轮通过！")
                try:
                    memory.add_success(op_name, code, rounds=rnd)   # 积累经验库
                    memory.record_fix_pair(op_name, steps)          # 失败回灌: 记修复对
                except Exception:  # noqa: BLE001 —— 记忆写入失败不影响结果
                    self._log("[agent] (warn) 写入经验库失败")
                break

            # 失败样本回灌：同类错误的历史修复示范作为 few-shot 修法注入
            if self.memory_mode and fix_cat:
                _fix = memory.retrieve_fix(fix_cat, exclude_op=op_name, k=1)
                if _fix:
                    self._fix_used += 1
                    self._log(f"[agent] 失败回灌: 注入 1 条同类错误({fix_cat})修复示范"
                              f"(来自 {_fix['op']})")
                    messages.append({"role": "user",
                                     "content": memory.format_fix_ref(_fix)})

            # Reflexion：把上一版代码 + 结构化反馈追加进对话
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": prompts.feedback_user_message(feedback)})

        summary = {
            "op": op_name,
            "success": last_ok,
            "rounds_used": len(steps),
            "total_tokens": tokens,
            "wall_s": round(time.time() - t_start, 2),
            "final_status": steps[-1].status if steps else "no_run",
            "final_max_abs_err": steps[-1].max_abs_err if steps else None,
            "final_speedup_vs_eager": ((steps[-1].perf or {}).get("speedup_vs_eager")
                                        if steps and steps[-1].perf else None),
            "memory_used": self._memory_used,
            "cache_hits": cache_hits,
            "fix_refs_used": self._fix_used,
            "interrupted": interrupted,
        }
        if save:
            self._save(op_name, steps, summary)
        return summary, steps

    def _save(self, op_name: str, steps: list[AgentStep], summary: dict) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"traj_{op_name}_{ts}.jsonl")
        n = 1
        while os.path.exists(path):   # 多 seed 竞速同秒并行 → 防撞名覆盖
            path = os.path.join(RESULTS_DIR, f"traj_{op_name}_{ts}_{n}.jsonl")
            n += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            for st in steps:
                f.write(json.dumps(dataclasses.asdict(st), ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[agent] 轨迹已保存: {path}")
