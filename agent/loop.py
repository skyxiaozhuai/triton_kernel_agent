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
import json
import os
import time

from agent.llm import prompts
from agent.llm.client import LLMClient
from agent.tools import error_parser, executor
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
                 perf_mode: bool = False, perf_min_speedup: float = 0.7,
                 perf_retry: int = 2):
        self.client = LLMClient()
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.perf_mode = perf_mode          # 性能 critic 开关
        self.perf_min_speedup = perf_min_speedup
        self.perf_retry = perf_retry

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self, op_name: str, save: bool = True) -> tuple[dict, list[AgentStep]]:
        op = ops_registry.get_op(op_name)
        messages = prompts.build_initial_messages(op.OP_META)
        steps: list[AgentStep] = []
        t_start = time.time()
        tokens = {"prompt": 0, "completion": 0}

        self._log(f"[agent] 任务: {op_name} | 模型: {self.client.model} | "
                  f"max_rounds: {self.max_rounds}")

        last_ok = False
        step_perf_tries = 0
        for rnd in range(1, self.max_rounds + 1):
            t0 = time.time()
            self._log(f"[round {rnd}/{self.max_rounds}] 调用 LLM 生成 ...")
            text, usage = self.client.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens)
            tokens["prompt"] += usage.get("prompt_tokens", 0)
            tokens["completion"] += usage.get("completion_tokens", 0)

            code = prompts.extract_python_code(text)

            # 有效性快速检查：空代码 / 缺 launch 直接反馈，不浪费一轮沙箱
            valid, invalid_reason = prompts.check_code_valid(code)
            if valid:
                self._log(f"[round {rnd}] 沙箱执行 ...")
                rep = executor.run(op_name, code)
                status = rep.status
                max_err = rep.max_abs_err
                feedback = error_parser.to_feedback_text(rep)
                ok = rep.ok
            else:
                self._log(f"[round {rnd}] 代码无效(空/缺 launch)，跳过沙箱")
                status, max_err, feedback, ok = "invalid_code", None, invalid_reason, False

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
                break

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
        }
        if save:
            self._save(op_name, steps, summary)
        return summary, steps

    def _save(self, op_name: str, steps: list[AgentStep], summary: dict) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"traj_{op_name}_{ts}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            for st in steps:
                f.write(json.dumps(dataclasses.asdict(st), ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[agent] 轨迹已保存: {path}")
