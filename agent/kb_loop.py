"""KernelBench agent 闭环 —— 对官方 KernelBench 题目跑"生成→判卷→Reflexion"。

与 KernelAgent（agent/loop.py）同构，但规格来自题目源码、判卷用 eager forward 当 golden：
    规格(题目 .py 源码 + 硬契约) -> LLM 生成 kernel+def launch(*inputs)
    -> 静态闸门(AST 结构+反作弊) -> 子进程判卷(executor_kb，多 case)
    -> error_parser 反馈 -> Reflexion 回填，直到 pass 或 max_rounds。

设计取舍：KernelBench 是"外部题目"，不注册进我们的 op registry / 正样本 RAG
（category 体系不同、防污染），故本闭环暂不带 memory 注入；多 seed 竞速/失败回灌
是通用能力，如需可在 run_problem 复用（当前单 seed，服务器批量时再扩展）。
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import time

from agent.llm import prompts
from agent.llm.client import LLMClient
from agent.loop import AgentStep
from agent.tools import error_parser, static_check
from agent.tools import executor_kb
from agent.tools.executor import PROJECT_ROOT
from benchmarks.kernelbench import problem as kb_problem

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


class KernelBenchAgent:
    def __init__(self, max_rounds: int = 6, max_tokens: int = 8192,
                 temperature: float = 0.2, verbose: bool = True,
                 client: LLMClient | None = None, log_prefix: str = ""):
        self.client = client if client is not None else LLMClient()
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.log_prefix = log_prefix

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"{self.log_prefix}{msg}", flush=True)

    def run_problem(self, problem_path: str, num_cases: int = 2,
                    save: bool = True) -> tuple[dict, list[AgentStep]]:
        prob = kb_problem.load_problem(problem_path)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prob.spec_text()},
        ]
        steps: list[AgentStep] = []
        tokens = {"prompt": 0, "completion": 0}
        t_start = time.time()
        self._log(f"[kb] 题目: {prob.name} (level {prob.level}) | 模型: "
                  f"{self.client.model} | max_rounds: {self.max_rounds}")

        last_ok = False
        for rnd in range(1, self.max_rounds + 1):
            t0 = time.time()
            self._log(f"[round {rnd}/{self.max_rounds}] 调用 LLM 生成 ...")
            text, usage = self.client.chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens)
            tokens["prompt"] += usage.get("prompt_tokens", 0)
            tokens["completion"] += usage.get("completion_tokens", 0)
            code = prompts.extract_python_code(text)

            valid, invalid_reason = prompts.check_code_valid(code)
            gate_category = None
            if valid:
                sc = static_check.check_generated_code(code)
                if not sc["ok"]:
                    valid, invalid_reason = False, sc["reason"]
                    gate_category = sc["category"]
            if valid:
                self._log(f"[round {rnd}] 沙箱判卷 (cases={num_cases}) ...")
                rep = executor_kb.run_problem(problem_path, code,
                                              num_cases=num_cases)
                status, max_err, ok = rep.status, rep.max_abs_err, rep.ok
                feedback = error_parser.to_feedback_text(rep)
            else:
                self._log(f"[round {rnd}] 代码未过检查，跳过沙箱")
                status = ("invalid_code" if gate_category is None
                          else f"static_{gate_category}")
                max_err = None
                feedback = (invalid_reason if gate_category is None
                            else f"❌ 静态闸门未通过：{invalid_reason}")
                ok = False

            steps.append(AgentStep(
                round=rnd, status=status, code=code, feedback=feedback,
                max_abs_err=max_err,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                wall_s=round(time.time() - t0, 2), reply_len=len(text)))

            tag = "✔" if ok else "✗"
            err = f"  max_abs_err={max_err:.3e}" if max_err is not None else ""
            self._log(f"[round {rnd}] {tag} status={status}{err}")
            if ok:
                last_ok = True
                self._log(f"[kb] ✔ {prob.name} 在第 {rnd} 轮通过！")
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": prompts.feedback_user_message(feedback)})

        summary = {
            "op": f"kernelbench/{prob.name}",
            "success": last_ok,
            "rounds_used": len(steps),
            "total_tokens": tokens,
            "wall_s": round(time.time() - t_start, 2),
            "final_status": steps[-1].status if steps else "no_run",
            "final_max_abs_err": steps[-1].max_abs_err if steps else None,
        }
        if save:
            self._save(prob.name, steps, summary)
        return summary, steps

    def _save(self, name: str, steps: list[AgentStep], summary: dict) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"traj_kb_{name}_{ts}.jsonl")
        n = 1
        while os.path.exists(path):
            path = os.path.join(RESULTS_DIR, f"traj_kb_{name}_{ts}_{n}.jsonl")
            n += 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            for st in steps:
                f.write(json.dumps(dataclasses.asdict(st), ensure_ascii=False) + "\n")
        self._log(f"[kb] 轨迹已保存: {path}")
