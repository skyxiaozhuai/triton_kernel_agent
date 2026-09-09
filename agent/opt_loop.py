"""优化端闭环（KernelOptimizer）—— 借鉴 KernelAgent 优化端的轻量实现。

对一个"已正确"的 kernel（来自经验库 / 文件），进入 hardware-guided 优化循环：
    剖析(NCU roofline) → 反馈给 LLM → 生成优化版 → 验证正确+do_bench
    → 更快则接受(重剖析) / 否则记 rejected → 连续无改进收敛 → best + 曲线

设计要点：
- 消息不累积长历史：每轮重建 system + user（规格 + 当前 best 代码 + ncu 剖析 +
  上一轮结论），单发 LLM —— 省 token、可控（官方亦用最近 attempts 块式）。
- 接受阈值 improve_min（相对 ms 改进，默认 2%），避免 do_bench 噪声误接受；
- ncu 剖析只在 best 变化时跑一次（每次 ~30s），rejected 轮复用当前剖析。
"""
from __future__ import annotations

from collections import deque

import datetime
import json
import os
import time

from agent import memory
from agent.llm import prompts
from agent.llm.client import LLMClient
from agent.tools import error_parser, executor, ncu_profiler, static_check
from benchmarks import ops_registry

RESULTS_DIR = os.path.join(executor.PROJECT_ROOT, "results")


class KernelOptimizer:
    def __init__(self, max_tokens: int = 16384, temperature: float = 0.2,
                 verbose: bool = True, client: LLMClient | None = None,
                 log_prefix: str = "", attempt_window: int = 4,
                 empty_retries: int = 2):
        self.client = client if client is not None else LLMClient()
        self.attempt_window = attempt_window   # 最近被拒尝试注入窗口(对齐官方 attempt_history)
        self.empty_retries = empty_retries     # 空代码(截断)自动重试次数
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.log_prefix = log_prefix

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"{self.log_prefix}{msg}", flush=True)

    # ---------- 评估：静态闸门 + 正确性 + 性能 ----------
    def _eval(self, op, code: str) -> dict:
        sc = static_check.check_generated_code(code, op_name=op.OP_META["name"])
        if not sc["ok"]:
            return {"ok": False, "why": f"静态闸门未过: {sc['reason']}"}
        rep = executor.run(op.OP_META["name"], code, perf=True)
        if not rep.ok:
            return {"ok": False,
                    "why": error_parser.to_feedback_text(rep)}
        return {"ok": True,
                "ms": float(rep.perf["launch_ms"]),
                "spd": float(rep.perf["speedup_vs_eager"])}

    def _profile(self, op_name: str, code: str) -> str | None:
        self._log("  [ncu] 剖析当前 best ...")
        prof = ncu_profiler.profile(op_name, code)
        return ncu_profiler.format_feedback(prof) if prof else None

    @staticmethod
    def _short_spec(op) -> str:
        """优化模式用精简规格（省略长 description，避免 prompt 过长挤占输出预算）。"""
        meta = op.OP_META
        lines = [f"- name: {meta.get('name')}",
                 f"- signature: {meta.get('signature')}",
                 f"- dtype: {meta.get('dtype')}"]
        if meta.get("notes"):
            lines.append(f"- notes: {meta['notes']}")
        if meta.get("launch_sig"):
            lines.append(f"- launch_sig: {meta['launch_sig']}")
        return "Operator semantics:\n" + "\n".join(lines) + "\n" + prompts.gpu_context_note()

    @staticmethod
    def _avoid_for(status: str, note: str) -> str:
        """从被拒结果生成一条确定性 AVOID 教训（对齐官方 reflexion 的 avoid 通道）。"""
        if status == "not_faster":
            return ("避免无实质改进的微调(do_bench 噪声)；要基于剖析改变访存/并行度/"
                    "BLOCK/tile，而不是换 num_warps 碰运气")
        return (note or "")[:200]

    @staticmethod
    def _fmt_attempts(attempts) -> str:
        """把最近被拒尝试压成截断文本（对齐官方 attempt_history 注入）。"""
        if not attempts:
            return ""
        parts = ["== 最近被拒的尝试（请在 best 上避免重犯；代码已截断）=="]
        for a in attempts:
            ms_s = f" ms={a.get('ms'):.4f}" if a.get("ms") is not None else ""
            parts.append(f"- round{a.get('round')} status={a.get('status')}{ms_s}")
            code = a.get("code") or ""
            parts.append("  ```python\n  " + code[:600]
                         + ("..." if len(code) > 600 else "") + "\n  ```")
            parts.append("  原因: " + (a.get("note") or "")[:200])
            if a.get("avoid"):
                parts.append("  AVOID: " + a["avoid"][:180])
        return "\n".join(parts)

    def _opt_user(self, op, best_code: str, best_ms: float, spd: float,
                  attempts, prof_text: str | None) -> str:
        spec = self._short_spec(op)
        parts = [spec, "\n=== 当前最佳代码（请在此基础上改进）===",
                 f"```python\n{best_code}\n```",
                 f"\n当前最佳: launch_ms={best_ms:.4f}ms  "
                 f"speedup_vs_eager={spd:.3f}x"]
        attempts_txt = self._fmt_attempts(attempts)
        if attempts_txt:
            parts.append("\n" + attempts_txt)
        if prof_text:
            parts.append("\n" + prof_text)
        parts += ["\n请让上面的 kernel 更快且保持数值正确（可调整 BLOCK/grid/num_warps/向量化/复用）。",
                  "直接输出一个 ```python 代码块（含 import、@triton.jit kernel、def launch）。",
                  "不要输出代码块以外的解释。"]
        return "\n".join(parts)

    # ---------- 主入口 ----------
    def optimize(self, op_name: str, init_code: str | None = None,
                 opt_rounds: int = 6, stall_limit: int = 2,
                 improve_min: float = 0.02, use_ncu: bool = True,
                 save: bool = True) -> dict:
        op = ops_registry.get_op(op_name)
        if init_code is None:
            rec = memory.get(op_name)
            if not rec:
                return {"ok": False,
                        "error": f"经验库无 {op_name} 的成功代码；先 run_agent 或给 --code-file"}
            init_code = rec["code"]

        # 初始代码必须正确且能测出基线
        ev0 = self._eval(op, init_code)
        if not ev0["ok"]:
            return {"ok": False, "error": f"初始代码不正确: {ev0['why']}"}
        best, best_ms, best_spd = init_code, ev0["ms"], ev0["spd"]
        self._log(f"[opt] {op_name} 基线: launch_ms={best_ms:.4f}ms "
                  f"(speedup_vs_eager={best_spd:.3f})")

        prof_text = self._profile(op_name, best) if use_ncu else None
        curve = [{"round": 0, "ms": best_ms, "speedup": best_spd, "note": "init"}]
        attempts = deque(maxlen=self.attempt_window)   # 最近被拒尝试窗口(官方 attempt_history)
        tokens = {"prompt": 0, "completion": 0}
        t_start = time.time()
        stall, improved_any, rejected_note = 0, False, None

        for r in range(1, opt_rounds + 1):
            if stall >= stall_limit:
                self._log(f"[opt] 连续 {stall} 轮无改进，收敛停止")
                break
            self._log(f"[round {r}/{opt_rounds}] 调用 LLM 优化 ...")
            user = self._opt_user(op, best, best_ms, best_spd, attempts, prof_text)
            # 调用 LLM；空代码(推理模型偶发把 reasoning 打满被截断)时同轮自动重试
            code, text = "", ""
            for _try in range(1 + self.empty_retries):
                text, usage = self.client.chat(
                    [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
                    temperature=self.temperature, max_tokens=self.max_tokens)
                tokens["prompt"] += usage.get("prompt_tokens", 0)
                tokens["completion"] += usage.get("completion_tokens", 0)
                code = prompts.extract_python_code(text)
                if code.strip():
                    break
                self._log(f"[round {r}] 空代码(可能截断)，自动重试 "
                          f"{_try + 1}/{self.empty_retries} ...")
            if not code.strip():
                rejected_note = ("代码仍为空：多次输出都没有完整代码块。请务必只输出一个 "
                                 "```python 代码块(含 import、kernel、def launch)，不要解释。")
                stall += 1
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "rejected(empty code)"})
                attempts.append({"round": r, "status": "empty", "ms": None,
                                 "code": code, "note": rejected_note,
                                 "avoid": self._avoid_for("empty", rejected_note)})
                self._log(f"[round {r}] ✗ 空代码(多次重试后仍空)")
                continue

            ev = self._eval(op, code)
            if not ev["ok"]:
                rejected_note = ev["why"][:600]
                stall += 1
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": f"rejected: {rejected_note[:60]}"})
                attempts.append({"round": r, "status": "error", "ms": None,
                                 "code": code, "note": rejected_note,
                                 "avoid": self._avoid_for("error", rejected_note)})
                self._log(f"[round {r}] ✗ 未通过: {rejected_note[:80]}")
                continue
            ms, spd = ev["ms"], ev["spd"]
            if ms < best_ms * (1 - improve_min):
                best, best_ms, best_spd = code, ms, spd
                stall, improved_any = 0, True
                rejected_note = None
                prof_text = self._profile(op_name, best) if use_ncu else None
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "accepted"})
                self._log(f"[round {r}] ✔ 更快: {best_ms:.4f}ms "
                          f"(speedup {best_spd:.3f})")
            else:
                stall += 1
                rejected_note = (f"未更快: 你的 {ms:.4f}ms ≥ best {best_ms:.4f}ms")
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "rejected(not faster)"})
                attempts.append({"round": r, "status": "not_faster", "ms": ms,
                                 "code": code, "note": rejected_note,
                                 "avoid": self._avoid_for("not_faster", rejected_note)})
                self._log(f"[round {r}] — 未更快 ({ms:.4f}ms vs best {best_ms:.4f}ms)")

        summary = {
            "op": op_name,
            "ok": True,
            "improved": improved_any,
            "rounds_used": len(curve) - 1,
            "best_ms": best_ms,
            "best_speedup_vs_eager": best_spd,
            "baseline_ms": curve[0]["ms"],
            "improve_pct": round((curve[0]["ms"] / best_ms - 1) * 100, 2)
                           if improved_any else 0.0,
            "curve": curve,
            "total_tokens": tokens,
            "wall_s": round(time.time() - t_start, 2),
            "ncu_used": use_ncu,
        }
        if save:
            self._save(op_name, summary, best)
        return summary

    def _save(self, op_name: str, summary: dict, best_code: str) -> str:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.join(RESULTS_DIR, f"opt_{op_name}_{ts}")
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "best_code": best_code},
                      f, ensure_ascii=False, indent=2)
        with open(stem + ".py", "w", encoding="utf-8") as f:
            f.write(best_code)
        if self.verbose:
            print(f"[opt] 结果已存: {stem}.json / {stem}.py")
        return stem + ".json"
