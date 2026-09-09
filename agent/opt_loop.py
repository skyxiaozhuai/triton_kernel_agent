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
                 empty_retries: int = 2, beam_width: int = 1,
                 prescribe: bool = False, prescribe_max_tokens: int = 600):
        self.client = client if client is not None else LLMClient()
        self.attempt_window = attempt_window   # 最近被拒尝试注入窗口(对齐官方 attempt_history)
        self.empty_retries = empty_retries     # 空代码(截断)自动重试次数
        self.beam_width = max(1, beam_width)   # 优化端每轮候选数(对齐官方 beam)
        self.prescribe = prescribe             # 诊断先行(对齐官方 BottleneckAnalyzer)
        self.prescribe_max_tokens = prescribe_max_tokens
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
                  attempts, prof_text: str | None,
                  direction: str | None = None) -> str:
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
        parts.append("\n请让上面的 kernel 更快且保持数值正确。")
        if direction:
            parts.append(f"[本候选方向] 请优先沿这个方向改：{direction}")
        parts += ["直接输出一个 ```python 代码块（含 import、@triton.jit kernel、def launch）。",
                  "不要输出代码块以外的解释。"]
        return "\n".join(parts)

    # ---------- 诊断先行(prescribe) + 候选方向(beam-lite) ----------
    @staticmethod
    def _fallback_directions(prof_text: str | None) -> list[str]:
        """本地确定性方向：无剖析 / LLM 诊断失败时的兜底（供 beam 多候选用）。"""
        generic = ["增大/重排 BLOCK tile 并相应调 num_warps（大 tile 配大 warps）",
                   "调 num_warps / num_stages 与流水线深度（避免与 tile 冲突的噪声微调）",
                   "优化访存：向量化、连续读写、减少冗余 load，提升局部性"]
        if prof_text:
            low = prof_text
            if "memory-bound" in low or ("DRAM" in low and "%" in low and "高" in low):
                return [generic[2], generic[0]]
            if "compute-bound" in low:
                return [generic[0], generic[1]]
            if "占用" in low and ("低" in low or "under" in low):
                return [generic[0]]
        return generic[:3]

    @staticmethod
    def _parse_directions(text: str) -> list[str]:
        """从 LLM 诊断输出解析候选方向（去序号/前缀/空行，限 3 条）。"""
        out: list[str] = []
        for line in (text or "").splitlines():
            line = line.strip().lstrip("-*•0123456789.、)①②③ ")
            if not line or "```" in line:
                continue
            if line[:2] in ("瓶颈", "诊断", "建议", "方向", "1)", "2)", "3)") or "方向" in line[:6]:
                continue
            line = line.rstrip("。；;，,")
            if 6 <= len(line) <= 160:
                out.append(line)
            if len(out) >= 3:
                break
        return out or KernelOptimizer._fallback_directions(None)

    def _prescribe(self, op, prof_text: str | None, best_ms: float) -> list[str]:
        """诊断先行(对齐官方 BottleneckAnalyzer)：先让 LLM 归纳瓶颈 + 互斥方向。"""
        spec = self._short_spec(op)
        head = f"{spec}\n\n当前 best: {best_ms:.4f}ms\n\n"
        prof = ("硬件剖析(roofline):\n" + prof_text + "\n\n") if prof_text else ""
        ask = ("请先诊断，不要写 kernel 代码：\n"
               "1) 用一句话说瓶颈（结合剖析指标 DRAM/SM/占用率/命中率）；\n"
               "2) 给最多 3 个【互斥】优化方向，每行一个，只描述改法"
               "（例如换更大 BLOCK tile 并配 num_warps / 改 num_stages 流水 / 优化访存与向量化）。")
        text, _usage = self.client.chat(
            [{"role": "system", "content": prompts.SYSTEM_PROMPT},
             {"role": "user", "content": head + prof + ask}],
            temperature=0.1, max_tokens=self.prescribe_max_tokens)
        dirs = self._parse_directions(text)
        self._log("  [prescribe] 方向 ×%d: %s" % (len(dirs),
                  " | ".join(d[:36] for d in dirs)))
        return dirs

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

    def optimize_beam(self, op_name: str, init_code: str | None = None,
                      opt_rounds: int = 6, stall_limit: int = 2,
                      improve_min: float = 0.02, use_ncu: bool = True,
                      save: bool = True, beam_width: int | None = None,
                      prescribe: bool | None = None) -> dict:
        """优化端"诊断先行 + 多候选 beam"（对齐官方 BottleneckAnalyzer + beam）。

        每轮：prescribe(LLM 给互斥方向)或本地方向 → 对每个方向各生成一个候选
              (beam_width 个) → 逐候选静态+executor 验证 → 取本轮最快正确者。
        比 optimize() 的单条贪心轨迹更能跳出局部最优（4096³ 曾两轮盲改无效）。
        beam_width=1 且 prescribe=False 时 ≈ optimize() 行为。
        """
        beam_width = beam_width if beam_width is not None else self.beam_width
        prescribe = prescribe if prescribe is not None else self.prescribe
        op = ops_registry.get_op(op_name)
        if init_code is None:
            rec = memory.get(op_name)
            if not rec:
                return {"ok": False,
                        "error": f"经验库无 {op_name} 的成功代码；先 run_agent 或给 --code-file"}
            init_code = rec["code"]

        ev0 = self._eval(op, init_code)
        if not ev0["ok"]:
            return {"ok": False, "error": f"初始代码不正确: {ev0['why']}"}
        best, best_ms, best_spd = init_code, ev0["ms"], ev0["spd"]
        self._log(f"[opt] {op_name} 基线: launch_ms={best_ms:.4f}ms "
                  f"(speedup_vs_eager={best_spd:.3f}) beam_width={beam_width}"
                  + ("  prescribe=on" if prescribe else ""))

        prof_text = self._profile(op_name, best) if use_ncu else None
        curve = [{"round": 0, "ms": best_ms, "speedup": best_spd, "note": "init"}]
        attempts = deque(maxlen=self.attempt_window)
        tokens = {"prompt": 0, "completion": 0}
        t_start = time.time()
        stall, improved_any, rejected_note = 0, False, None

        def _chat(user: str, mt: int) -> str:
            text, usage = self.client.chat(
                [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                temperature=self.temperature, max_tokens=mt)
            tokens["prompt"] += usage.get("prompt_tokens", 0)
            tokens["completion"] += usage.get("completion_tokens", 0)
            return text

        def _gen(user: str) -> tuple[str, str]:
            """生成（带同轮空代码重试），返回 (code, text)。"""
            code, text = "", ""
            for _try in range(1 + self.empty_retries):
                text = _chat(user, self.max_tokens)
                code = prompts.extract_python_code(text)
                if code.strip():
                    break
                self._log(f"    空代码(截断)，自动重试 {_try + 1}/{self.empty_retries}")
            return code.strip(), text

        for r in range(1, opt_rounds + 1):
            if stall >= stall_limit:
                self._log(f"[opt] 连续 {stall} 轮无改进，收敛停止")
                break
            # 1) 方向集：prescribe(LLM) 或本地 fallback；单候选退化为 [None]
            if prescribe:
                dirs = (self._prescribe(op, prof_text, best_ms)
                        or self._fallback_directions(prof_text))
            else:
                dirs = ([None] if beam_width <= 1
                        else self._fallback_directions(prof_text))
            if beam_width > 1 and dirs and dirs != [None]:
                dirs = dirs[:beam_width]

            # 2) 逐方向生成候选
            cands: list[tuple[str, str | None]] = []   # (code, direction)
            for i, direction in enumerate(dirs):
                self._log(f"[round {r}/{opt_rounds}] 候选 {i + 1}/{len(dirs)}"
                          + (f"  方向: {direction[:48]}" if direction else "") + " ...")
                user = self._opt_user(op, best, best_ms, best_spd,
                                      attempts, prof_text, direction=direction)
                code, _text = _gen(user)
                if code:
                    cands.append((code, direction))
                else:
                    self._log("    ✗ 空代码(多次重试仍空)")
            if not cands:
                rejected_note = ("全部候选为空代码：多次输出都没有代码块。请务必只输出一个 "
                                 "```python 代码块(含 import、kernel、def launch)，不要解释。")
                stall += 1
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "rejected(empty)"})
                attempts.append({"round": r, "status": "empty", "ms": None,
                                 "code": "", "note": rejected_note,
                                 "avoid": self._avoid_for("empty", rejected_note)})
                self._log(f"[round {r}] ✗ 本批 {len(dirs)} 候选全为空")
                continue

            # 3) 验证本批，取最快正确候选
            best_cand = None    # (code, ev)
            first_fail = None
            n_ok = 0
            for code, _direction in cands:
                ev = self._eval(op, code)
                if not ev["ok"]:
                    first_fail = first_fail or {"code": code, "why": ev["why"]}
                    self._log(f"    ✗ 候选失败: {ev['why'][:60]}")
                    continue
                n_ok += 1
                if best_cand is None or ev["ms"] < best_cand[1]["ms"]:
                    best_cand = (code, ev)
            if best_cand is None:
                note = first_fail["why"][:600] if first_fail else "全部候选失败"
                stall += 1
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "rejected: " + note[:50]})
                attempts.append({"round": r, "status": "error", "ms": None,
                                 "code": first_fail["code"][:600] if first_fail else "",
                                 "note": note,
                                 "avoid": self._avoid_for("error", note)})
                self._log(f"[round {r}] ✗ 本批 {len(cands)} 候选均未通过")
                continue

            code, ev = best_cand
            ms, spd = ev["ms"], ev["spd"]
            if ms < best_ms * (1 - improve_min):
                best, best_ms, best_spd = code, ms, spd
                stall, improved_any, rejected_note = 0, True, None
                prof_text = self._profile(op_name, best) if use_ncu else None
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": f"accepted({n_ok}/{len(cands)} 候选过)"})
                self._log(f"[round {r}] ✔ 更快: {best_ms:.4f}ms "
                          f"(speedup {best_spd:.3f})")
            else:
                stall += 1
                rejected_note = f"未更快: 本批最快 {ms:.4f}ms ≥ best {best_ms:.4f}ms"
                curve.append({"round": r, "ms": best_ms, "speedup": best_spd,
                              "note": "rejected(not faster)"})
                attempts.append({"round": r, "status": "not_faster", "ms": ms,
                                 "code": code, "note": rejected_note,
                                 "avoid": self._avoid_for("not_faster", rejected_note)})
                self._log(f"[round {r}] — 本批 {len(cands)} 候选最快 {ms:.4f}ms，未更快")

        summary = {
            "op": op_name, "ok": True, "improved": improved_any,
            "rounds_used": len(curve) - 1,
            "best_ms": best_ms, "best_speedup_vs_eager": best_spd,
            "baseline_ms": curve[0]["ms"],
            "improve_pct": round((curve[0]["ms"] / best_ms - 1) * 100, 2)
                           if improved_any else 0.0,
            "curve": curve, "total_tokens": tokens,
            "wall_s": round(time.time() - t_start, 2),
            "ncu_used": use_ncu, "beam": beam_width, "prescribe": prescribe,
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
