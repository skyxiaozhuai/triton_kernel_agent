"""沙箱执行器 —— 在独立子进程里运行"生成代码"，返回结构化报告。

为什么必须沙箱（面试可讲）：
  生成代码不可信：可能死循环 / 触发 CUDA error / segfault / OOM。
  独立子进程 + 超时 kill ⇒ orchestrator 主进程永不因坏代码崩溃。

执行模型（harness 模式）：
  判定逻辑永远是我们可信的代码，LLM 只负责 kernel 与 launch。
  约定生成代码必须包含：
      @triton.jit
      def <kernel>(...) -> ...            # 一个或多个 kernel
      def launch(<输入名...>, meta) -> torch.Tensor  # 负责 grid/调用 kernel/返回输出

  executor 把它拼进标准 harness：
      op.generate_inputs() 生成输入 + op.golden() 算期望
      -> 调 launch(**inputs, meta=meta) -> 与 golden 数值对齐
  结果以一行带前缀的 JSON 打到 stdout 供 executor 解析；详细 traceback 打到 stderr
  （供 error_parser 做错误分类）。
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")

RESULT_PREFIX = "TRITON_AGENT_RESULT:"

_HARNESS_TEMPLATE = """\
import json, sys, traceback
sys.path.insert(0, __ROOT__)
import torch
from benchmarks import ops_registry
op = ops_registry.get_op(__OPNAME__)

__CODE__

def _main():
    result = {"ok": False, "message": "", "max_abs_err": None, "perf": None}
    try:
        cases = op.generate_cases()   # 多 case 全过才算 PASS
        all_ok = True
        max_err = 0.0
        msg = "PASS"
        for ci, case in enumerate(cases):
            args = {k: v for k, v in case.items() if k != "meta"}
            meta = case["meta"]
            call_args = dict(args)
            call_args.update(meta)     # meta 标量(N/M/K...)一并作 launch 具名参数
            out = launch(**call_args)
            gold = op.golden(**args)
            if out is None:
                all_ok = False
                msg = f"launch() 返回 None (case {ci})"
                break
            if out.shape != gold.shape:
                all_ok = False
                msg = (f"输出 shape 不符 (case {ci}): out={tuple(out.shape)} "
                       f"vs golden={tuple(gold.shape)}")
                break
            err = float((out - gold).abs().max().item())
            max_err = max(max_err, err)
            if not op.check(out, gold):
                all_ok = False
                result["max_abs_err"] = err
                msg = f"数值不对齐 (case {ci}): max_abs_err={err:.3e}"
                break
        if all_ok:
            result["max_abs_err"] = max_err
            result["ok"] = True
            result["message"] = msg     # PASS
            if __PERF__:      # 性能 critic 数据（用主 case 测；失败不阻塞结论）
                try:
                    case0 = cases[0]
                    args0 = {k: v for k, v in case0.items() if k != "meta"}
                    meta0 = case0["meta"]
                    call0 = dict(args0)
                    call0.update(meta0)
                    from triton.testing import do_bench
                    launch_ms = do_bench(lambda: launch(**call0), warmup=20, rep=80)
                    eager_ms = do_bench(lambda: op.golden(**args0), warmup=20, rep=80)
                    result["perf"] = {"launch_ms": round(float(launch_ms), 4),
                                      "eager_ms": round(float(eager_ms), 4),
                                      "speedup_vs_eager": round(float(eager_ms) / float(launch_ms), 3)}
                except Exception:
                    result["perf"] = None
        else:
            result["message"] = msg
    except Exception:
        result["message"] = "异常: " + traceback.format_exc(limit=20)
    print("__PREFIX__" + json.dumps(result))
    sys.stdout.flush()

if __name__ == "__main__":
    _main()
"""


@dataclasses.dataclass
class ExecReport:
    """一次沙箱执行的完整结果。"""
    status: str            # pass | correctness | error | syntax | timeout
    ok: bool               # pass 时为 True
    message: str           # 人类可读摘要（喂给 LLM 的雏形）
    max_abs_err: float | None
    stdout: str
    stderr: str
    wall_s: float
    script_path: str | None = None
    perf: dict | None = None     # 可选性能测量 {launch_ms, eager_ms, speedup_vs_eager}


def _build_harness(op_name: str, code: str, perf: bool = False) -> str:
    src = _HARNESS_TEMPLATE
    src = src.replace("__ROOT__", json.dumps(PROJECT_ROOT))
    src = src.replace("__OPNAME__", json.dumps(op_name))
    src = src.replace("__CODE__", code)
    src = src.replace("__PREFIX__", RESULT_PREFIX)
    src = src.replace("__PERF__", "True" if perf else "False")
    return src


def _parse_result_line(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def run(op_name: str, code: str, timeout: float = 90.0,
        keep_script: bool = True, perf: bool = False) -> ExecReport:
    """在子进程沙箱里执行 op_name 的生成代码，返回 ExecReport。

    perf=True 时，harness 在数值通过后额外用 do_bench 测量 launch 与 eager 耗时。
    """
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    script = _build_harness(op_name, code, perf=perf)
    script_path = os.path.join(SCRATCH_DIR, f"run_{op_name}_{uuid.uuid4().hex[:8]}.py")
    with open(script_path, "w") as f:
        f.write(script)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        returncode = -1
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\n[TIMEOUT] 超过沙箱时间限制被终止"
        timed_out = True
    wall_s = time.time() - t0

    if timed_out:
        return ExecReport(status="timeout", ok=False, message="执行超时，已终止",
                          max_abs_err=None, stdout=stdout, stderr=stderr,
                          wall_s=wall_s, script_path=script_path if keep_script else None)

    result = _parse_result_line(stdout)
    if result is None:
        # 没有结果 JSON：通常是模块级语法错误或启动失败
        if "SyntaxError" in stderr:
            status = "syntax"
            msg = "生成代码存在语法错误: " + stderr.strip().splitlines()[-1]
        else:
            status = "error"
            msg = f"harness 未输出结果 (returncode={returncode})"
        return ExecReport(status=status, ok=False, message=msg, max_abs_err=None,
                          stdout=stdout, stderr=stderr, wall_s=wall_s,
                          script_path=script_path if keep_script else None)

    if result["ok"]:
        status, ok = "pass", True
    elif result["max_abs_err"] is not None:
        status, ok = "correctness", False
    else:
        status, ok = "error", False
    return ExecReport(status=status, ok=ok, message=result.get("message", ""),
                      max_abs_err=result.get("max_abs_err"),
                      stdout=stdout, stderr=stderr, wall_s=wall_s,
                      script_path=script_path if keep_script else None,
                      perf=result.get("perf"))


if __name__ == "__main__":
    # 自测：正确代码应 pass
    good = '''
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
    rep = run("vector_add", good)
    print(f"good code  -> status={rep.status}, ok={rep.ok}, msg={rep.message[:80]}")
    rep2 = run("vector_add", good, perf=True)
    print(f"good code(perf) -> ok={rep2.ok}, perf={rep2.perf}")
