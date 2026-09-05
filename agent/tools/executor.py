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
    result = {"ok": False, "message": "", "max_abs_err": None}
    try:
        args = op.generate_inputs()
        meta = args.pop("meta")
        gold = op.golden(**args)
        call_args = dict(args)
        call_args.update(meta)     # meta 里的标量(N/M/K...)一并作 launch 具名参数
        out = launch(**call_args)
        if out is None:
            result["message"] = "launch() 返回了 None，未产生输出"
        elif out.shape != gold.shape:
            result["message"] = "输出 shape 不符: out=" + str(tuple(out.shape)) + " vs golden=" + str(tuple(gold.shape))
        else:
            err = float((out - gold).abs().max().item())
            result["max_abs_err"] = err
            if op.check(out, gold):
                result["ok"] = True
                result["message"] = "PASS"
            else:
                result["message"] = "数值不对齐: max_abs_err=" + format(err, ".3e")
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


def _build_harness(op_name: str, code: str) -> str:
    src = _HARNESS_TEMPLATE
    src = src.replace("__ROOT__", json.dumps(PROJECT_ROOT))
    src = src.replace("__OPNAME__", json.dumps(op_name))
    src = src.replace("__CODE__", code)
    src = src.replace("__PREFIX__", RESULT_PREFIX)
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
        keep_script: bool = True) -> ExecReport:
    """在子进程沙箱里执行 op_name 的生成代码，返回 ExecReport。"""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    script = _build_harness(op_name, code)
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
                      script_path=script_path if keep_script else None)


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
