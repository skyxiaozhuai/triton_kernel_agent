"""KernelBench 判卷执行器 —— 在独立子进程里判生成的 kernel 是否符合官方题目。

与 executor.py（我们的 op 判卷）同构，但 golden 源换成"题目的 eager forward"：
    加载 problem .py -> Model(*get_init_inputs()) -> 对每个 case:
        inputs = problem.get_inputs()
        gold   = model(*inputs)          # 可信 golden（题目自身实现）
        out    = launch(*inputs)         # LLM 生成的 kernel 入口（位置传参）
    -> shape/dtype/数值(allclose rtol/atol 1e-2) 全过才算 PASS

仍然：独立子进程 + 超时；结果一行 TRITON_AGENT_RESULT: JSON 哨兵；LLM 只写 kernel+launch。
真跑需要 GPU/显存（多数 L1 默认 shape 很大），本机只做装载/预检，正式跑在服务器。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid

from .executor import (PROJECT_ROOT, RESULT_PREFIX, SCRATCH_DIR, ExecReport,
                       _parse_result_line)

_HARNESS_KB = """\
import json, sys, traceback, importlib.util, os
sys.path.insert(0, __ROOT__)
import torch

__CODE__

def _load_problem(path):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("_kb_prob", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _main():
    result = {"ok": False, "message": "", "max_abs_err": None}
    try:
        M = _load_problem(__PATH__)
        model = M.Model(*M.get_init_inputs())
        all_ok = True
        max_err = 0.0
        for ci in range(__NCASES__):
            ins = M.get_inputs()
            gold = model(*ins)
            out = launch(*ins)
            if out is None:
                all_ok = False
                result["message"] = f"launch() 返回 None (case {ci})"
                break
            if out.shape != gold.shape:
                all_ok = False
                result["message"] = (f"输出 shape 不符 (case {ci}): "
                                     f"out={tuple(out.shape)} vs golden={tuple(gold.shape)}")
                break
            if out.dtype != gold.dtype:
                all_ok = False
                result["message"] = (f"输出 dtype 不符 (case {ci}): "
                                     f"{out.dtype} vs golden {gold.dtype}")
                break
            err = float((out.float() - gold.float()).abs().max().item())
            max_err = max(max_err, err)
            if not torch.allclose(out.float(), gold.float(), rtol=1e-2, atol=1e-2):
                all_ok = False
                result["max_abs_err"] = err
                result["message"] = f"数值不对齐 (case {ci}): max_abs_err={err:.3e}"
                break
        if all_ok:
            result["ok"] = True
            result["message"] = "PASS"
            result["max_abs_err"] = max_err
    except Exception:
        result["message"] = "异常: " + traceback.format_exc(limit=20)
    print("__PREFIX__" + json.dumps(result))
    sys.stdout.flush()

if __name__ == "__main__":
    _main()
"""


def _build_harness(problem_path: str, code: str, num_cases: int) -> str:
    src = _HARNESS_KB
    src = src.replace("__ROOT__", json.dumps(PROJECT_ROOT))
    src = src.replace("__CODE__", code)
    src = src.replace("__PATH__", json.dumps(os.path.abspath(problem_path)))
    src = src.replace("__NCASES__", str(int(num_cases)))
    src = src.replace("__PREFIX__", RESULT_PREFIX)
    return src


def run_problem(problem_path: str, code: str, num_cases: int = 2,
                timeout: float = 180.0, keep_script: bool = True) -> ExecReport:
    """在子进程沙箱里判 KernelBench 题目生成代码，返回 ExecReport。"""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    script = _build_harness(problem_path, code, num_cases)
    script_path = os.path.join(SCRATCH_DIR, f"kb_{uuid.uuid4().hex[:8]}.py")
    with open(script_path, "w") as f:
        f.write(script)

    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=timeout)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        returncode, timed_out = -1, True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\n[TIMEOUT] 超过沙箱时间限制被终止"
    wall_s = time.time() - t0
    script_ref = script_path if keep_script else None

    if timed_out:
        return ExecReport(status="timeout", ok=False, message="执行超时，已终止",
                          max_abs_err=None, stdout=stdout, stderr=stderr,
                          wall_s=wall_s, script_path=script_ref)

    result = _parse_result_line(stdout)
    if result is None:
        if "SyntaxError" in stderr:
            status, msg = "syntax", "生成代码存在语法错误: " + stderr.strip().splitlines()[-1]
        else:
            status, msg = "error", f"harness 未输出结果 (returncode={returncode})"
        return ExecReport(status=status, ok=False, message=msg, max_abs_err=None,
                          stdout=stdout, stderr=stderr, wall_s=wall_s,
                          script_path=script_ref)
    # PASS 双信号：哨兵 ok 且退出码 0
    if result["ok"] and returncode != 0:
        return ExecReport(status="error", ok=False,
                          message=f"结果哨兵为 PASS 但退出码非 0 (returncode={returncode})",
                          max_abs_err=result.get("max_abs_err"),
                          stdout=stdout, stderr=stderr, wall_s=wall_s,
                          script_path=script_ref)
    if result["ok"]:
        status, ok = "pass", True
    elif result.get("max_abs_err") is not None:
        status, ok = "correctness", False
    else:
        status, ok = "error", False
    return ExecReport(status=status, ok=ok, message=result.get("message", ""),
                      max_abs_err=result.get("max_abs_err"),
                      stdout=stdout, stderr=stderr, wall_s=wall_s,
                      script_path=script_ref)
