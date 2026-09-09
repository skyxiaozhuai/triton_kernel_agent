#!/usr/bin/env python3
"""report_traj_html 回归自测：构造最小轨迹 → 渲染 HTML → 断言内容与转义。

agent 组：纯 stdlib，不跑 GPU / 不 import torch。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# scripts/ 非包目录，用 importlib 按文件路径加载 report_traj_html
_spec = importlib.util.spec_from_file_location(
    "report_traj_html", os.path.join(ROOT, "scripts", "report_traj_html.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_traj, render = _mod.load_traj, _mod.render  # noqa: E402


def _write_traj(path: str) -> None:
    steps = [
        {"round": 1, "status": "correctness", "wall_s": 12.3,
         "prompt_tokens": 400, "completion_tokens": 200,
         "max_abs_err": 0.042,
         "feedback": "数值偏差：<b>这里该被转义</b>",
         "code": "import torch\n@triton.jit\ndef k():\n    pass"},
        {"round": 2, "status": "pass", "wall_s": 8.1,
         "prompt_tokens": 300, "completion_tokens": 150,
         "max_abs_err": 0.0,
         "feedback": "",
         "code": "import torch\ndef launch(): ..."},
    ]
    summary = {"op": "vector_add", "success": True, "rounds_used": 2,
               "final_status": "pass", "final_max_abs_err": 0.0,
               "total_tokens": {"prompt": 700, "completion": 350},
               "wall_s": 20.4, "memory_used": "vector_add"}
    with open(path, "w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        traj = os.path.join(tmp, "traj_t.jsonl")
        _write_traj(traj)
        summary, steps = load_traj(traj)
        assert summary.get("success") is True and len(steps) == 2, "解析失败"
        doc = render(summary, steps, traj)

        # 结构
        assert "<html" in doc and "</html>" in doc, "缺 html 骨架"
        assert "vector_add" in doc, "缺 op 名"
        assert "✔ 通过" in doc, "缺成功徽标"
        assert "kernel 代码" in doc, "缺代码折叠"

        # HTML 转义（feedback 里的 <b> 不应成为标签；JS 风格尖括号被转义）
        assert "&lt;b&gt;" in doc, "feedback 未转义"
        assert "<b>这里该被转义</b>" not in doc, "feedback 转义失败"

        # 可写文件
        out = os.path.join(tmp, "report.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(doc)
        assert os.path.getsize(out) > 1000, "html 太小"
    print("report html 自测通过 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
