"""错误解析器：把沙箱执行报告(raw stderr/status)转成结构化、可喂给 LLM 的反馈。

设计动机（面试可讲）：
  直接丢几百行 traceback 给 LLM 收敛慢且烧 token。
  先把失败归类 + 提炼关键行 + 附带该类别最常见原因，
  LLM 只看"几行精炼反馈"，迭代效率显著更高。

分类：correctness | compile | cuda | syntax | timeout | runtime | unknown
"""
from __future__ import annotations

import re

from .executor import ExecReport

CATEGORY_ADVICE = {
    "correctness": "kernel 能编译运行但数值不对。检查: 索引/mask、reduce 轴、是否漏归一化/累加清零、dtype。",
    "compile": "Triton 编译失败。检查: tl 名字拼写、constexpr、tl.arange 与 tile shape、指针算术、当前 arch 不支持的指令。",
    "cuda": "CUDA 运行时错误。检查: 指针越界、grid/block 配置、非法内存访问、显存超限。",
    "syntax": "Python 语法错误，通常漏冒号/括号/缩进或缩进不一致。",
    "timeout": "运行超时。检查: grid 是否过大、循环边界、mask 是否导致 load 不到、是否死循环。",
    "runtime": "运行期异常，看 detail 定位。",
    "unknown": "无法自动归类，请结合 detail 与原始 stderr 判断。",
}

# 关键词 -> 类别
_KEYWORDS = [
    ("compile", ["CompilationError", "triton.compiler", "invalid syntax in kernel",
                 "ttir", "mlir", "cannot find", "undeclared identifier"]),
    ("cuda", ["CUDA error", "illegal memory access", "device-side assert",
              "out of memory", "CUBLAS", "an illegal memory access"]),
]


def _keyword_category(stderr: str) -> str | None:
    low = stderr.lower()
    for cat, keys in _KEYWORDS:
        for k in keys:
            if k.lower() in low:
                return cat
    return None


def _pick_key_line(stderr: str) -> str:
    """从 stderr 里挑一行最能代表错误的文本。"""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    # 优先带 Error/assert 的行
    err_lines = [ln for ln in lines if re.search(r"error|assert|exception", ln, re.I)]
    if err_lines:
        return err_lines[-1][:300]
    if lines:
        return lines[-1][:300]
    return ""


def classify(report: ExecReport) -> dict:
    """把 ExecReport 归类，返回 {category, summary, detail, advice}。"""
    if report.ok:
        return {"category": "pass", "summary": "已通过",
                "detail": report.message, "advice": ""}

    # 先按沙箱判定的粗状态
    if report.status == "timeout":
        return {"category": "timeout", "summary": "运行超时(被沙箱终止)",
                "detail": report.message, "advice": CATEGORY_ADVICE["timeout"]}
    if report.status == "syntax":
        return {"category": "syntax", "summary": "Python 语法错误",
                "detail": _pick_key_line(report.stderr), "advice": CATEGORY_ADVICE["syntax"]}
    if report.status == "correctness":
        return {"category": "correctness", "summary": "数值与 golden 不对齐",
                "detail": report.message, "advice": CATEGORY_ADVICE["correctness"]}

    # status == error：深入 stderr/message 细分类
    text = f"{report.stderr}\n{report.message}"
    cat = _keyword_category(text) or "runtime"
    summary = _pick_key_line(text) or report.message[:200]
    return {"category": cat, "summary": summary,
            "detail": report.message, "advice": CATEGORY_ADVICE.get(cat, "")}


def to_feedback_text(report: ExecReport) -> str:
    """把报告转成一段精炼的、直接进 LLM 上下文的反馈文本。"""
    info = classify(report)
    if info["category"] == "pass":
        return "✅ 运行与数值均通过。"
    lines = [
        f"❌ 类别: {info['category']}",
        f"摘要: {info['summary']}",
    ]
    if info["detail"] and info["detail"] != info["summary"]:
        # 截断 detail，避免上下文过长
        lines.append(f"详情: {info['detail'][:1200]}")
    if info["advice"]:
        lines.append(f"排查建议: {info['advice']}")
    return "\n".join(lines)
