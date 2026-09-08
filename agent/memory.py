"""RAG 经验库 v1（零依赖）—— 跨任务记忆 + 检索增强。

设计要点（见 PLAN §10）：
- 每次 agent 成功时，把最终通过 kernel 存为 results/memory/<op>.json；
- 检索按 OP_META.category 精确匹配（elementwise / softmax / reduction / gemm）；
- 防作弊：retrieve() 只返回「同 category 但 ≠ 当前 op」的成功样例；
- 注入：由 loop / prompts 在生成前调用 retrieve()，作为参考注入上下文。

升级路径：v1 category 匹配 → v2 embedding 相似 → v3 LangChain 检索器 / 纳入 Triton 文档。
（当前为纯文件实现，无第三方依赖，可随时单测。）
"""
from __future__ import annotations

import datetime
import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(PROJECT_ROOT, "results", "memory")


def _ensure_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _path(op_name: str) -> str:
    return os.path.join(MEMORY_DIR, f"{op_name}.json")


def _category_of(op_name: str, category: str | None = None) -> str:
    if category:
        return category
    try:
        from benchmarks import ops_registry
        return str(ops_registry.get_op(op_name).OP_META.get("category", "unknown"))
    except Exception:  # noqa: BLE001 —— op 未注册/导入失败时给 unknown
        return "unknown"


def add_success(op_name: str, code: str, category: str | None = None,
                rounds: int | None = None, perf: dict | None = None) -> str:
    """记录一次成功 kernel（同 op 覆盖为最新）。返回记录文件路径。"""
    _ensure_dir()
    rec = {
        "op": op_name,
        "category": _category_of(op_name, category),
        "added_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rounds": rounds,
        "perf": perf,
        "code": code,
    }
    path = _path(op_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return path


def get(op_name: str) -> dict | None:
    """取单个 op 的成功记录；无则 None。"""
    p = _path(op_name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_all() -> list[dict]:
    """全部成功记录。"""
    _ensure_dir()
    out = []
    for fn in sorted(os.listdir(MEMORY_DIR)):
        if fn.endswith(".json"):
            rec = get(fn[:-5])
            if rec:
                out.append(rec)
    return out


def retrieve_by_category(category: str, exclude_op: str | None = None,
                         k: int = 2) -> list[dict]:
    """返回指定 category 的成功样例，排除 exclude_op 自身（防作弊），最多 k 条。"""
    got = [r for r in list_all()
           if r.get("category") == category and r.get("op") != exclude_op]
    return got[:k]


def retrieve(op_name: str, k: int = 2) -> list[dict]:
    """对给定 op 检索参考样例：同 category 且非自身（RAG 注入用）。"""
    return retrieve_by_category(_category_of(op_name), exclude_op=op_name, k=k)


# ============ 失败样本回灌 v1（受控自改进 · PLAN §10 方向②） ============
# 记录一次成功 run 的"最后一个失败步 → 成功代码"修复对，存 results/memory/failures/；
# 后续 run 在同类别错误时，检索【其它算子】的历史修复示范注入反馈 —— 跨任务借鉴。
# 注意与"正样本 RAG"(成功 kernel) 分开存储；排除同 op 防作弊(同 op 成功代码=答案)。

FAILURES_DIR = os.path.join(MEMORY_DIR, "failures")

_STATUS_NORM = {
    "invalid_code": "structure",
    "static_cheat": "cheat",
    "static_structure": "structure",
    "static_syntax": "syntax",
}


def normalize_status(status: str | None) -> str:
    """把 loop 的 step.status 归一化成可检索的错误类别键。"""
    if not status:
        return "unknown"
    if status.startswith("static_"):
        return _STATUS_NORM.get(status, status[len("static_"):])
    return _STATUS_NORM.get(status, status)


def _cat_from_feedback(feedback: str | None) -> str | None:
    """从 error_parser.to_feedback_text 的 '❌ 类别: xxx' 提取细分类别。"""
    if not feedback:
        return None
    m = re.search(r"❌\s*类别:\s*(\w+)", feedback)
    return m.group(1) if m else None


def _fail_path(op_name: str) -> str:
    return os.path.join(FAILURES_DIR, f"{op_name}.json")


def _load_failures(op_name: str) -> dict:
    p = _fail_path(op_name)
    if not os.path.exists(p):
        return {"op": op_name, "pairs": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def record_fix_pair(op_name: str, steps, max_pairs: int = 6) -> str | None:
    """记录一次成功 run 的"失败→成功"修复对（steps: list[AgentStep]）。

    取最后一个失败步(status != pass) 与最终成功步配对入库。
    返回记录文件路径；没有失败步可记时返回 None。
    """
    if not steps:
        return None
    ok_step = steps[-1]
    bad = None
    for st in reversed(steps[:-1]):
        if getattr(st, "status", "pass") != "pass":
            bad = st
            break
    if bad is None:
        return None
    os.makedirs(FAILURES_DIR, exist_ok=True)
    rec = _load_failures(op_name)
    rec.setdefault("pairs", [])
    rec["pairs"].append({
        # 细分类优先(如 compile/cuda 从 feedback 提取)，否则回退 step.status
        "err_category": normalize_status(
            _cat_from_feedback(getattr(bad, "feedback", ""))
            or getattr(bad, "status", "unknown")),
        "bad_code": getattr(bad, "code", ""),
        "feedback": (getattr(bad, "feedback", "") or "")[:800],
        "good_code": getattr(ok_step, "code", ""),
        "rounds": len(steps),
        "added_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    rec["pairs"] = rec["pairs"][-max_pairs:]      # 每 op 只留最近 max_pairs 条
    path = _fail_path(op_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return path


def _iter_all_pairs():
    """遍历所有失败记录里的修复对，带所属 op。"""
    if not os.path.isdir(FAILURES_DIR):
        return
    for fn in sorted(os.listdir(FAILURES_DIR)):
        if not fn.endswith(".json"):
            continue
        op = fn[:-5]
        rec = _load_failures(op)
        for pair in rec.get("pairs", []):
            pair = dict(pair)
            pair["op"] = rec.get("op", op)
            yield pair


def retrieve_fix(err_category: str | None, exclude_op: str | None = None,
                 k: int = 1):
    """检索【其它算子】(排除自身防作弊) 同类错误的历史修复示范。

    返回最近 1 条 dict，或 k>1 时返回 list；无匹配返回 None。
    """
    key = normalize_status(err_category)
    got = [p for p in _iter_all_pairs()
           if normalize_status(p.get("err_category")) == key
           and p.get("op") != exclude_op]
    if not got:
        return None
    return got[-1] if k <= 1 else got[-k:]


def format_fix_ref(fix: dict) -> str:
    """把一条历史修复示范格式化成注入反馈的文本（few-shot 修法示范）。"""
    return (
        f"<历史同类错误修复示范> 曾有一个算子 {fix.get('op')} 遇到同类错误"
        f"({fix.get('err_category')})，最终在第 {fix.get('rounds')} 轮内修好。"
        f"它修复后通过的正确代码如下（参考它的修法，勿照抄结构）：\n"
        f"{fix.get('good_code', '')}\n</历史同类错误修复示范>"
    )


if __name__ == "__main__":
    # 简单自测走一遍
    print("MEMORY_DIR:", MEMORY_DIR)
    print("当前记录数:", len(list_all()))
    for r in list_all():
        print(f"  - {r['op']} [{r['category']}] {r['added_at']} (code {len(r['code'])} chars)")
