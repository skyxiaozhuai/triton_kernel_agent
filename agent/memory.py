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


if __name__ == "__main__":
    # 简单自测走一遍
    print("MEMORY_DIR:", MEMORY_DIR)
    print("当前记录数:", len(list_all()))
    for r in list_all():
        print(f"  - {r['op']} [{r['category']}] {r['added_at']} (code {len(r['code'])} chars)")
