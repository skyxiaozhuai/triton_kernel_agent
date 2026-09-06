#!/usr/bin/env python3
"""RAG 经验库 v1 自测：add / get / list / retrieve(排除自身、category 匹配)。

在临时目录中隔离测试，不污染真实 results/memory/。
用法: python scripts/test_memory.py
纯文件逻辑，无需 GPU / LLM / 网络。
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agent.memory as m  # noqa: E402


def main() -> int:
    # 隔离到临时目录（memory 函数运行时读取模块级 MEMORY_DIR）
    tmp = tempfile.mkdtemp(prefix="triton_mem_")
    m.MEMORY_DIR = tmp
    failed = 0

    def check(label: str, cond: bool) -> None:
        nonlocal failed
        tag = "✔" if cond else "✗"
        print(f"{tag} {label}")
        if not cond:
            failed += 1

    # --- add / get ---
    m.add_success("vec_x", "CODE_VEC", category="elementwise")
    m.add_success("relu_y", "CODE_RELU", category="elementwise")
    m.add_success("mat_z", "CODE_MAT", category="gemm")
    check("add + get 返回 code", m.get("vec_x")["code"] == "CODE_VEC")
    check("覆盖同 op 为最新", (m.add_success("vec_x", "CODE_VEC2",
          category="elementwise"), m.get("vec_x")["code"])[1] == "CODE_VEC2")

    # --- list_all ---
    all_ = m.list_all()
    check("list_all 共 3 条", len(all_) == 3)

    # --- retrieve 排除自身 ---
    refs = m.retrieve_by_category("elementwise", exclude_op="relu_y")
    check("retrieve 排除自身", all(r["op"] != "relu_y" for r in refs))
    check("retrieve k 上限", len(m.retrieve_by_category("elementwise", k=1)) == 1)

    # --- 跨 category 不串 ---
    refs2 = m.retrieve_by_category("gemm", exclude_op=None)
    check("跨 category 精确匹配", [r["op"] for r in refs2] == ["mat_z"])

    # --- 真实 op 联动（category 从 registry 推断）---
    m.add_success("vector_add", "def launch(x1, x2, n): ...")   # registry category
    check("真实 op category 推断", m.get("vector_add")["category"] == "elementwise")
    # vector_add 自身不该出现在 retrieve(vector_add) 里
    check("retrieve(真实 op) 排除自身",
          all(r["op"] != "vector_add" for r in m.retrieve("vector_add", k=10)))

    # --- 清理 ---
    shutil.rmtree(tmp, ignore_errors=True)
    print("-" * 50)
    print("memory 自测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
