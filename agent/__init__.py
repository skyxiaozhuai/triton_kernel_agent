"""agent 核心逻辑（D3+ 逐步实现）。

设计取向：自写 orchestrator（不套 LangGraph），让 Agent 循环可控、可解释。
最终形态见 PLAN.md：Planner → Coder → Executor → Critic → Reflexion。

子包：
- roles/   : Planner / Coder / Critic 角色
- tools/   : executor 沙箱 / error_parser / benchmark
- llm/     : LLM client 与 prompt 模板
"""
