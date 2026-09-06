# 简历条目（成品版，2026-09-06 已用真实数据回填）

## 中文版（可直接贴简历）

**LLM Agent 驱动的自主 Triton Kernel 生成与优化系统**（个人项目 · 独立完成）
*Agent / Reflexion / Tool-Use / Triton / PyTorch*

**项目概述**：端到端 Autonomous Coding Agent —— 输入 PyTorch 算子签名与语义描述，自动完成 Triton kernel 的「生成 → 编译 → 数值验证 → 性能调优」闭环，全程零人工介入。

- 自研 orchestrator（不套框架），实现「LLM 生成 → 沙箱执行 → 结构化反馈 → Reflexion 迭代」闭环；生成代码在独立子进程 + 超时隔离中运行，主进程零崩溃；全程轨迹 JSONL 化、可复现。
- 设计可信 harness 判卷 + 错误分类解析：模型从不自评，kernel 必须通过编译、shape、与 PyTorch golden 的数值对齐（**fp32 + fp16 双精度、多组 shape（主/非整除/极小）共 5 case**）才判通过。
- 实现**双 critic 终止策略**：正确性对齐 + `triton.testing.do_bench` 性能门槛（vs eager）同时达标才收工，性能优化有轮数预算避免死循环。
- 在 4 类算子（elementwise / row-softmax / 跨 block reduce / GEMM）**12 次独立生成中全部通过（100%），平均 1.7 轮收敛**；支持硬件精度自适应（sm_80+ 自动切 tf32）；新增算子仅需注册表登记一行，agent 代码零改动。
- 扩展：跨任务经验库 RAG v1（成功 kernel 自动入库、同类检索注入参考、可 A/B），代码与完整评测脚本开源在 GitHub。

## 英文版（备用）

**LLM Agent for Autonomous Triton Kernel Generation & Optimization** (Personal Project)

- Built a self-orchestrated coding agent (no agent framework) that turns a PyTorch operator spec into a verified Triton kernel via generate → sandbox-execute → structured-feedback → reflexion, fully machine-graded with runnable, reproducible trajectories.
- Sandboxed untrusted code in subprocesses with timeout; verified correctness against PyTorch eager across **fp32+fp16 and 5 shape cases (main / non-divisible / tiny)** per kernel — the model never grades itself.
- Added a dual critic (correctness + `do_bench` vs eager) so the agent stops only when both pass, with bounded optimization rounds.
- **12/12 (100%) kernels passed across 4 op families (elementwise / softmax / cross-block reduce / GEMM), averaging 1.7 iterations**; hardware-aware precision (auto tf32 on sm_80+); adding an operator is a one-line registry change.
- Extensible: cross-task retrieval-augmented memory v1 (auto-store successful kernels, inject same-class references with A/B switch). Code open-sourced.

> 简历话术提醒：若目标 Agent 岗，面试把重点放在"工具反馈压缩幻觉 / critic 何时停 / 失败归因 / 检索记忆"；Triton 是验证场，别被带进 CUDA 调参细节。数字均可复现（`run_all --repeat 3`）。
