# 简历条目（成品版，2026-09-09 已用真实数据回填）

## 中文版（可直接贴简历）

**LLM Agent 驱动的自主 Triton Kernel 生成与优化系统**（个人项目 · 独立完成）
*Agent / Reflexion / Hardware Profiling / Tool-Use / Triton / PyTorch*

**项目概述**：端到端 Autonomous Coding Agent —— 输入 PyTorch 算子签名与语义描述，自动完成 Triton kernel 的「生成 → 编译 → 数值验证 → 硬件剖析 → 性能调优」闭环，全程零人工介入。

- 自研 orchestrator（不套框架），实现「LLM 生成 → 沙箱执行 → 结构化反馈 → Reflexion 迭代」闭环；生成代码在独立子进程 + 超时隔离中运行，主进程零崩溃；全程轨迹 JSONL 化、可复现。
- **防幻觉前置静态闸门 + 可信判卷**：进沙箱前用 AST 做结构/反作弊/语法三查（畸形或"外包给 torch 计算"的代码直接拦截、不烧 GPU）；判卷 = 编译 + shape + 与 PyTorch golden 数值对齐（**fp32+fp16、主/非整除/极小多组 shape 共 5 case**）+ PASS 双信号；模型从不自评。
- 实现**双 critic + 有界优化**：正确性达标外，`do_bench` vs eager 达标才收工；性能端升级为 **NCU 硬件剖析（roofline）驱动**——用真实 DRAM/SM/占用率/L1-L2 命中指标做反馈循环（`agent/tools/ncu_profiler.py` + `opt_loop.py`）。
- 覆盖 4 大类 kernel 形态（elementwise / softmax / 跨 block reduce / GEMM）**12 次独立生成全部通过（100%）、平均 1.7 轮收敛**；支持硬件精度自适应（sm_80+ 自动切 tf32）；新增算子仅需注册表登记一行，agent 代码零改动。
- **融合算子族**：add_relu / relu_sum / matmul_bias_relu 单 kernel 融合（中间结果不落全局内存），vs 分离实现分别 **1.65x / 2.56x / 1.18x**。
- **剖析驱动的真实优化产出**：matmul 4096³ 由 NCU 剖析定位 compute-bound 后，做 tile×num_warps 联合参数扫描找到 **+12% 配置（67.6→60.3ms）**；剖析佐证"大 tile 减 DRAM 往返 > 高占用"（DRAM 35%→18%、SM 75%→78%），并发现 tile 与 warps 强耦合不可独立调。
- **hard 算子 + 端到端一键闭环**：新增 hard `layer_norm`（两遍行归约 + weight/bias 仿射），agent **3 轮收敛**（空回复 → 数值接近 → 通过，err~2e-3）；`run_agent --op X --opt` 一键端到端——生成正确后自动接 NCU 剖析驱动的优化端，输出「① 生成 / ② 剖析优化」报告；轨迹可渲染成单文件 **HTML**（失败 → 成功全程可视化，demo/复盘用）。
- **通用大 shape**：任意算子可用 `OP_SHAPE` / `--shape` 调大输入（此前仅 matmul 的 `MATMUL_SHAPE`）——本地即可把 vector_add / softmax / reduce 放大验证访存/计算行为；matmul 4096³ +12% 证据链即靠 shape 调大跑出。
- **算子族扩展（conv / online）**：conv2d、conv2d_pad(stride+pad)、softmax_online（online 单遍 = flash-attention 核心思想）均 **agent 第 1 轮通过**；conv 朴素版经剖析驱动优化 **~30x**（1.78→0.058ms）。覆盖达 elementwise / reduce / norm / GEMM / conv / online 六大形态 12 算子。
- **工程鲁棒性**：LLM thinking 开关（复杂 kernel 优化从卡 40min → 25s、不再推理截断）；全局 Triton guideline 模板每轮注入 system（对齐 KernelAgent `templates/triton_guidelines.j2`，防长迭代失忆）。
- 工程与业界对标：多 seed 竞速 + digest 去重、失败样本回灌（跨算子同类错误检索注入、排除同算子防作弊）、窗口化 Reflexion（对齐 Meta KernelAgent attempt 窗口）、KernelBench 官方评测集适配接口、CI 一把梭测试（11/11）。

## 英文版（备用）

**LLM Agent for Autonomous Triton Kernel Generation & Optimization** (Personal Project)

- Built a self-orchestrated coding agent (no agent framework) that turns a PyTorch operator spec into a verified Triton kernel via generate → sandbox-execute → structured-feedback → reflexion, fully machine-graded with runnable, reproducible trajectories.
- **Anti-hallucination AST gate + trusted grading**: pre-sandbox structural/anti-cheat/syntax checks (malformed or torch-delegated code is rejected before it burns GPU); grading = compile + shape + numeric alignment vs PyTorch eager across **fp32+fp16 and 5 shape cases (main / non-divisible / tiny)** with a double PASS signal — the model never grades itself.
- Dual critic + bounded optimization (correctness + `do_bench` vs eager); the performance side is upgraded to a **hardware-profiling-driven loop** using real **NVIDIA NCU roofline metrics** (DRAM/SM/occupancy/L1-L2 hit rate) as feedback.
- **12/12 (100%) kernels passed across 4 op families (elementwise / softmax / cross-block reduce / GEMM), averaging 1.7 iterations**; hardware-aware precision (auto tf32 on sm_80+); one-line registry for new ops.
- Fused-op family: add_relu / relu_sum / matmul_bias_relu in single kernels (no global-memory round trip) → **1.65x / 2.56x / 1.18x** vs separate kernels.
- **Real profiling-driven win**: on matmul 4096³, NCU pinpointed compute-bound, then a tile×num_warps joint sweep found a **+12% config (67.6→60.3 ms)**; profiling shows larger tiles cut DRAM traffic (35%→18%) and beat higher occupancy — tiles and warps are coupled and can't be tuned independently.
- **Hard op + one-command end-to-end**: added a hard `layer_norm` (two-pass row reduce + affine) — the agent converged in **3 rounds** (empty reply → near-miss → pass, err ~2e-3); `run_agent --op X --opt` runs the full pipeline in one command (correctness agent → NCU-profiling-driven optimizer → a single report); trajectories render to self-contained **HTML** for demos/retro.
- **Generic large shapes**: any op can be scaled up via `OP_SHAPE` / `--shape` (previously only matmul's `MATMUL_SHAPE`) — lets me verify memory/compute behavior for vector_add / softmax / reduce locally; the matmul 4096³ +12% evidence chain relies on this.
- **Op-family expansion (conv / online)**: conv2d, conv2d_pad (stride+pad), softmax_online (online single-pass = the flash-attention idea) all passed on **round 1**; profiling-driven optimization took the naive conv from 1.78 → 0.058 ms (**~30x**). Coverage spans elementwise / reduce / norm / GEMM / conv / online — 12 ops.
- **Engineering robustness**: an LLM thinking on/off switch (complex-kernel optimization went from a 40-min hang to 25 s, no more reasoning truncation) and a global Triton-guideline template injected into the system prompt every round (mirrors KernelAgent's triton_guidelines.j2; prevents long-iteration drift).
- Engineering & industry alignment: multi-seed racing with dedup, failure-sample re-injection (cross-op retrieval, self-op excluded to avoid cheating), windowed Reflexion (aligned with Meta KernelAgent's attempt window), KernelBench adapter, and a one-shot CI test suite (11/11).

> 简历话术提醒：若目标 Agent 岗，面试把重点放在"工具反馈压缩幻觉 / critic 何时停 / 失败归因 / 剖析驱动调优闭环"；Triton 是验证场，别被带进 CUDA 调参细节。所有数字真实可复现：12/12（`run_all --repeat 3`，9-06 稳健评测）、融合加速（128³ 小 shape 的 bench_fused）、matmul +12%（4096³，复现命令见 README/PLAN §11）。
