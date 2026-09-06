# 面试速记卡 —— Triton Kernel Generation Agent

> 用途：秋招前快速过一遍。要点式，配合仓库 README（架构图）与代码使用。
> GitHub: https://github.com/skyxiaozhuai/triton_kernel_agent

## 1. 一句话定位 + 30 秒陈述

**一句话**：基于 LLM 的自主 Triton kernel 生成与优化 Agent —— 给定 PyTorch 算子描述，自动「规划 → 生成 → 编译 → 数值验证 → 性能调优」，通过可信 harness 机器判卷自我迭代直到正确性与性能达标。

**30 秒陈述（背熟）**：
> "我实现了一个自动生成并优化 Triton kernel 的 agent。输入算子的语义描述，它自己写 kernel、在沙箱里编译运行、和 PyTorch eager 结果做数值对齐；不对就解析错误反馈回去改，直到通过。判卷是可信代码，不是模型自评。我把它做成了双 critic——正确性过了还要过 do_bench 性能门槛才算收工。在 4 类算子、每类跑 3 次共 12 次独立生成里全部通过，平均 1.7 轮收敛，每个 kernel 都要同时通过 fp32 和 fp16 的多组 shape 校验。整个架构从零手写，约 1700 行，带完整轨迹与评测脚本。"

## 2. 必须记住的数字（面试随时被抽查）

| 指标 | 数值 | 口径 |
|---|---|---|
| 生成通过率 | **12/12 (100%)** | 4 类算子 × repeat 3 独立生成 |
| 平均收敛轮数 | **~1.7 轮** | (matmul 1.3 + softmax 2.0 + sum 2.3 + vector 1.0)/4 |
| 覆盖算子 | vector_add / relu / softmax / sum_1d / matmul | 4 大类 kernel 形态 |
| 判卷强度 | 每 kernel **5 case** | fp32×3 + fp16×2（主/非整除/极小 shape）|
| 代码量 | ~1700 行 | agent(核心~700) + benchmarks + scripts |
| 双 critic | 正确性 + 性能(do_bench vs eager) | 性能门槛可配 |

## 3. 高频问答（应答要点，别背书）

**Q1 为什么不直接用 torch.compile？**
> "compile 是黑盒优化；我要验证的是'agent 能不能从零写出可编译、数值正确的 kernel'，这需要白盒的可信判卷（编译+数值对齐+性能测量），torch.compile 给不了这个评估闭环。性能对比里我仍把 compile 当基线。"

**Q2 为什么自研 orchestrator 而不用 LangGraph？**
> "我的流程是确定的单闭环：生成→沙箱→判卷→反馈。用几十行循环能表达清楚、完全可控、可解释。LangGraph 的价值在复杂状态机/并行分支/持久化，这里用不到；而且面试要能讲透每步，自研比框架更能体现设计。"

**Q3 为什么 Triton 而不是让模型直接写 CUDA？**
> "Triton 更接近 Python、自动管理 block 调度，让 agent 聚焦算法而不是线程索引细节，生成成功率更高；同时它仍是要认真设计的 kernel（mask/reduce/共享内存语义）。Coding agent 写 CUDA 是后续可扩展方向。"

**Q4 agent 怎么防止"胡编乱造"？**
> "模型从不自我评判。每次生成的 kernel 都进沙箱子进程：能编译吗？shape 对吗？数值与 PyTorch golden 对齐吗（多 dtype 多 shape）？三个关卡全是机器打分。'防幻觉'靠 ground truth 把关，不靠模型自觉。"

**Q5 什么时候停？**
> "双 critic：正确性全 case 对齐 = 硬门槛；性能模式开启时还要 do_bench vs eager 达标才收工。性能优化有轮数预算（perf_retry），避免死循环烧钱——正确性优先、性能有界改进。"

**Q6 错误反馈怎么设计的？为什么不是把 stderr 直接丢给模型？**
> "几百行 traceback 喂给模型收敛慢且烧 token。我用 error_parser 把失败归成 correctness/compile/cuda/syntax/timeout 几类，附一行摘要+定位+该类最常见原因，LLM 只看精炼反馈。反馈质量决定 agent 收敛速度，这是 coding agent 成败的分水岭。"

**Q7 有没有印象深刻的失败案例？**
> "matmul 基线 6 轮失败。归因发现两层问题：一是推理模型偶发空回复，agent 白跑沙箱轮；二是模型不知道目标 GPU(GTX1650/sm_75) 的 fp32 tl.dot 没有 tf32，默认路径内部崩溃。修复：空回复直接拦截不进沙箱 + 把硬件约束写进算子规格让模型一次写对 + 提高 token 上限——matmul 从 6 轮失败变 1 轮通过。这个'发现问题→归因→针对性修复→量化提升'的过程比结果更体现工程能力。"

**Q8 沙箱怎么保证主进程安全？**
> "生成代码在独立子进程 + 超时 kill 里跑，崩了不连累 orchestrator；sandbox 把不可信代码和可信判卷隔离，主进程零崩溃。"

**Q9 RAG/经验库是干嘛的？**
> "v1：成功 kernel 自动入库，按算子类别检索同类（排除自身防作弊）作为生成参考，目标是新算子借鉴同类更快收敛。目前是零依赖 category 匹配、带 --memory 开关可做 A/B。升级路径是 embedding 语义检索甚至接 LangChain——但原则是先证明价值再上框架。"

**Q10 跟业界 KernelAgent / KernelBench 的区别？**
> "Meta 的 KernelAgent 是工业级：多 worker 并行、从 PyTorch 程序提取子图、NCU 硬件剖析优化。我做的是两周内的最小可复现核心闭环——但核心判卷理念一致：可信 harness + 禁 PyTorch fallback + PyTorch eager 当 ground truth，这点我从零独立实现并验证了。我的扩展路线（难度分级、性能剖析、检索记忆）对齐它的能力分层。"

## 4. 主动讲的"设计亮点"（面试加分）

1. **可信/不可信分离**：判卷权威永远在自己代码里，LLM 只产 kernel
2. **反馈质量 = 收敛速度**：错误分类 + 精炼反馈，而非原始 traceback
3. **双 critic + 有界优化**：正确性硬门槛，性能有预算上限防死循环
4. **硬件感知**：把 sm_75 的 tl.dot 精度约束动态告诉模型（正确性从 6 轮→1 轮的根因）
5. **判卷不断变严**：多 case（非整除/极小 shape）→ fp16 双精度，防止"对一个 shape 蒙对"
6. **全程可复现**：轨迹 jsonl + 评测脚本 + A/B 能力（`run_all --memory on/off`）

## 5. 可能的 challenge 与应对

| challenge | 应对 |
|---|---|
| "样本太小（12 次）" | 承认是独立生成的稳健抽样而非大规模；强调每 kernel 过 5 组多精度 case，比"过一个大 shape"严格得多 |
| "带宽饱和算子没加速？" | elementwise 逼近带宽上限是物理预期；性能对比要在服务器大 shape、计算密集算子(GEMM/attention)才显差距——待跑 |
| "是不是 torch.compile 套壳？" | 不是，agent 从零生成可编译 kernel 并过机器判卷，compile 只是性能基线 |
| "模型会不会背答案？" | 判卷输入是随机生成、多 dtype/multi case；RAG 检索还刻意排除当前算子防作弊 |
| "为什么不加 bf16/fp8？" | 判卷基建已按 dtype 参数化，加 dtype 只是加 case；fp8 需要 sm_90+ 硬件才值得 |
