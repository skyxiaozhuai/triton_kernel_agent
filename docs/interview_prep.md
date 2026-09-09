# 面试速记卡 —— Triton Kernel Generation Agent

> 用途：秋招前快速过一遍。要点式，配合仓库 README（架构图）与代码使用。
> GitHub: https://github.com/skyxiaozhuai/triton_kernel_agent

## 1. 一句话定位 + 30 秒陈述

**一句话**：基于 LLM 的自主 Triton kernel 生成与优化 Agent —— 给定 PyTorch 算子描述，自动「规划 → 生成 → 编译 → 数值验证 → 性能调优」，通过可信 harness 机器判卷自我迭代直到正确性与性能达标。

**30 秒陈述（背熟）**：
> "我实现了一个自动生成并优化 Triton kernel 的 agent。输入算子的语义描述，它自己写 kernel、在沙箱里编译运行、和 PyTorch eager 结果做数值对齐；不对就解析错误反馈回去改，直到通过。判卷是可信代码，不是模型自评。我把它做成双 critic——正确性过了还要过 do_bench 性能门槛才算收工。在 4 类算子、每类跑 3 次共 12 次独立生成里全部通过，平均 1.7 轮收敛，每个 kernel 都要同时通过 fp32 和 fp16 的多组 shape 校验。性能端我又接了 NVIDIA NCU 做 roofline 剖析，用真实 DRAM/SM/占用率做反馈驱动优化；在 matmul 4096³ 上通过 tile×num_warps 联合扫描实测提速 +12%（67.6→60.3ms）。整个架构从零手写（agent+工具 ~2100 行，全仓 ~6300），带完整轨迹与评测脚本。"

## 2. 必须记住的数字（面试随时被抽查）

| 指标 | 数值 | 口径 |
|---|---|---|
| 生成通过率 | **12/12 (100%)** | 4 类算子 × repeat 3 独立生成（9-06 稳健评测） |
| 平均收敛轮数 | **~1.7 轮** | (matmul 1.3 + softmax 2.0 + sum 2.3 + vector 1.0)/4 |
| 覆盖算子 | vector_add / relu / softmax / sum_1d / matmul / add_relu / relu_sum / matmul_bias_relu / **layer_norm** | 4 大类 kernel 形态 + 融合族 3 + hard 1 |
| 判卷强度 | 每 kernel **5 case** | fp32×3 + fp16×2（主/非整除/极小 shape）|
| 硬件剖析 | **NCU 真实 roofline** | DRAM/SM 吞吐、占用率、L1/L2 命中（优化端信号源） |
| 剖析驱动优化 | matmul 4096³ **+12%（67.6→60.3ms）** | tile×num_warps 联合扫描 → 128×128×32+nw8 |
| hard 算子 layer_norm | agent **3 轮收敛**(err 1.95e-3) | 空回复 → correctness → pass（第二个失败归因案例） |
| 优化端 beam | 一轮 **+17.6%**(vector_add) | beam2+prescribe：诊断方向 → 多候选 → 取最快 |
| 融合 vs 分离 | add_relu **1.65x** / relu_sum **2.56x** / matmul_bias_relu **1.18x** | 128³ 小 shape（GEMM 收益需大 shape） |
| 防幻觉闸门 | AST 静态三查 + PASS 双信号 | 结构/反作弊/语法，畸形代码不烧 GPU |
| 测试/CI | 一把梭 **11/11** | core/agent/gpu 分组(含 opt_beam/report_html) + GitHub Actions |
| 代码量 | 全仓 ~6300 行 | agent+tools ~2100 + benchmarks/ops ~1400 + scripts/CLI/测试 ~2800 |
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

> "第二个案例（优化端）：matmul 4096³ 上 run_opt 两轮 LLM 盲改都没达标（82.9/65.7ms）。剖析后归因：起点已 compute-bound、模型只在 block size 附近做噪声微调，**缺 tile 扫描维度**。修复：我写工具做 tile×num_warps×stages 联合本地扫描（无 LLM），找到 +12%。教训：模型盲试不是万能，剖析定位 + 参数化搜索比多让 LLM 猜几轮更可靠——这条回答能体现'工具化思维'。"

**Q8 沙箱怎么保证主进程安全？**
> "生成代码在独立子进程 + 超时 kill 里跑，崩了不连累 orchestrator；sandbox 把不可信代码和可信判卷隔离，主进程零崩溃。"

**Q9 RAG/经验库是干嘛的？**
> "v1：成功 kernel 自动入库，按算子类别检索同类（排除自身防作弊）作为生成参考，目标是新算子借鉴同类更快收敛。目前是零依赖 category 匹配、带 --memory 开关可做 A/B。升级路径是 embedding 语义检索甚至接 LangChain——但原则是先证明价值再上框架。"

**Q10 跟业界 KernelAgent / KernelBench 的区别？**
> "Meta 的 KernelAgent 是工业级：多 worker/多 seed 并行、从 PyTorch 程序提取子图、NCU 剖析 + beam 优化。我做的是能跑通的最小闭环，但核心判卷理念一致且我从零独立实现：可信 harness + 禁 PyTorch fallback + eager 当 ground truth。这周我对照官方源码又落地了三处同源能力：① AST 静态闸门（官方用正则 strip 注释，我用 AST 更精确地区分 tl.* 与 torch 调用）② 窗口化 Reflexion（对齐它的 attempt_history：滑动窗口压缩历史，不无限堆消息）③ NCU roofline 剖析做优化反馈 + 多 seed 竞速。规模差异我选择不过度工程：不搬它的多进程 GPU 锁、28 维 NCU 指标、Fuser 子图全链。"

**Q11 性能是怎么优化的？讲讲优化端。
> "两条腿：① 工具化参数搜索（本地 do_bench，无 LLM）：tile×num_warps×stages 联合扫，因为实验发现这些参数强耦合——小 tile(64) 配 num_warps=8 反而比基线慢 26%，必须联合调。② 剖析驱动：每个版本先用 NCU 采真实 roofline（DRAM/SM/占用率/L1-L2），判断 memory-bound 还是 compute-bound 再决定优化方向，避免盲改。在 matmul 4096³ 上这条闭环找到了 +12%：基线 64×64×32+nw4=67.6ms → 128×128×32+nw8=60.3ms。"

**Q12 调优版占用率只有 25%、反而更快，怎么解释？**
> "反直觉点但合理：大 tile 让每线程承担更多计算（寄存器占用高），每 SM 塞下的 block 变少 → 占用率下降；但它换来两个好处：DRAM 流量近减半（剖析：35%→18%）+ SM 效率略升（75%→78%），对 compute-bound 的 GEMM 而言减主存往返 > 堆占用。所以占用率必须结合 SM/DRAM 一起读，单一 under-utilized 提示会误导方向——这也是我坚持用剖析而非拍脑袋调参的原因。"

**Q13 推理模型输出为空(截断)怎么办？
> "layer_norm 首轮 6 轮里 5 轮 content 为空——归因：推理模型(deepseek)把 max_tokens(当时 8192)全打满在 reasoning，没剩给代码。修复两招：① max_tokens 提到 16384；② 给生成循环加**同轮空代码自动重试**(empty_retries=2，同 messages 直接重发、不浪费轮次)。修完 layer_norm 3 轮收敛。这条也说明 coding agent 的工程稳健性：要给'模型偶发失败'留重试预算，不能把一次截断当一轮失败烧掉。"

**Q14 优化端的 beam 是什么？为什么不用一条轨迹跑到底？**
> "单条贪心轨迹易卡局部——matmul 4096³ 那次 LLM 连盲改 2 轮都没用就收敛了。所以对齐官方 beam：每轮先让 LLM 基于剖析**诊断出几个互斥方向**(prescribe)，再朝每个方向各生成一个候选，逐一判卷取最快。等价官方 top-N × M-direction 的轻量版，不用多进程。vector_add 一轮就 +17.6%(2/2 候选过)。代价是 token≈×N，换的是跳出局部最优——适合计算密集、有真实调优空间的算子。"

## 4. 主动讲的"设计亮点"（面试加分）

1. **可信/不可信分离**：判卷权威永远在自己代码里，LLM 只产 kernel
2. **反馈质量 = 收敛速度**：错误分类 + 精炼反馈，而非原始 traceback
3. **双 critic + 有界优化**：正确性硬门槛，性能有预算上限防死循环
4. **硬件感知**：把 sm_75 的 tl.dot 精度约束动态告诉模型（正确性从 6 轮→1 轮的根因）
5. **判卷不断变严**：多 case（非整除/极小 shape）→ fp16 双精度，防止"对一个 shape 蒙对"
6. **全程可复现**：轨迹 jsonl + 评测脚本 + A/B 能力（`run_all --memory on/off`）
7. **剖析驱动调优闭环**：NCU roofline 定 memory/compute-bound → run_opt 优化循环；matmul 4096³ +12%，且用剖析解释了"占用率低反而快"
8. **防幻觉前置闸门**：AST 静态三查把"外包给 torch"的作弊/畸形代码拦在烧 GPU 之前
9. **失败样本回灌 + 多 seed 竞速**：失败→成功修复对跨算子复用；多 seed 任一通过即停 + digest 去重，形成自改进闭环
10. **端到端一键闭环**：`run_agent --op X --opt` 一条命令串起"生成正确 → NCU 剖析 → 优化"，summary 带 final_code 接力优化端，出①②一条龙报告
11. **轨迹 HTML + 诊断先行 + beam**：失败→成功全程可渲染成单文件 HTML(demo)；优化端每轮先诊断瓶颈给互斥方向，再多候选探索取最优(对齐官方 BottleneckAnalyzer + beam)

## 5. 可能的 challenge 与应对

| challenge | 应对 |
|---|---|
| "样本太小（12 次）" | 承认是独立生成的稳健抽样而非大规模；强调每 kernel 过 5 组多精度 case，比"过一个大 shape"严格得多；另有融合 bench 与 4096³ 单点强证据 |
| "带宽饱和算子没加速？" | 剖析直接证明是物理极限（vector_add DRAM 91%）；优化空间在 compute-bound 算子——matmul 4096³ 已实测 +12%（小 shape 128³ 喂不饱 GPU，必须大 shape 才显 compute 特性） |
| "优化端就是靠手动扫描，LLM 没提升？" | 诚实：单一 LLM 盲改在近极限时确实无效（两轮未达标）；但这正说明需要"剖析定位 + 参数化搜索"，LLM 负责结构性重写、工具负责数值搜索，两者互补 |
| "占用率 25% 反而更快？" | 大 tile 减 DRAM 往返 > 堆占用；剖析佐证 DRAM 35%→18%、SM 75%→78%，且 tile×warps 强耦合（小 tile 配大 warps 反而慢 26%） |
| "是不是 torch.compile 套壳？" | 不是，agent 从零生成可编译 kernel 并过机器判卷，compile 只是性能基线 |
| "模型会不会背答案？" | 判卷输入是随机生成、多 dtype/multi case；RAG 检索还刻意排除当前算子防作弊 |
| "为什么不加 bf16/fp8？" | 判卷基建已按 dtype 参数化，加 dtype 只是加 case；fp8 需要 sm_90+ 硬件才值得 |
