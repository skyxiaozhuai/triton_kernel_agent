# Triton Kernel Generation Agent

![CI](https://github.com/skyxiaozhuai/triton_kernel_agent/actions/workflows/ci.yml/badge.svg)

给定 PyTorch 算子签名与语义描述，LLM Agent 自动生成 Triton kernel，并通过「编译 → 数值验证 → 性能调优」闭环自主迭代，直到通过机器打分（正确性对齐 PyTorch、性能达标）。

> 📌 状态（2026-09-06 起步 · 2026-09-08 扩展）：M1 正确性**稳健评测 12/12 通过（100%）**（4 算子 × repeat 3，平均 1.7 轮收敛，每个 kernel 过 fp32+fp16 多 case 判卷）；双 critic（正确性 + do_bench 性能门槛）；硬件精度自适应（sm_80+ 自动切 tf32）。
>
> **2026-09-08（借鉴 PyTorch 官方 KernelAgent）**：AST 结构闸门 + 反作弊静态扫描（禁 torch 外包/反射）、PASS 双信号、多 seed 竞速（`--seeds N` 任一通过即早停）、难度路由（按 op 难度自动分配 seed）、**融合算子 `add_relu`**（单 kernel 融合 add+relu）、**失败样本回灌 v1**（跨算子借鉴同类错误修复示范 = 受控自改进）。代码 ~2300 行，git + GitHub 已同步。
>
> **2026-09-09（本地收口）**：**端到端一键闭环** `run_agent --op X --opt`（生成正确 → NCU 剖析 → 优化 一条龙出报告）；新增 **hard 算子 `layer_norm`**（两遍行归约 + 仿射，agent 3 轮收敛）；**轨迹 HTML 报告** `report_traj_html.py`；**通用大 shape**：任意算子可 `OP_SHAPE` / `--shape` 调大主 case；普通闭环加**同轮空代码自动重试**（推理模型截断致 content 为空时不再浪费轮次）。一把梭 10/10。

---

## 当前算子集

| 算子 | 类别 | 考察点 | 默认规模 |
|---|---|---|---|
| `vector_add` | elementwise 1D | block + mask | N=2^20 |
| `relu` | elementwise 1D（同族） | block + mask | N=2^20 |
| `add_relu` | **fused 1D**（add+relu 单 kernel） | 融合语义、中间结果不落全局内存 | N=2^20 |
| `relu_sum` | **fused reduction**（relu 并入跨 block 归约） | 融合省整张中间读写 | N=2^20 |
| `matmul_bias_relu` | **fused GEMM**（GEMM+epilogue bias/relu） | 单 kernel epilogue、中间 C 不落全局 | 128³ |
| `softmax` | row-reduce 2D | axis 归约、数值稳定（减 row max） | 1024×1024 |
| `matmul` | GEMM 2D | `tl.dot`、K 循环、fp32 累加 | 128³ |
| `sum_1d` | reduction 1D | 跨 block 归约（两阶段） | N=2^20 |
| `layer_norm` | **hard** two-pass row-reduce + affine | 行 mean/var 两遍归约、归一 + weight/bias 仿射融合 | 1024×512 |

> 新增算子：在 `benchmarks/ops/` 建模块，然后在 `ops_registry.py` 的 `for _mod in (...)` 里登记即可。
>
> 主 case 形状可调大（2026-09-09）：任意算子可设 `OP_SHAPE`（逗号分隔各维）或用 `run_agent/run_opt --shape` 调大输入（如 `OP_SHAPE=8388608`、`OP_SHAPE=2048,2048`、matmul `4096,4096,4096`），判卷/剖析/优化子进程自动继承；matmul 亦兼容历史 `MATMUL_SHAPE`。

---

## 评测结果（2026-09-06，真实运行）

### 正确性（M1，稳健评测：4 算子 × repeat 3 = 12 次独立生成）

| 算子 | 类别 | 通过 | 平均轮数 | 末轮 err(中位) |
|---|---|---|---|---|
| `vector_add` | elementwise | 3/3 ✔ | 1.0 | 0 |
| `softmax` | row-reduce | 3/3 ✔ | 2.0 | 3.8e-6 |
| `sum_1d` | reduction | 3/3 ✔ | 2.3 | 1.8e-4 |
| `matmul` | GEMM | 3/3 ✔ | 1.3 | 3.9e-3 |

> **12/12 自动生成通过（100%）**，平均 1.7 轮收敛；每个 kernel 均经 **fp32×3 + fp16×2 共 5 组 case**（主/非整除/极小 shape）判卷全对齐才算 pass。matmul 曾 6 轮失败（模型不知 sm_75 需 `input_precision="ieee"`），把硬件约束写入算子规格后 → 1–2 轮通过。

### 性能（do_bench，GTX1650 / sm_75 / 小 shape，趋势参考）

| 算子 | triton(ms) | eager(ms) | vs eager |
|---|---|---|---|
| `vector_add` | 0.076 | 0.076 | 0.997x（带宽饱和型） |
| `softmax` | 0.052 | 0.054 | 1.042x |

> agent 生成 kernel 带性能 critic：`vector_add --perf` 通过时 speedup_vs_eager=0.917x。正式性能对比（含 torch.compile）建议在服务器大 shape 上跑（本机数字仅验证链路）。

### 融合 vs 分离（2026-09-08，GTX1650 / sm_75 / 主 case，趋势参考）

| 融合算子 | fused(ms) | separate(ms) | 加速比 | 说明 |
|---|---|---|---|---|
| `add_relu` | 0.076 | 0.126 | **1.66x** | 省一次整张中间写+读 |
| `relu_sum` | 0.033 | 0.084 | **2.56x** | relu 并入归约，省全张量中间 |
| `matmul_bias_relu` | 0.014 | 0.016 | **1.18x** | GEMM epilogue 融合（128³ 小 shape，收益待大 shape）|

> 数值一致性副检通过（融合版 == 分离版 allclose）。`python scripts/bench_fused.py` 可复现。elementwise/reduction 融合在小 shape 已明显；**GEMM epilogue 融合（matmul_bias_relu）需大 shape**（省 [M,N] 中间写读），服务器跑更大 shape 会更显著。
>
> **agent 端到端生成验证**：`add_relu` 第 1 轮通过（err=0）；`matmul_bias_relu` 第 1 轮通过（`--memory` 注入同族 `matmul` 成功样本作参考，竞速 2 seed 全过）。

---

## 可信性与自改进（2026-09-08 · 借鉴 PyTorch KernelAgent）

- **AST 静态闸门**（进沙箱前，不烧 GPU）：结构（必含 `def launch` + ≥1 `@triton.jit` kernel + launch 必须真实调用 kernel）+ 反作弊（kernel 内禁 torch、全代码禁 torch 计算外包 / `@` / 反射 / 危险 import）。比官方"strip 注释+正则"更精确（AST 区分 `tl.*` 与 torch 调用），且多了官方没有的"launch 必须调 kernel"约束。
- **PASS 双信号**：哨兵 JSON `ok` **且** `returncode==0` 才算通过，堵"空跑/静默吞错"假阳性。
- **多 seed 竞速 + digest 去重**（`--seeds N`）：N 个独立 seed 并行，任一判卷通过即早停其余；共享 sha256 缓存避免重复代码重复烧 GPU。
- **难度路由**：按 `OP_META.difficulty` 自动分配 seed（easy=1/medium=2/hard=3），简单问题不多花、难问题不赌单一路线。
- **失败样本回灌 v1（受控自改进）**：成功 run 自动把"最后失败→成功"修复对入库；后续失败时检索【其它算子】同类错误的历史修复示范注入反馈（排除同 op 防作弊）。这是官方 KernelAgent 未实现、本项目的差异化点。
- **融合算子家族**（`add_relu` / `relu_sum`）：规格强制单 kernel 融合、中间结果不落全局内存（对齐官方 Fuser 理念的最简演示）；`scripts/bench_fused.py` 量化融合 vs 分离收益（本机 1.66x / 2.44x）。

## KernelBench 适配（2026-09-08 · 接口就绪，真跑待服务器）

把官方 KernelBench 题目（`Model.forward` + `get_inputs`）接入本闭环：

- **接口**：`benchmarks/kernelbench/problem.py`（加载/规格/判卷句柄）+ `executor_kb.py`（子进程判卷，golden = **eager forward**，可信且与官方语义一致）+ `kb_loop.py`（复用静态闸门/Reflexion）+ `scripts/run_kernelbench.py`（CLI）。
- **判卷**：`get_inputs()` 生成输入（多 case）→ eager `forward` 当 golden → shape/dtype/allclose(rtol/atol=1e-2) 全过才算 PASS；静态闸门仍强制 agent 真写 kernel（禁 torch 计算外包）。
- **命令**：`python scripts/run_kernelbench.py --level 1 --list`（列题）；`... --level 1 --id 19 --dry`（只加载+打印规格）；`... --level 1 --id 19 --cases 2`（真跑 agent 生成+判卷，**需 GPU/显存，放在服务器**）。
- **边界**：多数 L1 默认 shape 是 A100 级（如 19_ReLU ≈ 6GB），本机 4G 跑不了 → 真跑/批量与官方 scorer 对比都在服务器；根目录默认 `/home/claude/agent_project/KernelBench`，可用 `KERNELBENCH_ROOT` 覆盖。

## 优化端 run_opt（2026-09-09）

把一个**已正确** kernel 做 hardware-guided 持续优化（借鉴 KernelAgent opt）：

```
基线(经验库/文件/先生成) → 每轮 [NCU 剖析(roofline) → LLM 优化 → 验证+do_bench]
   → 更快(≥2%)则接受并重剖析 / 否则记 rejected → 连续无改进收敛 → best + 曲线
```

- `python scripts/run_opt.py --op vector_add`（经验库起步）；`--generate` 先生成；`--no-ncu` 关剖析；`--shape 8388608 / 2048,2048 / 4096,4096,4096` 覆盖**任意算子**主 case 形状（内部设 env `OP_SHAPE`，判卷/剖析子进程自动继承；matmul 亦兼容历史 `MATMUL_SHAPE`）；`--opt-rounds/--stall/--improve-min` 控制收敛。
- **beam + 诊断先行（对齐官方 beam_search + BottleneckAnalyzer）**：`--beam N` 每轮并行 N 个候选、`--prescribe` 每轮先让 LLM 基于剖析归纳瓶颈并给出互斥优化方向 → 逐候选验证取最快，跳出"单条贪心轨迹"的局部最优。真机（vector_add, beam=2+prescribe）：基线 0.0888 → 0.0755ms，**一轮 accepted +17.6%**。
- 输出：收敛曲线表 + `results/opt_<op>_<ts>.json/.py`（best 代码）。
- 本机验证（vector_add）：基线 ~0.075ms 已 memory-bound（DRAM 91%）；起点即近极限时 `run_opt` 会诚实判"未提速"收敛 —— 本机出趋势，真提升在服务器大 shape + tf32。
- matmul 大 shape 参数扫描（`scripts/sweep_matmul.py`）：对经验库 kernel 扫 tile×num_warps×num_stages + do_bench（本地、无 LLM），**4096³ 实测找到 +12% 配置** —— 见下方证据链。
- 已知注意：优化 prompt 较长，max_tokens 需给足（默认 16384），否则推理模型会截断导致空代码。

### matmul 4096³ 参数扫描证据链（2026-09-09，GTX1650/sm_75/fp32-ieee）

| 阶段 | 手段 | 结果 |
|---|---|---|
| 1. 剖析基线(64×64×32+nw4) | `ncu` | compute-bound：SM 75.2% / DRAM 35.3% / 占用 49.8% → 只有 tile 级优化有空间 |
| 2. LLM run_opt 盲改 2 轮 | `run_opt.py` | 82.9/65.7ms 未达标 → 模型盲试**缺 tile 扫描维度**（只动 block size 附近/噪声） |
| 3. 扫 num_warps/stages | `sweep_matmul.py` | 无效 → 参数强耦合，tile 不动时独立调不出提升 |
| 4. **tile×warps 联合扫** | 同上（`--tiles` 扩网格） | **128×128×32 + nw8 = 60.3ms（vs 基线 67.6ms，+12.1%，正确性 ✔）** |
| 5. 剖析调优配置 | `ncu` | SM 77.6%(升) / **DRAM 18.1%(近减半)** / 占用 25.0%(更低) → 减主存往返是主因 |

- 调优代码存档：`results/opt_matmul4096_best.py`（证据链见同名 `.json`），可 `import` + `launch` 直用。
- 反直觉点：**更快的那版占用率反而更低（25% vs 50%）** —— 大 tile 每线程计算更多、每 SM 塞不下那么多 block，但 DRAM 流量近减半 + SM 效率略升 → 整体更快。教训：占用率须结合 SM/DRAM 一起读，单一 `under-utilized` 会误导优化方向。
- 参数耦合教训：小 tile(64) 配 nw8 = 84ms（比基线还慢 26%）→ **大 tile 必须配大 num_warps，tile 与 warps 不能独立调**。
- 复现：`MATMUL_SHAPE=4096,4096,4096 python scripts/sweep_matmul.py --op matmul --tiles 64x64x32,128x128x32 --warps 4,8 --stages 2,3`

## Agent 工作流

```mermaid
flowchart LR
    U[算子签名+语义] --> C[LLM 生成 kernel+launch]
    C --> V{静态闸门<br/>AST 结构 + 反作弊}
    V -->|否| Fb[反馈重试<br/>(不进沙箱)]
    V -->|是| E[沙箱: 可信 harness 判卷<br/>生成输入+golden+子进程+超时]
    E --> Dc{正确性<br/>数值对齐?}
    Dc -->|否| F[error_parser 结构化反馈]
    F --> R[Reflexion + 失败回灌: 注入同类修复示范]
    R --> C
    Dc -->|是| P[性能 critic: do_bench vs eager]
    P -->|达标 或 优化轮用尽| Done[记录轨迹/经验库 + 报告 ✔]
    P -->|不达标| R
```

> 失败回灌需 `--memory` 开启；`--seeds N` 并行竞速（默认按难度自动）。

---

## 目录结构与职责

| 路径 | 职责 |
|---|---|
| `PLAN.md` | 两周计划、简历成品条目、面试问答清单 |
| `benchmarks/` | **算子集**：被 agent 挑战的"题目"与 ground truth（golden） |
| `benchmarks/ops_registry.py` | op 注册表：`list_ops()` / `get_op(name)` |
| `benchmarks/runner.py` | 单算子自检：`verify_op`（reference vs golden 数值对齐） |
| `benchmarks/ops/<name>.py` | 每个算子一个模块（接口约定见下文） |
| `agent/loop.py` | orchestrator 主循环：双 critic（正确性 + 性能）+ Reflexion + 轨迹 |
| `agent/tools/executor.py` | 沙箱执行器：可信 harness 判卷、子进程隔离、可选 do_bench |
| `agent/tools/error_parser.py` | 错误分类 → 结构化反馈 |
| `agent/tools/static_check.py` | 静态闸门：AST 结构 + 反作弊扫描（进沙箱前） |
| `agent/tools/executor_kb.py` | KernelBench 判卷执行器（子进程，eager forward 当 golden） |
| `agent/tools/ncu_profiler.py` | NCU 剖析工具：采 Triton kernel roofline 指标 → 优化反馈 |
| `agent/kb_loop.py` | KernelBench agent 闭环（生成→静态闸门→判卷→Reflexion） |
| `agent/memory.py` | RAG 经验库 v1 + 失败样本回灌 v1（results/memory/） |
| `benchmarks/kernelbench/` | KernelBench 适配层：problem 加载 / 规格 / 判卷句柄 |
| `agent/tools/benchmark.py` | do_bench 性能基准（vs eager / torch.compile） |
| `agent/llm/` | LLM client（读 .env）+ prompt 模板 |
| `agent/roles/` | Planner / Coder / Critic 角色（待拆分，暂并入 loop） |
| `scripts/check_env.py` | 环境自检：GPU / torch / triton + 最小 kernel 冒烟 |
| `scripts/smoke_test.py` | 全算子冒烟（reference vs golden） |
| `scripts/run_agent.py` | 单算子 agent CLI（`--perf` / `--memory` / `--seeds N` 竞速+难度路由） |
| `scripts/bench.py` | 性能对比表 CLI |
| `scripts/bench_fused.py` | 融合 vs 分离算子性能对比 CLI |
| `scripts/run_all.py` | 批量评测汇总 CLI（`--repeat` 可算成功率） |
| `scripts/run_all_tests.py` | 一把梭自测（core/agent/gpu 分组，`--ci` 供 CI） |
| `scripts/vis_traj.py` | 轨迹可视化 / 聚合统计（复盘为什么绕 N 轮） |
| `scripts/report_traj_html.py` | 轨迹 → 单文件 HTML 报告（summary + 逐轮卡片，demo/录屏用） |
| `scripts/run_kernelbench.py` | 跑官方 KernelBench 题目（`--list/--dry/--level/--id`，真跑需 GPU） |
| `scripts/ab_memory.py` | RAG 经验库 A/B（memory on/off 对比成功率/均轮/token，并行） |
| `scripts/profile_kernel.py` | NCU 剖析 CLI（`--op --from-memory/--ref/--code-file` → roofline 反馈） |
| `scripts/run_opt.py` | 优化端闭环（剖析→LLM 优化→验证→收敛，`results/opt_*`） |
| `.github/workflows/ci.yml` | GitHub Actions：CPU 环境跑非 GPU 单测（core+agent） |
| `results/` | 每次 agent 运行的轨迹 jsonl 与汇总报告（gitignore，不入库） |
| `requirements.txt` | 依赖与安装策略说明 |

## 算子模块接口约定

每个 op 模块（`benchmarks/ops/*.py`）须暴露：

- `OP_NAME`：唯一名
- `OP_META`：给 LLM 看的语义/签名/约束（dict）
- `generate_inputs(n=None, device, dtype) -> {x..., meta}`
- `golden(*inputs)`：PyTorch eager 期望结果（机器打分 ground truth）
- `reference_triton(*inputs)`：手写参考 kernel —— **仅供自检，绝不喂给 agent**
- `check(out, ref) -> bool`：数值对齐断言

## 快速开始

```bash
conda activate triton_env
python scripts/check_env.py                          # ① 环境自检（需 GPU）
python scripts/smoke_test.py                         # ② 算子 reference vs golden 全通过
python scripts/run_agent.py softmax                  # ③ 单算子 agent 生成（正确性闭环）
python scripts/run_agent.py vector_add --perf        # ④ 双 critic（含性能门槛）
python scripts/bench.py                              # ⑤ do_bench 性能对比表
python scripts/run_all.py --perf                     # ⑥ 批量评测汇总
python scripts/run_all_tests.py --ci                 # ⑦ 一把梭自测(--ci 免 GPU；去 --ci 含 GPU executor)
python scripts/vis_traj.py --latest                  # ⑧ 查看最近一条 agent 轨迹(复盘/可观测)
python scripts/run_agent.py matmul --opt --shape 4096,4096,4096  # ⑨ 端到端：生成正确→NCU 剖析→优化 一条龙
python scripts/report_traj_html.py --latest          # ⑩ 最近轨迹渲染成 HTML 报告(浏览器打开/demo)
```

> 首次运行前在项目根 `.env` 配好 `DEEPSEEK_API_KEY`（已被 .gitignore 忽略）。所有命令建议用 `triton_env` 环境的 python 执行（本机 shell 常停在 base，用绝对路径 `/home/claude/miniconda3/envs/triton_env/bin/python`）。

## 开发约定

- **GPU 分工**：笔记本 GTX1650（4G, sm_75）只做小 shape 冒烟；性能/大 shape/GEMM 在服务器跑。
- **Triton**：用 torch bundled 版本，勿单独 `pip install triton`（避免 sm_75 兼容坑）。
- 手写 `reference_triton` 相当于"答案"，只用于自检 runner，**绝不**出现在喂给 LLM 的上下文里。
