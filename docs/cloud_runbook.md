# 上云跑批 Runbook（服务器跑性能数字 + KernelBench）

> 目标：在租来的 GPU 服务器上，用自研 Triton Kernel Agent 跑通两件事：
> ① 自研算子族的**性能硬数字**（do_bench vs eager/compile）② 官方 **KernelBench L1** 批量跑。
>
> 相关脚本：`scripts/cloud_run.py`（一键跑批）、`scripts/run_kernelbench.py`（KernelBench 单题）、
> `scripts/check_env.py` / `scripts/smoke_test.py`（预检）、`scripts/run_all.py`（全算子评测）。
>
> ⚠️ 除特别说明外，**所有命令都在项目根目录**（`triton_kernel_agent/`）执行。

---

## 0. 先澄清三个常见误解

1. **不需要 fork KernelBench。** 公开仓库 `git clone` 就行，不需要任何权限；fork 只在"你要在 GitHub 上维护自己的改动 / 给官方发 PR"时才有意义。
2. **本地那份 KernelBench 不用传。** 服务器上重新 clone 即可（官方数据集：level1 100 题 / level2 100 题）。
3. **本项目零第三方依赖。** LLM 调用用标准库 `urllib`，`.env` 是手写解析 → 服务器上只需 **torch（自带 triton）+ numpy**。

---

## 1. 租机器建议

| 项 | 建议 | 备注 |
|---|---|---|
| GPU | **RTX 4090 24G** | 性价比最高；A100/H100 更稳但贵 |
| 驱动 | cu124 需 **≥ 525.60.13** | 上机先 `nvidia-smi` 确认 |
| 权限 | 有 **root** | NCU 采样需要（非 root 要设驱动参数） |
| 磁盘 | ≥ 50G | torch + 检索缓存 + 轨迹文件 |

---

## 2. 上机第一步：确认命令真的落在服务器上

```bash
hostname && nvidia-smi
```

---

## 3. 拉代码（两个仓库）

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/skyxiaozhuai/triton_kernel_agent.git
git clone https://github.com/ScalingIntelligence/KernelBench.git
```

---

## 4. 建环境（关键：**不要单独装 triton**）

```bash
conda create -n triton_env python=3.11 -y
conda activate triton_env
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按服务器 CUDA 版本选
pip install numpy
```

> ⚠️ 不要 `pip install triton`：torch 自带**版本匹配的 bundled triton**；
> 单独装新版会在特定架构（如 sm_75）上编译报错。

---

## 5. `.env` 就位（gitignore 了，不会随 clone 过去）

在 `~/work/triton_kernel_agent/.env` 写入：

```
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
# 可选：关掉推理过程 DEEPSEEK_THINKING=off
```

验证（**不会打印 key**）：

```bash
python scripts/test_llm.py
```

---

## 6. 三步预检（不花 LLM 钱）

```bash
export KERNELBENCH_ROOT=~/work/KernelBench      # ⚠️ 必须设：代码里的默认路径是本机路径
python scripts/check_env.py                     # GPU / torch / triton + 最小 kernel 冒烟
python scripts/smoke_test.py                    # 全算子 reference vs golden 对齐
python scripts/run_kernelbench.py --level 1 --list        # 应列出 100 题
python scripts/run_kernelbench.py --level 1 --id 19 --dry # 加载题目+打印规格（不跑 GPU）
```

`--dry` 通过 = 适配层是通的；去掉 `--dry` 才会真正调用 agent + GPU 判卷。

---

## 7. 一键跑批

```bash
python scripts/cloud_run.py --dry                        # 只打印计划
python scripts/cloud_run.py --only env tests             # 环境自检 + 一把梭测试
python scripts/cloud_run.py --only perf --perf-ops vector_add,softmax,matmul
python scripts/cloud_run.py --only perf --shape 4096,4096,4096   # 覆盖主 case shape
python scripts/cloud_run.py --only kb --kb-ids 1-10      # KernelBench L1 前 10 题（先小批验证）
python scripts/cloud_run.py --only kb                    # 全量 100 题（很贵很慢，务必挂 tmux）
```

**阶段**：`env`（自检）→ `tests`（一把梭）→ `perf`（全算子端到端 + 性能 critic）→ `kb`（KernelBench 逐题跑 agent）

**产物**：`results/cloud_<ts>.md`（汇总：各阶段 exit code / 耗时 / KernelBench 通过 X/Y）
+ `results/summary_*.json` + `results/traj_*.jsonl`（每轮代码与反馈）

---

## 8. 长任务必须挂 tmux

```bash
tmux new -s cloud
python scripts/cloud_run.py --only kb 2>&1 | tee results/log_kb_all.txt
#   Ctrl-b d  脱离
#   tmux attach -t cloud   回来
```

> ⚠️ VS Code 断开 SSH → 集成终端会一起死；**tmux 里的进程不会**。

---

## 9. 坑清单（踩过的）

| 坑 | 现象 | 处理 |
|---|---|---|
| **显存** | KernelBench L1 官方 shape 是 **A100 级**，个别题 OOM | 适配层**没有**内置缩小 shape 的开关（`--shape` 只对自研 8 算子生效）→ 用 `--kb-ids` 跳过，或临时 patch 题目的 shape 常量 |
| **网络** | 服务器访问不了 `api.deepseek.com` | 换 base_url / 走代理 |
| **成本** | kb 全量 = 100 题 × 最多 6 轮 × LLM 调用 | 先 `--kb-ids 1-10` 验证，再分批挂跑 |
| **NCU** | `ERR_NVGPUCTRPERM` | `agent/tools/ncu_profiler.py` 已内置路径探测 + 权限诊断；按提示设 `NVreg_RestrictProfilingToAdminUsers=0` 或用 sudo |
| **结果回传** | `results/` 被 gitignore | 跑完用 `scp` / `rsync` 拉回本地，再回填 README / 简历数字 |

---

## 10. 项目现状速览（给接手的 AI 或人）

- **定位**：LLM Agent 自动生成 / 优化 Triton kernel，可信沙箱判卷（自写 orchestrator，不套框架）。
- **已有算子（8+）**：vector_add / relu / softmax / sum_1d / matmul / layer_norm / conv2d + 融合族（relu_sum / matmul_bias_relu）。
- **核心能力**：Reflexion 闭环、双 critic（正确性 + do_bench 性能）、RAG 经验库（成功 kernel 入库 + 同类检索）、失败样本回灌 v1、AST 静态闸门、NCU 剖析驱动的优化端（beam + prescribe）、KernelBench 适配层。
- **入口**：`scripts/run_agent.py`（单算子）、`scripts/run_all.py`（批量评测）、`scripts/run_opt.py`（优化端）、`scripts/run_kernelbench.py`（KernelBench 单题）、`scripts/cloud_run.py`（上云一键）。
- **文档**：`README.md`（结果表）、`PLAN.md`（计划与扩展路线）、`docs/interview_prep.md`、`docs/resume.md`。
