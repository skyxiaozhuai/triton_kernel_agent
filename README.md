# Triton Kernel Generation Agent

给定 PyTorch 算子签名与语义描述，LLM Agent 自动生成 Triton kernel，并通过「编译 → 数值验证 → 性能调优」闭环自主迭代，直到通过机器打分（正确性对齐 PyTorch、性能达标）。

> 当前进度：Week1 D1–D2 ✅ —— 环境验证通过；benchmark 算子集已铺 4 个（覆盖 elementwise / softmax / GEMM / 跨 block reduce），手写 reference 均与 golden 对齐。`agent/` 核心待填充（D3+）。

---

## 当前算子集

| 算子 | 类别 | 考察点 | 默认规模 |
|---|---|---|---|
| `vector_add` | elementwise 1D | block + mask | N=2^20 |
| `softmax` | row-reduce 2D | axis 归约、数值稳定（减 row max） | 1024×1024 |
| `matmul` | GEMM 2D | `tl.dot`、K 循环、fp32 累加 | 128³ |
| `sum_1d` | reduction 1D | 跨 block 归约（两阶段） | N=2^20 |

> 新增算子：在 `benchmarks/ops/` 建模块，然后在 `ops_registry.py` 的 `for _mod in (...)` 里登记即可。


---

## 目录结构与职责

| 路径 | 职责 |
|---|---|
| `PLAN.md` | 两周计划、简历成品条目、面试问答清单 |
| `benchmarks/` | **算子集**：被 agent 挑战的"题目"与 ground truth（golden） |
| `benchmarks/ops_registry.py` | op 注册表：`list_ops()` / `get_op(name)` |
| `benchmarks/runner.py` | 单算子自检：`verify_op`（reference vs golden 数值对齐） |
| `benchmarks/ops/<name>.py` | 每个算子一个模块（接口约定见下文） |
| `agent/` | **Agent 核心**（D3+ 实现，当前仅目录占位） |
| `agent/roles/` | Planner / Coder / Critic 角色模块 |
| `agent/tools/` | 执行沙箱 `executor` / `error_parser` 错误解析 / `benchmark` 性能基准 |
| `agent/llm/` | LLM client（DeepSeek/OpenAI 兼容）与 prompt 模板 |
| `scripts/check_env.py` | 环境自检：GPU / torch / triton + 最小 kernel 冒烟 |
| `scripts/smoke_test.py` | 全算子冒烟入口（遍历注册表验证） |
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
python scripts/check_env.py             # ① 环境自检（需 GPU）
python -m benchmarks.ops.vector_add     # ② 单算子冒烟
python scripts/smoke_test.py            # ③ 全算子冒烟
```

## 开发约定

- **GPU 分工**：笔记本 GTX1650（4G, sm_75）只做小 shape 冒烟；性能/大 shape/GEMM 在服务器跑。
- **Triton**：用 torch bundled 版本，勿单独 `pip install triton`（避免 sm_75 兼容坑）。
- 手写 `reference_triton` 相当于"答案"，只用于自检 runner，**绝不**出现在喂给 LLM 的上下文里。
