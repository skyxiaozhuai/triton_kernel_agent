"""KernelBench 适配层自测（纯 CPU，无 GPU/无 LLM/无真实 KernelBench 依赖）。

- 用临时目录自造一个迷你 KernelBench problem（Model/forward/get_inputs）；
- 验证 loader / spec_text / get_inputs / golden(eager) / check（含 shape/dtype 拒绝）；
- resolve_problem 的 levelN/<id>_*.py 解析；
- 若本机存在真实 KernelBench（/home/claude/agent_project/KernelBench），额外对
  真实题目(level1/19_ReLU.py，缩小 shape 后)做一次加载 + golden 冒烟。

运行：python scripts/test_kernelbench.py
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.kernelbench import problem as kbp  # noqa: E402

MINI_PROBLEM = '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    """mini relu problem for adapter tests."""
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 8
dim = 64

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
'''


def _write(path: str, code: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def main() -> int:
    failed = 0

    def check(label: str, cond: bool) -> None:
        nonlocal failed
        print(f"{'✔' if cond else '✗'} {label}")
        if not cond:
            failed += 1

    with tempfile.TemporaryDirectory() as d:
        # --- 1. loader + spec ---
        prob_path = _write(os.path.join(d, "mini_relu.py"), MINI_PROBLEM)
        prob = kbp.load_problem(prob_path)
        check("loader: name", prob.name == "mini_relu")
        spec = prob.spec_text()
        check("spec: 含 forward 源码", "torch.relu(x)" in spec)
        check("spec: 含 launch 契约", "launch(*inputs)" in spec
              and "@triton.jit" in spec and "get_inputs" in spec)

        # --- 2. get_inputs / golden(eager) / check ---
        ins = prob.get_inputs()
        check("get_inputs: 1 个输入且 shape", len(ins) == 1
              and tuple(ins[0].shape) == (8, 64))
        gold = prob.golden(*ins)
        check("golden: eager forward 形状", tuple(gold.shape) == (8, 64))
        check("check: 自身 golden 通过", prob.check(gold, gold))

        wrong_shape = gold.reshape(64, 8)
        check("check: shape 不符拒绝", not prob.check(wrong_shape, gold))
        wrong_dtype = gold.double()
        check("check: dtype 不符拒绝", not prob.check(wrong_dtype, gold))

        # --- 3. resolve_problem: levelN/<id>_*.py ---
        kbroot = os.path.join(d, "KernelBench")
        os.makedirs(os.path.join(kbroot, "level1"), exist_ok=True)
        _write(os.path.join(kbroot, "level1", "1_mini_relu.py"), MINI_PROBLEM)
        resolved = kbp.resolve_problem(1, 1, root=kbroot)
        check("resolve_problem: level1/1_*.py", os.path.basename(resolved)
              == "1_mini_relu.py")

    # --- 4. 若本机有真实 KernelBench：对 19_ReLU 缩小 shape 冒烟 ---
    real_root = kbp.default_kernelbench_root()
    real_19 = real_root / "KernelBench" / "level1" / "19_ReLU.py"
    if real_19.exists():
        prob = kbp.load_problem(real_19)
        # 缩小题目 shape（仅测试用途；服务器跑原尺寸）
        prob.module.batch_size = 4
        prob.module.dim = 32
        ins = prob.get_inputs()
        gold = prob.golden(*ins)
        check(f"真实题目 19_ReLU: shape={tuple(ins[0].shape)} golden 形状",
              tuple(ins[0].shape) == (4, 32)
              and tuple(gold.shape) == (4, 32))
        check("真实题目 19_ReLU: check(eager vs eager)", prob.check(gold, gold))
    else:
        print("(跳过真实 KernelBench 冒烟：未找到 "
              f"{real_root / 'KernelBench'})")

    print("-" * 50)
    print("kernelbench 适配自测通过 ✔" if failed == 0 else f"{failed} 项失败")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
