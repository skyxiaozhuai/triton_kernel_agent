"""KernelBench problem 加载 + 规格 + 判卷句柄（不依赖 GPU / LLM）。

用法：
    prob = load_problem("/path/to/KernelBench/KernelBench/level1/19_ReLU.py")
    spec = prob.spec_text()            # 给 LLM 的完整规格（源码 + 契约）
    inputs = prob.get_inputs()         # 题目输入（可能巨大，A100 级）
    gold = prob.golden(*inputs)        # eager forward = 可信 golden
    prob.check(out, gold)              # 数值判卷
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path

import torch

# KernelBench 判卷容差（eager kernel 与 eager forward 的常见对齐水平）
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2

# 给 LLM 的硬契约（进沙箱前由 static_check 校验）
CONTRACT = (
    "请用 PyTorch Triton 实现与上面 Model.forward 语义完全等价的 kernel。\n"
    "硬性要求：\n"
    "1. 必须提供 @triton.jit 定义的 kernel，并在 def launch(*inputs) -> torch.Tensor "
    "里调用它（launch 的入参与 get_inputs() 返回的顺序一一对应，位置传参）；\n"
    "2. 输出张量 dtype/shape 必须与 forward 返回一致；\n"
    "3. 禁止在代码里调用 torch 做 forward 的计算（会触发反作弊闸门）；\n"
    "4. 输入可能非常大，grid 要按实际元素数正确配置（可用 mask 处理非整除）。"
)


def default_kernelbench_root() -> Path:
    """KernelBench 根目录：env KERNELBENCH_ROOT 优先，否则默认路径。"""
    env = os.environ.get("KERNELBENCH_ROOT")
    if env:
        return Path(env)
    return Path("/home/claude/agent_project/KernelBench")


def resolve_problem(level: int, problem_id: int,
                    root: Path | str | None = None) -> Path:
    """按 level + id 解析题目文件路径（levelN/<id>_*.py）。"""
    base = Path(root) if root else default_kernelbench_root()
    lvl = base / "KernelBench" / f"level{level}"
    if not lvl.is_dir():
        # 允许根直接含 levelN（不同 clone 布局）
        lvl = base / f"level{level}"
    hits = sorted(lvl.glob(f"{problem_id}_*.py"))
    if not hits:
        raise FileNotFoundError(f"在 {lvl} 下找不到 {problem_id}_*.py")
    return hits[0]


class KBProblem:
    """单个 KernelBench 题目的句柄。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.stem
        self.source = self.path.read_text(encoding="utf-8")
        m = re.search(r"/(level\d+)/", self.path.as_posix())
        self.level = int(m.group(1)[5:]) if m else None
        self._module = self._load_module()

    # ---------- 模块加载 ----------
    def _load_module(self):
        path = os.path.abspath(str(self.path))
        sys_path_insert = os.path.dirname(path)
        if sys_path_insert not in sys.path:
            sys.path.insert(0, sys_path_insert)
        spec = importlib.util.spec_from_file_location("_kb_problem", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 problem: {self.path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @property
    def module(self):
        """原始模块（测试可临时 patch 其 shape 常量来缩小输入）。"""
        return self._module

    @property
    def model_cls(self):
        return self._module.Model

    # ---------- 输入 / golden（调用题目自身逻辑） ----------
    def get_inputs(self) -> list[torch.Tensor]:
        return self._module.get_inputs()

    def get_init_inputs(self) -> list:
        return self._module.get_init_inputs()

    def make_model(self):
        return self.model_cls(*self.get_init_inputs())

    def golden(self, *inputs) -> torch.Tensor:
        """eager forward 作为可信 golden（题目自身实现）。"""
        return self.make_model()(*inputs)

    def check(self, out: torch.Tensor, gold: torch.Tensor,
              rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
        if out.shape != gold.shape:
            return False
        if out.dtype != gold.dtype:
            return False
        return bool(torch.allclose(out.float(), gold.float(), rtol=rtol, atol=atol))

    # ---------- 给 LLM 的规格 ----------
    def forward_src(self) -> str:
        try:
            return inspect.getsource(self.model_cls.forward)
        except (OSError, TypeError):
            return ""

    def spec_text(self) -> str:
        head = (f"# KernelBench {self.level and f'Level {self.level} · '}{self.name}\n"
                f"# 题目源码（class Model + forward + get_inputs）如下：\n")
        return head + self.source + "\n" + CONTRACT

    def __repr__(self):  # pragma: no cover
        return f"KBProblem({self.path})"


def load_problem(path: str | Path) -> KBProblem:
    return KBProblem(path)
