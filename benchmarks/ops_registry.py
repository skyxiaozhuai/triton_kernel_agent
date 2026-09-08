"""算子注册表：所有 benchmark op 统一从这里获取。

约定：每个 op 是一个 Python 模块，暴露以下接口：
    OP_NAME          : str，算子唯一名
    OP_META          : dict，元信息（签名/语义描述/约束 —— 给 LLM 看）
    generate_inputs  : dict{x..., meta} —— 主 case（单点）
    generate_cases   : list[dict] —— 多组 shape（主/非整除/小），
                       runner 与沙箱 harness 全过才算正确
    golden(*inputs)  -> torch.Tensor（PyTorch eager 期望结果）
    reference_triton(*inputs) -> torch.Tensor（手写参考 kernel，不喂给 agent）
    check(out, ref)  -> bool（数值对齐断言）

用法：
    op = get_op("vector_add")
    op.generate_inputs() / op.golden(...) / op.reference_triton(...)
"""
from . import ops
from .ops import add_relu, matmul, relu, softmax, sum_1d, vector_add

_REGISTRY: dict[str, object] = {}

for _mod in (vector_add, relu, add_relu, softmax, matmul, sum_1d):
    _REGISTRY[_mod.OP_NAME] = _mod


def list_ops() -> list[str]:
    return sorted(_REGISTRY)


def get_op(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"未知算子: {name}。可用: {list_ops()}")
    return _REGISTRY[name]


def get_all_ops():
    return dict(_REGISTRY)
