"""通用 shape 覆盖 —— 让**任意算子**都能用 env 调大主 case 输入形状。

背景：matmul 先用 MATMUL_SHAPE 支持了 4096³ 大 shape（compute-bound 才有调优
空间）；这里把它推广到所有算子 —— 跑 bench / 剖析 / run_opt / agent 时都能把
vector_add / softmax / sum_1d … 的输入放大，出更大 shape 的性能趋势。

约定（读 env，子进程自动继承父进程设置）：
- 通用 env `OP_SHAPE="d1,d2,..."`：逗号分隔，**维度数须与该算子主 case 默认一致**
  才生效（如 vector_add=1 维、softmax=2 维 M,N、matmul=3 维 M,K,N），否则忽略。
- matmul 历史别名 `MATMUL_SHAPE="M,K,N"` 继续兼容（优先级最高）。

用法（在 op 模块里）：
    from benchmarks.shape_env import get_op_shape
    DEFAULT = (N_DEFAULT,)  # 或 (M, N) / (M, K, N)
    def _shape():
        return get_op_shape(OP_NAME, DEFAULT)
    # generate_inputs / generate_cases 里用 _shape() 代替写死常量
"""
from __future__ import annotations

import os

_GENERIC = "OP_SHAPE"
# 历史别名：某些 op 早期用专属 env（matmul 的 MATMUL_SHAPE）
_ALIAS = {"matmul": "MATMUL_SHAPE"}


def _parse(raw: str) -> tuple[int, ...] | None:
    try:
        vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError:
        return None
    return vals if vals else None


def get_op_shape(op_name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """读 env 覆盖主 case 形状；解析失败 / 维度不符则回退默认。

    优先 matmul 的 MATMUL_SHAPE（历史），再通用 OP_SHAPE。
    """
    keys = [k for k in (_ALIAS.get(op_name), _GENERIC) if k]
    for key in keys:
        raw = os.environ.get(key)
        if not raw:
            continue
        vals = _parse(raw)
        if vals is not None and len(vals) == len(default):
            return vals
    return tuple(default)
