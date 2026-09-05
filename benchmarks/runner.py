"""单算子自检 / 校验工具。

当前阶段（正确性闭环）提供进程内 verify_op / verify_all：
对一个 op 跑 generate_inputs -> reference_triton 与 golden 数值对齐，
用于确认"我们的题目定义 + 参考实现"本身是对的（自检 runner）。

D3 之后，"执行 agent 生成的代码"会提升为独立子进程沙箱
（见 agent/tools/executor.py），此处只管基准自身正确性。
"""
from . import ops_registry


def verify_op(name: str, verbose: bool = True):
    """校验单个 op：reference_triton 必须在所有 case 上与 golden 对齐。"""
    op = ops_registry.get_op(name)
    ok_all, last_ref, last_gold = True, None, None
    for ci, args in enumerate(op.generate_cases()):
        ins = {k: v for k, v in args.items() if k != "meta"}
        ref = op.reference_triton(**ins)
        gold = op.golden(**ins)
        ok = op.check(ref, gold)
        ok_all = ok_all and ok
        last_ref, last_gold = ref, gold
        if verbose:
            print(f"[{name}#{ci}] allclose={ok}  meta={args.get('meta')}")
    return ok_all, last_ref, last_gold


def verify_all(verbose: bool = True) -> dict[str, bool]:
    results = {}
    for name in ops_registry.list_ops():
        ok, *_ = verify_op(name, verbose=verbose)
        results[name] = ok
    return results


if __name__ == "__main__":
    results = verify_all()
    bad = [k for k, v in results.items() if not v]
    print("全部通过 ✔" if not bad else f"失败算子: {bad}")
    raise SystemExit(1 if bad else 0)
