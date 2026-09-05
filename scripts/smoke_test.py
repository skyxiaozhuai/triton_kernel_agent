#!/usr/bin/env python3
"""全算子冒烟入口：对注册表里的每个 op 跑 golden vs reference 对齐。

用法（在项目根目录）：
    python scripts/smoke_test.py            # 跑全部算子
    python scripts/smoke_test.py vector_add # 只跑指定算子

需要 CUDA。先跑 python scripts/check_env.py 确认环境。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，冒烟需要 GPU。先跑: python scripts/check_env.py")
        return 2

    from benchmarks.runner import verify_all, verify_op

    names = sys.argv[1:]
    if names:
        bad = []
        for name in names:
            ok, *_ = verify_op(name)
            if not ok:
                bad.append(name)
    else:
        results = verify_all()
        bad = [k for k, v in results.items() if not v]
    print("全部通过 ✔" if not bad else f"失败算子: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
