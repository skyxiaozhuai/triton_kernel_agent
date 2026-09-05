"""Agent 角色模块（D6 抽出为清晰角色，D3-D5 先在主循环里实现）。

- planner : 读算子 OP_META，规划网格/block/算法思路（可轻量，甚至并入 coder）
- coder   : 调 LLM 生成 Triton kernel 源码
- critic  : 正确性 critic + 性能 critic（决定"是否达标可停"）
"""
