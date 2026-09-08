"""KernelBench 适配层：把官方 KernelBench 题目接入我们的 agent 闭环。

官方题目 = 一个 .py：class Model(nn.Module).forward + get_inputs() + get_init_inputs()。
本适配层把它包装成 agent 能消费的规格 + 判卷句柄，判卷 golden = eager forward
（可信、与官方语义一致）。真跑需要 GPU + 显存（多数 L1 默认 shape 是 A100 级），
本机只做加载/规格/CPU 预检，正式跑在服务器（见 scripts/run_kernelbench.py）。

根目录默认 /home/claude/agent_project/KernelBench，可用环境变量 KERNELBENCH_ROOT 覆盖。
"""
