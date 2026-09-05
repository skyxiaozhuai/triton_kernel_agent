"""Agent 工具层（D3-D5 逐步实现）。

- executor.py     : 独立子进程沙箱执行生成代码（超时隔离），主进程零崩溃
- error_parser.py : 编译/运行/CUDA/数值错误分类 -> 结构化反馈文本
- benchmark.py    : do_bench 计时 + 对比 torch.compile / eager
"""
