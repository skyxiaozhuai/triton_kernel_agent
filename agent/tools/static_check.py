"""静态闸门：进沙箱前对生成代码做结构 + 反作弊检查（零成本，不烧 GPU）。

借鉴 PyTorch 官方 KernelAgent（/home/claude/agent_project/kernel_agent）的两层设计，
并做了升级：官方用"strip 注释后正则扫描"，我们改用 **AST 精确分析**——
  1) 能区分 tl.*（Triton，合法）与 torch.* / tensor.*（可能作弊）；
  2) 能精确定位 @triton.jit kernel 函数体 vs launch，分而治之；
  3) 结构上要求 launch **必须真实调用某个 @triton.jit kernel**——
     官方没有这一条，它能从结构上杜绝"launch 里用 torch 直接算出答案"。

三层闸门（任一层失败即打回，不进沙箱、不计成功轮次）：
  A. 结构：ast.parse 语法、顶层必须含 def launch、至少一个 @triton.jit kernel、
     launch 必须调用 kernel、launch 不得直接 return 输入张量间的运算表达式。
  B. 反作弊（kernel 体）：kernel 内禁止出现任何 torch 引用（只准用 triton/tl）。
  C. 反作弊（全代码）：禁止把计算委托给 torch（torch.<op> / tensor.<op> / @ 矩阵乘）、
     禁止反射/动态执行/读文件等越权手段、禁止 import 危险模块。

设计取舍（面试可讲）：静态闸门是"第一道防线"，判卷（可信 harness + golden）仍是
最终裁判——两道都过了才算 pass；静态拦不住的高级作弊会被数值判卷兜底。
"""
from __future__ import annotations

import ast

# torch 计算性 API：命中即视为"把任务计算委托给 torch"（无论 torch.xxx 还是 tensor.xxx）
_CALC_ATTRS = {
    # 矩阵/张量积
    "matmul", "mm", "bmm", "einsum", "tensordot", "dot", "mv", "addmm", "baddbmm",
    # 卷积/池化
    "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "max_pool1d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
    # 激活/归一化（直接调用即等于外包 softmax/relu/... 类任务）
    "softmax", "log_softmax", "softmin", "relu", "relu_", "sigmoid", "tanh", "gelu",
    "silu", "hardtanh", "softplus", "softsign", "leaky_relu",
    "layer_norm", "batch_norm", "group_norm", "instance_norm", "normalize",
    # 归约（外包 sum/mean/... 类任务）
    "sum", "mean", "max", "min", "var", "std", "prod", "cumsum", "cumprod",
    "norm", "argmax", "argmin", "sort", "topk", "median", "mode", "all", "any",
    # 其它整算子
    "flip", "roll", "diag", "trace", "fft", "istft", "lstsq", "solve", "cholesky",
    "qr", "svd", "eig", "eigvalsh", "pinverse", "cross", "histc", "bincount",
    "interpolate", "grid_sample", "scatter_add", "index_add", "index_select",
}
# torch.<op>(...) 顶层模块 API 直接用（与 _CALC_ATTRS 交集即可，这里列出 torch 侧同义入口）
_CALC_TORCH_ATTRS = _CALC_ATTRS | {
    # 只有 torch 侧、无法/极少作为方法出现的计算入口
    "add", "sub", "mul", "div", "pow", "sqrt", "rsqrt", "exp", "log", "abs",
    "clamp", "where", "cat", "stack", "split", "chunk", "gather", "outer",
}
# 反射 / 越权 Name 调用
_REFLECT_NAMES = {"eval", "exec", "globals", "locals", "vars", "__import__", "compile"}
# 危险 import（生成代码不该读文件/起进程/反序列化）
_BLOCKED_IMPORTS = {
    "inspect", "os", "subprocess", "socket", "ctypes", "importlib",
    "pickle", "shutil", "pathlib", "requests", "urllib", "http", "sqlite3", "json",
}

# 每个 op 任务对应的"被禁止的 torch 捷径"，用于给 LLM 更精确的反馈（可空；命中按通用文案）
_OP_ATTR_HINT = {
    "vector_add": ["add", "add_", "__add__"],
    "relu": ["relu", "relu_", "clamp"],
    "softmax": ["softmax", "log_softmax", "softmax_"],
    "sum_1d": ["sum", "mean"],
    "matmul": ["matmul", "mm", "bmm", "einsum", "tensordot"],
}


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _kernel_body_has_import(func: ast.FunctionDef) -> bool:
    """kernel 函数体内出现 import —— triton.jit 按模块全局查名字，函数内 import 必 NameError。"""
    for node in ast.walk(func):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
    return False


def _torch_ref_in_kernel(func: ast.FunctionDef) -> bool:
    """kernel 函数体内出现任何 torch 名字（kernel 只允许 triton/tl）。"""
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == "torch":
            return True
    return False


def _iter_import_names(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield n.module.split(".")[0]


def _check_calc_or_reflect(tree: ast.AST) -> str | None:
    """全代码（含 launch）扫描：torch 计算 / tensor 方法计算 / @ 矩阵乘 / 反射越权。"""
    for node in ast.walk(tree):
        # @ 运算符 = torch 侧矩阵乘，Triton 项目内无合法用途
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return "使用了 @(矩阵乘)运算符：请用 tl.dot 在 kernel 里实现，禁止把矩阵乘交给 torch"
        # torch.<calc>(...) 或 <obj>.<calc>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value
            if _is_name(value, "torch"):
                if attr in _CALC_TORCH_ATTRS:
                    return f"调用了 torch.{attr}(...)：这是把任务计算交给 torch，请在 Triton kernel 里用 tl 实现"
            elif not _is_name(value, "tl"):
                if attr in _CALC_ATTRS:
                    return (f"对张量调用了 .{attr}(...)：疑似用 torch 直接算答案，"
                            f"请改成 Triton kernel 里的 tl 实现")
        # 反射 / 动态执行
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _REFLECT_NAMES:
                return f"禁止调用 {node.func.id}(...)（反射/动态执行，可窃取测试数据或绕过判卷）"
        if isinstance(node, ast.Attribute):
            if node.attr in ("f_locals", "f_globals", "__globals__"):
                return f"禁止访问帧对象 .{node.attr}（可窃取 harness 里的 golden/测试数据）"
            if _is_name(node.value, "sys") and node.attr == "_getframe":
                return "禁止 sys._getframe（帧注入，可窃取测试局部变量）"
            if _is_name(node.value, "inspect"):
                return "禁止使用 inspect.*（反射查看/调用栈）"
    return None


def _launch_return_rule(func: ast.FunctionDef, kernel_names: set[str]) -> str | None:
    """launch 规则：必须调用某 kernel；不得直接 return 输入张量的运算表达式。

    反作弊：vector_add 作弊形态 `return x1 + x2` / matmul 作弊 `return a @ b` 等，
    即"绕过 kernel 直接用 torch 把输入算成答案"。@ 已被全局禁，这里再补 BinOp(+,-,*,/…)。
    """
    calls_kernel = False
    for node in ast.walk(func):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in kernel_names):
            calls_kernel = True
    if not calls_kernel:
        return ("launch 没有调用任何 @triton.jit kernel（应为 kernel[grid](...)）。"
                "禁止用 torch 直接算出答案返回，必须在 kernel 里计算、写输出 buffer")
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.BinOp):
                return ("launch 直接 return 了张量运算表达式（如 x1 + x2 / a * b）："
                        "这是用 torch 直接算答案。请在 kernel 里计算，再 return kernel 写好的 buffer")
    return None


def _format_op_hint(op_name: str | None, reason: str) -> str:
    if not op_name:
        return reason
    attrs = _OP_ATTR_HINT.get(op_name)
    if attrs and "torch." in reason:
        return reason + f"（本算子 {op_name} 尤其禁止 torch.{attrs[0]} 等捷径）"
    return reason


def check_generated_code(code: str, op_name: str | None = None) -> dict:
    """对生成代码做三层静态闸门。

    返回 {"ok": bool, "category": "pass"|"syntax"|"structure"|"cheat", "reason": str}。
    category/reason 用于直接生成给 LLM 的反馈，并决定是否进沙箱。
    """
    if not code or not code.strip():
        return {"ok": False, "category": "structure",
                "reason": "代码为空，无法检查。"}

    # —— A1. 语法（AST）——
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        ln = f"line {exc.lineno}" if exc.lineno else "unknown"
        return {"ok": False, "category": "syntax",
                "reason": f"Python 语法错误（{ln}: {exc.msg}）。请先修正语法。"}

    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    # —— A2. 结构：顶层必须有 @triton.jit kernel + def launch ——
    kernel_names: set[str] = set()
    for f in funcs:
        for dec in f.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec  # 兼容 @triton.jit 与 @triton.jit(...)
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                kernel_names.add(f.name)
    launches = [f.name for f in funcs if f.name == "launch"]
    if not launches:
        return {"ok": False, "category": "structure",
                "reason": "缺少顶层 def launch 函数（负责 grid/调用 kernel/返回输出）。"}
    if not kernel_names:
        return {"ok": False, "category": "structure",
                "reason": "缺少 @triton.jit kernel：请提供真正的 Triton kernel 实现，"
                          "而不是只写 launch 用 torch 算答案。"}

    # —— import 黑名单（模块级，危险/外包模块）——
    for imp in _iter_import_names(tree):
        if imp in _BLOCKED_IMPORTS:
            return {"ok": False, "category": "cheat",
                    "reason": f"禁止 import {imp}（越权读文件/起进程/反射，生成代码不需要它）。"}

    # —— B. kernel 体：禁止 torch / import ——
    for f in funcs:
        if f.name in kernel_names:
            if _kernel_body_has_import(f):
                return {"ok": False, "category": "structure",
                        "reason": "kernel 函数体内出现了 import（triton.jit 会因名字不在全局作用域而 "
                                  "NameError）。请把所有 import 移到模块顶层。"}
            if _torch_ref_in_kernel(f):
                return {"ok": False, "category": "cheat",
                        "reason": "kernel 函数体内出现了 torch：kernel 只允许用 triton/tl 原语"
                                  "（如 tl.load/store/dot/exp），计算必须由 Triton 完成。"}

    # —— C. 全代码：torch 计算委托 / @ / 反射 ——
    for f in funcs:
        reason = _check_calc_or_reflect(f)
        if reason:
            return {"ok": False, "category": "cheat",
                    "reason": _format_op_hint(op_name, reason)}

    # —— A3. 结构：launch 必须调用 kernel + 不得直接 return 输入运算 ——
    for f in funcs:
        if f.name == "launch":
            reason = _launch_return_rule(f, kernel_names)
            if reason:
                return {"ok": False, "category": "structure", "reason": reason}

    return {"ok": True, "category": "pass", "reason": ""}
