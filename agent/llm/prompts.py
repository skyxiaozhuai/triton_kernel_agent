"""Prompt 模板与代码抽取。

设计要点：
  - 判定逻辑(harness)与生成任务分离：LLM 只负责 kernel + launch，无权自评。
  - system 固定给"Triton 专家"身份与硬性契约；
  - user 携带算子规格(OP_META + launch_sig)；
  - 失败重试时追加 一轮 assistant(上一版代码) + 一轮 user(结构化反馈) —— Reflexion。
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert Triton kernel writer. Given an operator specification, write ONE \
complete, correct Python module that implements it with Triton.

HARD CONTRACT (violating any of these fails the task):
1. Output ONLY one python code block. No explanation, no extra text.
   (A ```python ... ``` markdown code fence is allowed.)
2. The code must start with these imports:
   import torch
   import triton
   import triton.language as tl
3. Implement the computation with @triton.jit kernel(s). Do NOT compute the result \
with torch ops (torch.empty is only for allocating the output).
4. Define a launcher with EXACTLY the signature given by the spec (parameter names \
must match the spec's input tensor names and scalar names):
   def launch(<input tensors...>, <scalar ints...>) -> torch.Tensor:
       - allocate the output with torch.empty(...) on the same device
       - choose a grid and BLOCK size, then call your kernel(s)
       - return the output tensor
   Inputs arrive as CUDA tensors; shape scalars (N/M/K...) are passed as plain ints.
   Do not print, do not verify, do not write files.
5. Numerical requirements:
   - Use masking (offs < N, offs_m < M, ...) whenever a dimension may not divide
     evenly by BLOCK.
   - For reductions / softmax, subtract the max first for numerical stability.
   - Accumulate sums / dot in fp32.
   - Inputs and output may be float32 OR float16: allocate the output with the SAME
     dtype as the input (e.g. torch.empty_like(input)); for fp16, compute reductions /
     softmax / dot in fp32 internally, then cast the result back to fp16.
   - Prefer power-of-two BLOCK sizes (1024, 128, 64, ...).
6. Target GPU: see the "Target GPU" line at the end of the spec. If you use tl.dot
   with fp32 inputs, set input_precision accordingly (tf32 on sm_80+, otherwise ieee).
   Keep kernels simple and portable.
7. Your final "content" MUST be the complete, runnable code block containing BOTH the @triton.jit kernel(s) AND the launch function. NEVER reply with an empty message or only reasoning; if the output would be long, keep comments short — the completeness of launch() matters more than verbosity.

If anything is ambiguous, make a reasonable minimal assumption and note it in a code \
comment (never outside the code block).
"""


def format_op_spec(meta: dict) -> str:
    """把 OP_META 转成给 LLM 的算子规格文本。"""
    lines = [f"- name: {meta.get('name')}",
             f"- category: {meta.get('category')}",
             f"- dtype: {meta.get('dtype')}",
             f"- signature: {meta.get('signature')}"]
    if meta.get("launch_sig"):
        lines.append(f"- launch_sig: {meta['launch_sig']}")
    if meta.get("description"):
        lines.append(f"- description: {meta['description']}")
    if meta.get("notes"):
        lines.append(f"- notes: {meta['notes']}")
    return "Operator to implement:\n" + "\n".join(lines)


def gpu_context_note() -> str:
    """根据当前 GPU 生成一行硬件提示（尤其 fp32 tl.dot 的 input_precision）。"""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            prec = "tf32" if cap >= (8, 0) else "ieee"
            why = ("fast (tf32 tensor core)" if cap >= (8, 0)
                   else "exact fp32; tf32 not supported on this GPU")
            return (f"- Target GPU: {name} (sm_{cap[0]}{cap[1]}). "
                    f"For fp32 tl.dot use input_precision=\"{prec}\" ({why}).")
    except Exception:  # noqa: BLE001
        pass
    return "- Target GPU: unknown. For fp32 tl.dot, pass an explicit input_precision."


def build_initial_messages(op_meta: dict) -> list[dict]:
    spec = format_op_spec(op_meta) + "\n" + gpu_context_note()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spec},
    ]


def feedback_user_message(feedback_text: str) -> str:
    return (
        "Your previous attempt FAILED. Fix the code according to the feedback below. "
        "Output ONLY the corrected complete python code block.\n\n"
        f"--- feedback ---\n{feedback_text}"
    )


def perf_feedback_user_message(launch_ms, eager_ms, speedup, min_speedup) -> str:
    """性能未达标时的优化反馈（正确性已通过，只要求提速）。"""
    return (
        "Your kernel is CORRECT but too slow (performance critic failed).\n"
        f"- launch_ms={launch_ms}, eager_ms={eager_ms}, "
        f"speedup_vs_eager={speedup}x (required >= {min_speedup}x)\n"
        "- Likely causes: tiny grid / too much work per program, poor BLOCK size, "
        "uncoalesced memory access, avoidable overhead. Try a better BLOCK/grid "
        "layout or num_warps. Do NOT break correctness.\n"
        "Output ONLY the corrected complete python code block."
    )


def extract_python_code(text: str) -> str:
    """从模型回复中抽取 python 代码（兼容 ```python 围栏 / 裸代码）。"""
    s = text.strip()
    if "```" not in s:
        return s  # 无围栏，直接当作整段代码
    parts = s.split("```")
    candidates = []
    for i in range(1, len(parts), 2):  # 奇数索引是围栏内容
        content = parts[i]
        if content.startswith("python"):
            content = content[len("python"):].lstrip("\n")
        elif content.startswith("py"):
            content = content[len("py"):].lstrip("\n")
        candidates.append(content.strip("\n").strip())
    if not candidates:
        return s
    # 优先选同时含 @triton.jit 与 def launch 的块，否则选最长
    def score(c: str) -> tuple:
        return ("def launch" in c and "@triton.jit" in c,
                "def launch" in c,
                len(c))
    candidates.sort(key=score, reverse=True)
    return candidates[0]


def check_code_valid(code: str) -> tuple[bool, str]:
    """快速校验生成代码是否值得进沙箱。返回 (是否有效, 无效原因)。"""
    if not code.strip():
        return False, ("❌ 你的输出为空(content 里没有代码)。请重新输出完整的 "
                       "python 代码块，必须包含 @triton.jit kernel 与 def launch 函数。")
    if "def launch" not in code:
        return False, ("❌ 输出缺少 def launch 函数(可能代码不完整或被截断)。"
                       "请重新输出完整代码，确保包含 launch 的定义。")
    return True, ""
