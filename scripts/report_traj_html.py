#!/usr/bin/env python3
"""轨迹 HTML 报告 —— 把一条 agent 轨迹渲染成自包含页面（面试 demo / 录屏 / 复盘）。

特性：零第三方依赖、纯标准库；单文件 HTML（内嵌 CSS），可离线打开/分享。
内容：顶部摘要(op/成败/轮数/token/耗时/RAG/失败回灌/缓存命中/末轮 err/perf)
      + 逐轮卡片(状态徽标颜色/耗时/token/err → 可展开的 feedback 与 kernel 代码)

用法：
    python scripts/report_traj_html.py results/traj_matmul_x.jsonl
    python scripts/report_traj_html.py --latest
    python scripts/report_traj_html.py --op matmul        # 该算子最新一条
    python scripts/report_traj_html.py <path> --out /tmp/r.html   # 指定输出
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 状态徽标配色
_BADGE = {
    "pass": ("#2da44e", "✔ pass"),
    "correctness": ("#d1242f", "✗ correctness"),
    "compile": ("#cf222e", "✗ compile"),
    "cuda": ("#bf8700", "✗ cuda"),
    "syntax": ("#bf8700", "✗ syntax"),
    "timeout": ("#bf8700", "✗ timeout"),
    "error": ("#d1242f", "✗ error"),
    "perf_slow": ("#9a6700", "⟳ perf_slow"),
    "interrupted": ("#57606a", "⏹ interrupted"),
}
_BADGE_STATIC = ("#8250df", "⛨ static")
_OK_COLOR = "#1a7f37"
_ERR_COLOR = "#cf222e"


def load_traj(path: str) -> tuple[dict, list[dict]]:
    summary, steps = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "summary" in obj:
                summary = obj["summary"]
            else:
                steps.append(obj)
    return summary, steps


def _esc(s: str) -> str:
    return html.escape(str(s))


def _badge(status: str) -> str:
    if status.startswith("static"):
        color, label = _BADGE_STATIC
    else:
        color, label = _BADGE.get(status, ("#57606a", status))
    return f'<span class="badge" style="background:{color}">{label}</span>'


def render(summary: dict, steps: list[dict], path: str) -> str:
    op = summary.get("op", "?")
    success = bool(summary.get("success"))
    tok = summary.get("total_tokens") or {}
    tok_sum = tok.get("prompt", 0) + tok.get("completion", 0)
    cards = []
    for st in steps:
        status = st.get("status", "?")
        rnd = st.get("round", 0)
        stok = (st.get("prompt_tokens", 0) + st.get("completion_tokens", 0))
        err = st.get("max_abs_err")
        err_s = f"{err:.2e}" if err is not None else "—"
        rows = (f"<div class='meta'>round <b>{rnd}</b> · "
                f"<code>{_esc(status)}</code> · {stok} tok · "
                f"{st.get('wall_s', 0):.1f}s · err {err_s}</div>")
        body = ""
        if st.get("feedback"):
            body += (f"<div class='feedback'><b>反馈</b>"
                     f"<pre>{_esc(st['feedback'])}</pre></div>")
        if st.get("code"):
            code = st["code"]
            lines = code.splitlines()
            preview = lines[:4]
            body += ("<details><summary>kernel 代码 "
                     f"({len(lines)} 行)</summary><pre class='code'>"
                     + _esc("\n".join(lines)) + "</pre></details>")
        cards.append(
            f"<div class='card'><div class='cardhead'>{_badge(status)} {rows}</div>"
            f"{body}</div>")

    err = summary.get("final_max_abs_err")
    err_s = f"{err:.2e}" if err is not None else "—"
    perf = summary.get("final_speedup_vs_eager")
    extras = []
    if summary.get("memory_used"):
        extras.append(f"RAG 参考: {_esc(summary['memory_used'])}")
    if summary.get("fix_refs_used"):
        extras.append(f"失败回灌: {_esc(summary['fix_refs_used'])}")
    if summary.get("cache_hits"):
        extras.append(f"digest 缓存命中: {summary['cache_hits']}")
    if summary.get("interrupted"):
        extras.append("被其它 seed 早停")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>轨迹 · {op}</title>
<style>
  body{{font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
        background:#0d1117;color:#e6edf3;margin:0;padding:24px;line-height:1.5}}
  .wrap{{max-width:900px;margin:0 auto}}
  h1{{font-size:20px;margin:0 0 4px}}
  .sub{{color:#8b949e;font-size:12px;margin-bottom:16px}}
  .summary{{background:#161b22;border:1px solid #30363d;border-radius:10px;
            padding:16px;margin-bottom:20px}}
  .badge{{color:#fff;border-radius:6px;padding:2px 10px;font-size:12px;
         font-weight:600}}
  .stat{{display:inline-block;margin:6px 18px 0 0}}
  .stat b{{font-size:16px}}
  .stat span{{color:#8b949e;font-size:11px;display:block}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;
         padding:12px 14px;margin-bottom:12px}}
  .cardhead{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
  .meta{{color:#8b949e;font-size:12px}}
  .meta code{{background:#21262d;padding:1px 5px;border-radius:4px}}
  .feedback{{margin-top:10px}}
  .feedback pre{{background:#21262d;border-radius:8px;padding:10px;
                white-space:pre-wrap;word-break:break-word;font-size:12px;
                max-height:220px;overflow:auto;margin:6px 0 0}}
  details{{margin-top:10px}}
  summary{{cursor:pointer;color:#58a6ff;font-size:13px}}
  pre.code{{background:#0d1117;border:1px solid #30363d;border-radius:8px;
           padding:10px;overflow:auto;font-size:11px;line-height:1.45}}
  .ok{{color:{_OK_COLOR}}} .err{{color:{_ERR_COLOR}}}
</style></head><body><div class="wrap">
  <h1>{_esc(op)} · 轨迹</h1>
  <div class="sub">{os.path.basename(path)}</div>
  <div class="summary">
    <div style="margin-bottom:8px">{'<span class="badge" style="background:#1a7f37">✔ 通过</span>' if success else '<span class="badge" style="background:#d1242f">✗ 未通过</span>'}
      <span class="badge" style="background:#57606a">{summary.get('final_status','?')}</span></div>
    <span class="stat"><b>{summary.get('rounds_used', 0)}</b><span>轮数</span></span>
    <span class="stat"><b>{tok_sum}</b><span>token(p+c)</span></span>
    <span class="stat"><b>{summary.get('wall_s', 0):.1f}s</b><span>耗时</span></span>
    <span class="stat"><b class="{'ok' if success else 'err'}">{err_s}</b><span>末轮 err</span></span>
    <span class="stat"><b>{perf if perf is not None else '—'}x</b><span>末轮 perf</span></span>
    <span class="stat"><b>{'  ·  '.join(extras) if extras else '—'}</b><span>附加</span></span>
  </div>
  {''.join(cards)}
  <p class="sub">状态图例：✔ pass(绿) / ✗ correctness·compile·cuda·syntax·timeout(红/橙)
    / ⛨ static 闸门(紫) / ⟳ perf_slow(黄)</p>
</div></body></html>"""
    return html_doc


def _find_latest(op: str | None = None) -> str:
    pat = os.path.join(RESULTS, "traj_*.jsonl")
    if op:
        pat = os.path.join(RESULTS, f"traj_{op}_*.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"没有匹配轨迹: {pat}")
    return files[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None, help="轨迹 jsonl；缺省用 --latest")
    ap.add_argument("--latest", action="store_true", help="最新一条轨迹")
    ap.add_argument("--op", default=None, help="与 --latest 组合：该算子最新一条")
    ap.add_argument("--out", default=None, help="输出 html 路径（默认 results/report_*.html）")
    args = ap.parse_args()

    path = args.path
    if path is None:
        if not args.latest and not args.op:
            ap.error("给一个轨迹路径，或加 --latest（可配 --op）")
        path = _find_latest(args.op)
    if not os.path.exists(path):
        raise SystemExit(f"轨迹不存在: {path}")

    summary, steps = load_traj(path)
    if not summary:
        raise SystemExit(f"{path} 里没有 summary（不是完整轨迹？）")
    doc = render(summary, steps, path)
    out = args.out or os.path.join(
        RESULTS, "report_" + os.path.basename(path).replace(".jsonl", ".html"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"✔ HTML 报告已生成: {out}")
    print(f"  ({os.path.getsize(out)} bytes, 单文件自包含)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
