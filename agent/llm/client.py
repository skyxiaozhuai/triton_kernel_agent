"""DeepSeek / OpenAI 兼容 LLM client。

- 从项目根 .env 读取 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
  （已存在的同名环境变量优先，不覆盖）
- 只用标准库 urllib 实现 chat completion，零额外依赖
- chat() 返回 (文本, token用量)；test_connection() 验证 key/base/model 是否可用，
  全程不打印 key 本体
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def load_dotenv(path: str = ENV_PATH) -> None:
    """极简 .env 加载：把 KEY=VALUE 注入 os.environ（已有值不覆盖）。"""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


class LLMClient:
    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url
                         or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        # thinking 开关：None=不注入(用服务端默认)；env DEEPSEEK_THINKING=off/0 → 关
        self.thinking: bool | None = None
        _env_th = os.environ.get("DEEPSEEK_THINKING")
        if _env_th is not None:
            self.thinking = _env_th.strip().lower() not in (
                "0", "off", "false", "no", "disable", "disabled")

    @property
    def key_status(self) -> str:
        if not self.api_key:
            return "未配置"
        return f"已配置 (len={len(self.api_key)})"

    def chat(self, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 2048) -> tuple[str, dict]:
        """发一次 chat completion，返回 (内容文本, usage)。"""
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置：请在项目根 .env 填写真实 key。")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.thinking is not None:
            # 显式关/开思考：官方 extra_body={"thinking": {"type": "enabled|disabled"}}
            payload["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {err[:600]}") from e
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    def test_connection(self) -> bool:
        """极小请求验证 key/base/model 是否可用（不暴露 key）。"""
        print(f"[llm] base_url = {self.base_url}")
        print(f"[llm] model    = {self.model}")
        print(f"[llm] api_key  = {self.key_status}")
        try:
            text, usage = self.chat(
                [{"role": "user", "content": "Reply with exactly: pong"}],
                temperature=0.0, max_tokens=16)
        except Exception as e:  # noqa: BLE001
            print(f"[llm] ✗ 连接失败: {type(e).__name__}: {e}")
            return False
        print(f"[llm] ✓ 连接成功，回复: {text!r}")
        print(f"[llm]   usage = {usage}")
        return True


if __name__ == "__main__":
    raise SystemExit(0 if LLMClient().test_connection() else 1)
