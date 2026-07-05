#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat client for the (on-site released) teacher API.

Reads configuration from environment variables so no secret is committed:
  AGENTDOG_API_BASE   e.g. https://api.example.com/v1   (required)
  AGENTDOG_API_KEY    bearer token                        (required)
  AGENTDOG_API_MODEL  model name                          (default: gpt-4o-mini)

Uses only the standard library (urllib) to avoid adding dependencies. If the
on-site API is not OpenAI-compatible, this is the single file to adapt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class TeacherAPI:
    def __init__(self, base: str | None = None, key: str | None = None,
                 model: str | None = None, timeout: int = 60) -> None:
        self.base = (base or os.environ.get("AGENTDOG_API_BASE", "")).rstrip("/")
        self.key = key or os.environ.get("AGENTDOG_API_KEY", "")
        self.model = model or os.environ.get("AGENTDOG_API_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        if not self.base or not self.key:
            raise RuntimeError(
                "Set AGENTDOG_API_BASE and AGENTDOG_API_KEY before running API steps."
            )

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 256, retries: int = 3) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"API call failed after {retries} retries: {last_err}")


def self_test() -> None:
    api = TeacherAPI()
    print(api.chat([{"role": "user", "content": "Reply with the single word: ok"}]))


if __name__ == "__main__":
    self_test()
