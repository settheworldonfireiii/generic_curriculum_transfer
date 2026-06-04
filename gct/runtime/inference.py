from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gct.config.schema import ExperimentConfig


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw: dict[str, Any]


class SglangClient:
    """Minimal OpenAI-compatible completions client for SGLang servers."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.base_url = _normalize_base_url(config.inference.sglang_base_url)
        self.model = config.inference.sglang_model or config.model.name
        self.api_key = config.inference.sglang_api_key
        self.timeout_s = config.inference.sglang_timeout_s
        self.max_concurrency = max(1, config.inference.sglang_max_concurrency)

    def generate(self, prompt: str, seed: int | None = None) -> str:
        return self.generate_one(prompt, seed).text

    def generate_one(self, prompt: str, seed: int | None = None) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.config.model.max_new_tokens,
            "temperature": self.config.model.temperature,
            "top_p": self.config.model.top_p,
            "n": 1,
        }
        if seed is not None:
            payload["seed"] = seed
        response = self._post_json("/completions", payload)
        return GenerationResult(text=_extract_completion_text(response), raw=response)

    def generate_many(self, prompts: list[str], seeds: list[int | None] | None = None) -> list[str]:
        seeds = seeds or [None] * len(prompts)
        if len(seeds) != len(prompts):
            raise ValueError("seeds must have the same length as prompts")
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(prompts)))) as executor:
            futures = [executor.submit(self.generate, prompt, seed) for prompt, seed in zip(prompts, seeds, strict=True)]
            return [future.result() for future in futures]

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"SGLang request failed for {self.base_url + endpoint}: {exc}") from exc


def use_sglang(config: ExperimentConfig) -> bool:
    return config.inference.backend.lower() == "sglang"


def _normalize_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return stripped
    return stripped + "/v1"


def _extract_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return str(choice["text"]).strip()
    message = choice.get("message")
    if isinstance(message, dict) and "content" in message:
        return str(message["content"]).strip()
    return ""
