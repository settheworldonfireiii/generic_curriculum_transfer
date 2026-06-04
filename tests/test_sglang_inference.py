from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gct.config.load import load_config
from gct.runtime.inference import SglangClient


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"choices": [{"text": " 42 "}]}).encode("utf-8")


def test_sglang_client_posts_openai_compatible_completion(monkeypatch: Any) -> None:
    captured = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    config = load_config(
        Path("configs/competitive_math.yaml"),
        {
            "inference": {
                "backend": "sglang",
                "sglang_base_url": "http://localhost:30000",
                "sglang_api_key": "secret",
                "sglang_model": "served-llama",
                "sglang_timeout_s": 9.0,
            }
        },
    )

    output = SglangClient(config).generate("Problem?", seed=123)

    assert output == "42"
    assert captured["url"] == "http://localhost:30000/v1/completions"
    assert captured["body"]["model"] == "served-llama"
    assert captured["body"]["prompt"] == "Problem?"
    assert captured["body"]["seed"] == 123
    assert captured["auth"] == "Bearer secret"
    assert captured["timeout"] == 9.0
