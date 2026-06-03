from __future__ import annotations

from typing import Any


def build_supervised_texts(batch: dict[str, Any]) -> list[str]:
    texts = []
    for prompt, answer in zip(batch["prompt"], batch["answer"], strict=True):
        texts.append(f"Problem:\n{prompt}\n\nAnswer:\n{answer}")
    return texts


def build_generation_prompts(batch: dict[str, Any]) -> list[str]:
    return [f"Problem:\n{prompt}\n\nGive the final answer." for prompt in batch["prompt"]]

