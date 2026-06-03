from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from gct.config.schema import ExperimentConfig
from gct.utils.jsonl import append_jsonl, read_jsonl


def completed_sae(path: Path) -> set[tuple[str, int]]:
    return {(row["task_id"], int(row["layer"])) for row in read_jsonl(path)}


def extract_sae_features(
    config: ExperimentConfig,
    tasks_path: Path,
    out_path: Path,
    sae_repo: str,
    layers: list[int],
    top_k: int = 256,
) -> Path:
    try:
        import torch
        from sparsify import Sae
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SAE extraction requires torch, transformers, and eai-sparsify. "
            "Install the GPU environment first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=_torch_dtype(config.model.dtype, torch),
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()
    device = _model_device(model)
    tasks = list(read_jsonl(tasks_path))
    done = completed_sae(out_path) if config.runtime.resume else set()

    for layer in layers:
        hookpoint = f"layers.{layer}"
        print(f"Loading SAE {sae_repo} hookpoint={hookpoint}")
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint).to(device)
        sae.eval()
        for row in tqdm(tasks, desc=f"SAE layer {layer}"):
            if (row["task_id"], layer) in done:
                continue
            prompt = _feature_prompt(row["prompt"])
            hidden_states, attention_mask = _encode_hidden_states(
                model,
                tokenizer,
                prompt,
                config.model.max_input_tokens,
            )
            feature_ids, values, mean_l0, n_tokens = _top_pooled_features(
                sae,
                hidden_states[layer + 1],
                attention_mask,
                top_k,
                torch,
            )
            append_jsonl(
                out_path,
                [
                    {
                        "task_id": row["task_id"],
                        "layer": layer,
                        "hookpoint": hookpoint,
                        "features": {
                            f"L{layer}:{fid}": float(value)
                            for fid, value in zip(feature_ids, values, strict=True)
                        },
                        "mean_l0": float(mean_l0),
                        "n_tokens": int(n_tokens),
                        "metadata": row.get("metadata", {}),
                    }
                ],
            )
    return out_path


def _feature_prompt(prompt: str) -> str:
    return (
        "Solve the following problem. Show concise reasoning, and put the final answer in boxed form.\n\n"
        f"Problem:\n{prompt}\n\nSolution:"
    )


def _torch_dtype(name: str, torch: Any) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _encode_hidden_states(model: Any, tokenizer: Any, text: str, max_length: int) -> tuple[Any, Any]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(_model_device(model))
    with _inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["attention_mask"][0].bool()


def _top_pooled_features(
    sae: Any,
    hidden_state: Any,
    attention_mask: Any,
    top_k: int,
    torch: Any,
) -> tuple[list[int], list[float], float, int]:
    token_acts = hidden_state[0, attention_mask]
    encoded = sae.encode(token_acts)
    top_acts = encoded.top_acts.float()
    top_indices = encoded.top_indices.long()
    pooled = torch.zeros(sae.num_latents, dtype=torch.float32, device=top_acts.device)
    pooled.scatter_add_(0, top_indices.reshape(-1), top_acts.reshape(-1))
    pooled /= max(token_acts.shape[0], 1)
    mean_l0 = float((top_acts > 0).float().sum(dim=1).mean().item())
    k = min(top_k, pooled.numel())
    values, indices = torch.topk(pooled, k=k)
    keep = values > 0
    return (
        indices[keep].detach().cpu().tolist(),
        values[keep].detach().cpu().tolist(),
        mean_l0,
        int(attention_mask.sum().item()),
    )


def _inference_mode() -> Any:
    import torch

    return torch.inference_mode()

