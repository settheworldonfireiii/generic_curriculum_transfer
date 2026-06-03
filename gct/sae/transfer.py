from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gct.config.schema import ExperimentConfig
from gct.telemetry.metrics import AsyncMetrics, NullMetrics
from gct.utils.hashing import stable_seed
from gct.utils.jsonl import append_jsonl, read_jsonl


def load_top_neighbors(path: Path, rank: int = 1) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = [row for row in csv.DictReader(f) if int(row["neighbor_rank"]) == rank]
    rows.sort(key=lambda row: (-float(row["combined_similarity"]), row["target_task_id"]))
    return rows


def completed_transfer(path: Path) -> set[tuple[str, int]]:
    return {(row["target_task_id"], int(row["sample_index"])) for row in read_jsonl(path)}


def run_sae_transfer(
    config: ExperimentConfig,
    tasks_path: Path,
    neighbors_path: Path,
    anchor_solutions_path: Path,
    raw_out: Path,
    samples_per_target: int = 3,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("SAE transfer requires torch and transformers.") from exc

    tasks = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    anchor_solutions = _load_anchor_solutions(anchor_solutions_path)
    neighbors = load_top_neighbors(neighbors_path)
    done = completed_transfer(raw_out) if config.runtime.resume else set()
    metrics = (
        AsyncMetrics(config.runtime.output_dir / "metrics" / "sae_transfer_metrics.jsonl", config.telemetry.flush_interval_s)
        if config.telemetry.enabled
        else NullMetrics()
    )

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

    try:
        for row in tqdm(neighbors, desc="SAE transfer"):
            target_id = row["target_task_id"]
            anchor_id = row["anchor_task_id"]
            if target_id not in tasks or anchor_id not in tasks or anchor_id not in anchor_solutions:
                continue
            prompt = build_transfer_prompt(tasks[anchor_id]["prompt"], anchor_solutions[anchor_id], tasks[target_id]["prompt"])
            for sample_index in range(samples_per_target):
                if (target_id, sample_index) in done:
                    continue
                seed = stable_seed(config.runtime.seed, target_id, anchor_id, sample_index)
                started = time.perf_counter()
                output = _generate_once(model, tokenizer, prompt, config, seed, torch)
                elapsed = time.perf_counter() - started
                append_jsonl(
                    raw_out,
                    [
                        {
                            "target_task_id": target_id,
                            "anchor_task_id": anchor_id,
                            "sample_index": sample_index,
                            "raw_output": output,
                            "latency_s": elapsed,
                            "score_variant": row.get("score_variant", ""),
                            "layer_regime": row.get("layer_regime", ""),
                            "combined_similarity": row.get("combined_similarity", ""),
                        }
                    ],
                )
                metrics.observe("sae_transfer.latency_s", elapsed, {"score_variant": row.get("score_variant", "")})
                metrics.incr("sae_transfer.generations")
    finally:
        metrics.close()
    return raw_out


def build_transfer_prompt(anchor_problem: str, anchor_solution: str, target_problem: str) -> str:
    return (
        "A solved related problem is provided first. Use it as an analogy if useful, "
        "but solve the target problem from its own statement.\n\n"
        "Solved related problem:\n"
        f"{anchor_problem}\n\n"
        "Solved related solution:\n"
        f"{anchor_solution}\n\n"
        "Target problem:\n"
        f"{target_problem}\n\n"
        "Solve the target. Show concise reasoning, and put the final answer in boxed form.\n\n"
        "Target solution:"
    )


def _load_anchor_solutions(path: Path) -> dict[str, str]:
    solutions = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id") or row.get("target_task_id")
        if not task_id:
            continue
        if row.get("exact") is False:
            continue
        raw = row.get("raw_output") or row.get("solution") or row.get("answer")
        if raw and task_id not in solutions:
            solutions[task_id] = str(raw)
    return solutions


def _generate_once(model: Any, tokenizer: Any, prompt: str, config: ExperimentConfig, seed: int, torch: Any) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.model.max_input_tokens,
    ).to(_model_device(model))
    do_sample = config.model.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=config.model.temperature if do_sample else None,
            top_p=config.model.top_p if do_sample else None,
            max_new_tokens=config.model.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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

