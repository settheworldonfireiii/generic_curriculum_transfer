from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from gct.config.schema import ExperimentConfig
from gct.data.torch_dataset import build_dataloader
from gct.telemetry.metrics import AsyncMetrics, NullMetrics
from gct.telemetry.wandb import WandbRun
from gct.training.collate import build_generation_prompts
from gct.utils.hashing import stable_seed
from gct.utils.jsonl import append_jsonl, read_jsonl


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.device  # type: ignore[return-value]
    except AttributeError:
        return next(model.parameters()).device


def _completed(path: Path) -> set[tuple[str, int]]:
    return {(row["task_id"], int(row["sample_index"])) for row in read_jsonl(path)}


def run_ablation(config: ExperimentConfig, tasks_path: Path, raw_out: Path, samples_per_task: int = 3) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers before running ablations.") from exc

    metrics = (
        AsyncMetrics(config.runtime.output_dir / "metrics" / "ablation_metrics.jsonl", config.telemetry.flush_interval_s)
        if config.telemetry.enabled
        else NullMetrics()
    )
    done = _completed(raw_out) if config.runtime.resume else set()
    dataloader = build_dataloader(
        tasks_path,
        batch_size=max(1, config.runtime.generation_batch_size),
        num_workers=config.runtime.num_workers,
        prefetch_factor=config.runtime.prefetch_factor,
        shuffle=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=_torch_dtype(config.model.dtype),
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()

    with WandbRun(config, "ablation") as wandb_run:
        try:
            for batch in tqdm(dataloader, desc="ablation"):
                for sample_index in range(samples_per_task):
                    active_indices = [
                        idx
                        for idx, task_id in enumerate(batch["task_id"])
                        if (task_id, sample_index) not in done
                    ]
                    if not active_indices:
                        continue
                    prompts = [build_generation_prompts(batch)[idx] for idx in active_indices]
                    task_ids = [batch["task_id"][idx] for idx in active_indices]
                    started = time.perf_counter()
                    seed = stable_seed(config.runtime.seed, sample_index, *task_ids)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    encoded = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=config.model.max_input_tokens,
                    ).to(_model_device(model))
                    with torch.inference_mode():
                        output_ids = model.generate(
                            **encoded,
                            do_sample=config.model.temperature > 0,
                            temperature=config.model.temperature if config.model.temperature > 0 else None,
                            top_p=config.model.top_p if config.model.temperature > 0 else None,
                            max_new_tokens=config.model.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    elapsed = time.perf_counter() - started
                    input_width = encoded["input_ids"].shape[1]
                    rows: list[dict[str, Any]] = []
                    for offset, task_id in enumerate(task_ids):
                        new_tokens = output_ids[offset, input_width:]
                        rows.append(
                            {
                                "task_id": task_id,
                                "sample_index": sample_index,
                                "raw_output": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                                "latency_s": elapsed / max(1, len(task_ids)),
                                "batch_size": len(task_ids),
                            }
                        )
                    append_jsonl(raw_out, rows)
                    metrics.observe("ablation.batch_latency_s", elapsed, {"batch_size": len(task_ids)})
                    metrics.incr("ablation.generations", len(task_ids))
                    wandb_run.log(
                        {
                            "ablation/batch_latency_s": elapsed,
                            "ablation/batch_size": len(task_ids),
                        }
                    )
        finally:
            metrics.close()
