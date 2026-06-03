from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from gct.config.schema import ExperimentConfig
from gct.data.torch_dataset import build_dataloader
from gct.telemetry.metrics import AsyncMetrics, NullMetrics
from gct.telemetry.wandb import WandbRun
from gct.training.collate import build_supervised_texts


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


def run_training_loop(config: ExperimentConfig, tasks_path: Path, max_steps: int | None = None) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers before running model training.") from exc

    output_dir = config.runtime.output_dir
    metrics = (
        AsyncMetrics(output_dir / "metrics" / "train_metrics.jsonl", config.telemetry.flush_interval_s)
        if config.telemetry.enabled
        else NullMetrics()
    )

    dataloader = build_dataloader(
        tasks_path,
        batch_size=config.runtime.batch_size,
        num_workers=config.runtime.num_workers,
        prefetch_factor=config.runtime.prefetch_factor,
        shuffle=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=_torch_dtype(config.model.dtype),
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    step = 0
    with WandbRun(config, "train") as wandb_run:
        try:
            for batch in tqdm(dataloader, desc="training"):
                started = time.perf_counter()
                texts = build_supervised_texts(batch)
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.model.max_input_tokens,
                ).to(_model_device(model))
                labels = encoded["input_ids"].clone()
                outputs = model(**encoded, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                elapsed = time.perf_counter() - started
                metrics.observe("train.step_latency_s", elapsed)
                metrics.incr("train.examples", len(batch["task_id"]))
                wandb_run.log({"train/loss": float(loss.detach().cpu()), "train/step_latency_s": elapsed}, step=step)
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
        finally:
            metrics.close()
