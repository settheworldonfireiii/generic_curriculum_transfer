from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from gct.config.schema import ExperimentConfig
from gct.runtime.inference import SglangClient, use_sglang
from gct.sweep.grading import answers_match, extract_final_answer
from gct.sweep.summary import summarize_sweep
from gct.telemetry.metrics import AsyncMetrics, NullMetrics
from gct.telemetry.wandb import WandbRun
from gct.utils.hashing import stable_seed
from gct.utils.jsonl import append_jsonl, read_jsonl


def completed_sweep(path: Path) -> set[tuple[str, int]]:
    return {(row["task_id"], int(row["sample_index"])) for row in read_jsonl(path)}


def run_sweep(
    config: ExperimentConfig,
    tasks_path: Path,
    raw_out: Path,
    summary_out: Path,
    solved_ids_out: Path,
    samples_per_task: int = 3,
) -> Path:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    if not use_sglang(config):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Sweep requires torch and transformers for the hf inference backend.") from exc

    tasks = list(read_jsonl(tasks_path))
    done = completed_sweep(raw_out) if config.runtime.resume else set()
    metrics = (
        AsyncMetrics(config.runtime.output_dir / "metrics" / "sweep_metrics.jsonl", config.telemetry.flush_interval_s)
        if config.telemetry.enabled
        else NullMetrics()
    )

    sglang_client = SglangClient(config) if use_sglang(config) else None
    tokenizer = None
    model = None
    if sglang_client is None:
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

    step = 0
    with WandbRun(config, "sweep") as wandb_run:
        try:
            for row in tqdm(tasks, desc="sweep"):
                expected = extract_final_answer(row.get("answer")) or row.get("answer")
                prompt = build_problem_prompt(row["prompt"])
                for sample_index in range(samples_per_task):
                    if (row["task_id"], sample_index) in done:
                        continue
                    seed = stable_seed(config.runtime.seed, row["task_id"], sample_index)
                    started = time.perf_counter()
                    output = (
                        sglang_client.generate(prompt, seed)
                        if sglang_client is not None
                        else _generate_once(model, tokenizer, prompt, config, seed, torch)
                    )
                    elapsed = time.perf_counter() - started
                    predicted = extract_final_answer(output)
                    exact = answers_match(predicted, expected)
                    append_jsonl(
                        raw_out,
                        [
                            {
                                "task_id": row["task_id"],
                                "sample_index": sample_index,
                                "raw_output": output,
                                "predicted": predicted,
                                "expected": expected,
                                "exact": exact,
                                "latency_s": elapsed,
                                "metadata": row.get("metadata", {}),
                            }
                        ],
                    )
                    metrics.observe("sweep.latency_s", elapsed)
                    metrics.incr("sweep.generations")
                    metrics.incr("sweep.successes", 1.0 if exact else 0.0)
                    if _should_log_wandb(config.telemetry.wandb_log_interval, step):
                        wandb_run.log(
                            {
                                "sweep/latency_s": elapsed,
                                "sweep/exact": int(exact),
                                "sweep/generations": step + 1,
                            },
                            step=step,
                        )
                    step += 1
        finally:
            metrics.close()

        summary = summarize_sweep(raw_out, summary_out, solved_ids_out)
        total = len(summary)
        solved = sum(1 for row in summary.values() if row["solved"])
        wandb_run.log(
            {
                "sweep/solved_tasks": solved,
                "sweep/total_tasks": total,
                "sweep/solved_rate": solved / total if total else 0.0,
            },
            step=step,
        )
    return summary_out


def build_problem_prompt(prompt: str) -> str:
    return (
        "Solve the following problem. Show concise reasoning, and put the final answer in boxed form.\n\n"
        f"Problem:\n{prompt}\n\nSolution:"
    )


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


def _should_log_wandb(interval: int, step: int) -> bool:
    return interval > 0 and step % interval == 0
