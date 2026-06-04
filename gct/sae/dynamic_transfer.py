from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from gct.config.schema import ExperimentConfig
from gct.runtime.inference import SglangClient, use_sglang
from gct.runtime.work_queue import SqliteWorkQueue, default_worker_id
from gct.sae.transfer import _load_anchor_solutions, build_transfer_prompt, completed_transfer
from gct.telemetry.metrics import AsyncMetrics, NullMetrics
from gct.utils.hashing import stable_seed
from gct.utils.jsonl import append_jsonl, read_jsonl


def run_dynamic_sae_transfer(
    config: ExperimentConfig,
    queue_path: Path,
    tasks_path: Path,
    anchor_solutions_path: Path,
    raw_out: Path,
    samples_per_target: int = 3,
    claim_size: int = 1,
    lease_seconds: float = 900.0,
    worker_id: str | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    if not use_sglang(config):
        raise ValueError("run-dynamic-sae-transfer requires --inference-backend sglang")

    tasks = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    anchor_solutions = _load_anchor_solutions(anchor_solutions_path)
    done = completed_transfer(raw_out) if config.runtime.resume else set()
    queue = SqliteWorkQueue(queue_path)
    worker = worker_id or default_worker_id()
    client = SglangClient(config)
    metrics = (
        AsyncMetrics(config.runtime.output_dir / "metrics" / "dynamic_sae_transfer_metrics.jsonl", config.telemetry.flush_interval_s)
        if config.telemetry.enabled
        else NullMetrics()
    )

    completed_items = 0
    generated = 0
    try:
        while max_items is None or completed_items < max_items:
            leases = queue.claim(claim_size, worker, lease_seconds)
            if not leases:
                break
            for lease in leases:
                try:
                    row = lease.payload
                    target_id = row["target_task_id"]
                    anchor_id = row["anchor_task_id"]
                    if target_id not in tasks or anchor_id not in tasks or anchor_id not in anchor_solutions:
                        queue.complete(lease.item_id)
                        completed_items += 1
                        continue
                    prompt = build_transfer_prompt(
                        tasks[anchor_id]["prompt"],
                        anchor_solutions[anchor_id],
                        tasks[target_id]["prompt"],
                    )
                    started_item = time.perf_counter()
                    for sample_index in range(samples_per_target):
                        if (target_id, sample_index) in done:
                            continue
                        seed = stable_seed(config.runtime.seed, target_id, anchor_id, sample_index)
                        started = time.perf_counter()
                        output = client.generate(prompt, seed)
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
                                    "worker_id": worker,
                                }
                            ],
                        )
                        done.add((target_id, sample_index))
                        generated += 1
                        metrics.observe("dynamic_sae_transfer.latency_s", elapsed)
                        metrics.incr("dynamic_sae_transfer.generations")
                    queue.complete(lease.item_id)
                    completed_items += 1
                    metrics.observe("dynamic_sae_transfer.item_latency_s", time.perf_counter() - started_item)
                except Exception as exc:  # noqa: BLE001
                    queue.fail(lease.item_id, repr(exc))
    finally:
        metrics.close()

    stats = queue.stats()
    stats.update({"worker_id": worker, "completed_items_this_run": completed_items, "generations_this_run": generated})
    return stats
