from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gct.config.load import load_config
from gct.config.schema import ExperimentConfig
from gct.etl.local import prepare_dataset_local
from gct.etl.spark import prepare_dataset_spark
from gct.runtime.resources import detect_gpus, estimate_parallelism, write_resource_report
from gct.schedulers.curriculum import write_curriculum_plan
from gct.sae.neighbors import build_neighbor_rows, write_neighbors
from gct.sae.store import load_features_for_layers, load_task_summary
from gct.utils.jsonl import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gct", description="Generic curriculum transfer runner.")
    parser.add_argument("--config", type=Path, default=Path("configs/competitive_math.yaml"))
    parser.add_argument("--model", default=None, help="HF model name override.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--backend", choices=["local", "slurm", "ibm"], default=None)
    parser.add_argument("--inference-backend", choices=["hf", "sglang"], default=None)
    parser.add_argument("--sglang-base-url", default=None)
    parser.add_argument("--sglang-api-key", default=None)
    parser.add_argument("--sglang-model", default=None)
    parser.add_argument("--sglang-timeout-s", type=float, default=None)
    parser.add_argument("--sglang-max-concurrency", type=int, default=None)
    parser.add_argument("--wandb-api-key", default=None, help="W&B API key for online/offline synced runs.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Load HF data and save normalized tasks.")
    add_dataset_args(prepare)
    prepare.add_argument("--engine", choices=["hf", "spark"], default=None)

    plan = subparsers.add_parser("plan-curriculum", help="Create a knapsack-style curriculum plan.")
    plan.add_argument("--tasks", type=Path, default=None)
    plan.add_argument("--out", type=Path, default=None)
    plan.add_argument("--token-capacity", type=int, default=32000)

    shard = subparsers.add_parser("shard-plan", help="Shard a curriculum plan for SLURM/cloud workers.")
    shard.add_argument("--plan", type=Path, default=None)
    shard.add_argument("--out-dir", type=Path, default=None)
    shard.add_argument("--num-shards", type=int, default=4)

    train = subparsers.add_parser("run-train", help="Run the PyTorch DataLoader training loop.")
    train.add_argument("--tasks", type=Path, default=None)
    train.add_argument("--max-steps", type=int, default=None)

    sweep = subparsers.add_parser("run-sweep", help="Run model sweep and write solved/unsolved summary.")
    sweep.add_argument("--tasks", type=Path, default=None)
    sweep.add_argument("--raw-out", type=Path, default=None)
    sweep.add_argument("--summary-out", type=Path, default=None)
    sweep.add_argument("--solved-ids-out", type=Path, default=None)
    sweep.add_argument("--samples-per-task", type=int, default=3)

    ablation = subparsers.add_parser("run-ablation", help="Run batched generation ablations.")
    ablation.add_argument("--tasks", type=Path, default=None)
    ablation.add_argument("--raw-out", type=Path, default=None)
    ablation.add_argument("--samples-per-task", type=int, default=3)

    extract_sae = subparsers.add_parser("extract-sae", help="Extract pooled SAE task features.")
    extract_sae.add_argument("--tasks", type=Path, default=None)
    extract_sae.add_argument("--out", type=Path, default=None)
    extract_sae.add_argument("--sae-repo", default=None)
    extract_sae.add_argument("--layers", type=int, nargs="+", default=None)
    extract_sae.add_argument("--top-features-per-layer", type=int, default=None)

    neighbors = subparsers.add_parser("build-sae-neighbors", help="Build solved-anchor SAE neighbor schedules.")
    neighbors.add_argument("--summary", type=Path, required=True)
    neighbors.add_argument("--features", type=Path, default=None)
    neighbors.add_argument("--target-plan", type=Path, default=None)
    neighbors.add_argument("--out", type=Path, default=None)
    neighbors.add_argument("--layers", type=int, nargs="+", default=None)
    neighbors.add_argument("--layer-regime", default="combo")
    neighbors.add_argument("--variant", choices=["cos07", "cos01", "arc_weighted"], default=None)
    neighbors.add_argument("--top-k", type=int, default=None)

    transfer = subparsers.add_parser("run-sae-transfer", help="Run top-1 SAE-anchor transfer generations.")
    transfer.add_argument("--tasks", type=Path, default=None)
    transfer.add_argument("--neighbors", type=Path, required=True)
    transfer.add_argument("--anchor-solutions", type=Path, required=True)
    transfer.add_argument("--raw-out", type=Path, default=None)
    transfer.add_argument("--samples-per-target", type=int, default=3)

    init_transfer_queue = subparsers.add_parser(
        "init-transfer-work-queue",
        help="Create a SQLite dynamic work queue from top-ranked SAE neighbor rows.",
    )
    init_transfer_queue.add_argument("--neighbors", type=Path, required=True)
    init_transfer_queue.add_argument("--queue", type=Path, default=None)
    init_transfer_queue.add_argument("--rank", type=int, default=1)

    transfer_worker = subparsers.add_parser(
        "run-dynamic-sae-transfer",
        help="Run an SGLang-backed dynamic SAE-transfer worker from a SQLite queue.",
    )
    transfer_worker.add_argument("--queue", type=Path, default=None)
    transfer_worker.add_argument("--tasks", type=Path, default=None)
    transfer_worker.add_argument("--anchor-solutions", type=Path, required=True)
    transfer_worker.add_argument("--raw-out", type=Path, default=None)
    transfer_worker.add_argument("--samples-per-target", type=int, default=3)
    transfer_worker.add_argument("--claim-size", type=int, default=1)
    transfer_worker.add_argument("--lease-seconds", type=float, default=900.0)
    transfer_worker.add_argument("--worker-id", default=None)
    transfer_worker.add_argument("--max-items", type=int, default=None)

    queue_status = subparsers.add_parser("work-queue-status", help="Print SQLite dynamic work queue status.")
    queue_status.add_argument("--queue", type=Path, default=None)

    resource = subparsers.add_parser("resource-report", help="Detect GPU resources and estimate parallelism.")
    resource.add_argument("--out", type=Path, default=None)

    status = subparsers.add_parser("status", help="Summarize JSONL progress and latency.")
    status.add_argument("--raw", type=Path, required=True)
    status.add_argument("--expected", type=int, default=None)

    add_math_compat_subcommands(subparsers)

    args = parser.parse_args(argv)

    if args.command.startswith("math-"):
        run_math_compat_command(args)
        return

    config = config_from_args(args)

    if args.command == "prepare-data":
        engine = args.engine or config.dataset.preprocessing_engine
        if engine == "spark":
            output = prepare_dataset_spark(config)
        else:
            output = prepare_dataset_local(config)
        print(output)
        return

    if args.command == "plan-curriculum":
        tasks = args.tasks or config.runtime.output_dir / "datasets" / "tasks.jsonl"
        out = args.out or config.runtime.output_dir / "plans" / "curriculum.jsonl"
        output = write_curriculum_plan(tasks, out, args.token_capacity)
        print(output)
        return

    if args.command == "shard-plan":
        plan_path = args.plan or config.runtime.output_dir / "plans" / "curriculum.jsonl"
        out_dir = args.out_dir or config.runtime.output_dir / "plans"
        shard_plan(plan_path, out_dir, args.num_shards)
        return

    if args.command == "run-train":
        from gct.training.loop import run_training_loop

        tasks = args.tasks or config.runtime.output_dir / "plans" / "curriculum.jsonl"
        run_training_loop(config, tasks, max_steps=args.max_steps)
        return

    if args.command == "run-sweep":
        from gct.sweep.runner import run_sweep

        tasks = args.tasks or config.runtime.output_dir / "datasets" / "tasks.jsonl"
        raw_out = args.raw_out or config.runtime.output_dir / "sweep" / "raw.jsonl"
        summary_out = args.summary_out or config.runtime.output_dir / "sweep" / "summary.csv"
        solved_ids_out = args.solved_ids_out or config.runtime.output_dir / "sweep" / "solved_ids.txt"
        output = run_sweep(
            config,
            tasks,
            raw_out,
            summary_out,
            solved_ids_out,
            samples_per_task=args.samples_per_task,
        )
        print(output)
        return

    if args.command == "run-ablation":
        from gct.training.ablation import run_ablation

        tasks = args.tasks or config.runtime.output_dir / "plans" / "curriculum.jsonl"
        raw_out = args.raw_out or config.runtime.output_dir / "ablation" / "raw.jsonl"
        run_ablation(config, tasks, raw_out, samples_per_task=args.samples_per_task)
        return

    if args.command == "extract-sae":
        from gct.sae.extract import extract_sae_features

        tasks = args.tasks or config.runtime.output_dir / "datasets" / "tasks.jsonl"
        out = args.out or config.runtime.output_dir / "sae" / "task_layer_rows.jsonl"
        layers = args.layers or list(config.sae.layers)
        top_features = args.top_features_per_layer or config.sae.top_features_per_layer
        repo = args.sae_repo or config.sae.repo
        output = extract_sae_features(config, tasks, out, repo, layers, top_features)
        print(output)
        return

    if args.command == "build-sae-neighbors":
        features_path = args.features or config.runtime.output_dir / "sae" / "task_layer_rows.jsonl"
        variant = args.variant or config.sae.default_variant
        layers = set(args.layers or (list(config.sae.combo_layers) if args.layer_regime == "combo" else list(config.sae.layers)))
        top_k = args.top_k or config.sae.top_k_neighbors
        summary = load_task_summary(args.summary)
        features = load_features_for_layers(features_path, layers)
        allowed_target_ids = _target_ids_from_plan(args.target_plan) if args.target_plan else None
        rows = build_neighbor_rows(
            summary=summary,
            features=features,
            top_k=top_k,
            variant=variant,
            layer_regime=args.layer_regime,
            allowed_target_ids=allowed_target_ids,
        )
        out = args.out or config.runtime.output_dir / "sae" / f"neighbors_{args.layer_regime}_{variant}.csv"
        write_neighbors(out, rows)
        print(f"{out}: {len(rows)} rows")
        return

    if args.command == "run-sae-transfer":
        from gct.sae.transfer import run_sae_transfer

        tasks = args.tasks or config.runtime.output_dir / "datasets" / "tasks.jsonl"
        raw_out = args.raw_out or config.runtime.output_dir / "sae_transfer" / "raw.jsonl"
        output = run_sae_transfer(
            config,
            tasks,
            args.neighbors,
            args.anchor_solutions,
            raw_out,
            samples_per_target=args.samples_per_target,
        )
        print(output)
        return

    if args.command == "init-transfer-work-queue":
        from gct.runtime.work_queue import SqliteWorkQueue
        from gct.sae.transfer import load_top_neighbors

        queue_path = args.queue or config.runtime.output_dir / "queues" / "sae_transfer.sqlite"
        rows = load_top_neighbors(args.neighbors, rank=args.rank)
        queue = SqliteWorkQueue(queue_path)
        queue.initialize()
        inserted = queue.enqueue_many(
            rows,
            id_field="target_task_id",
            priority_field="combined_similarity",
        )
        print(json.dumps({"queue": str(queue_path), "inserted": inserted, "total_rows": len(rows)}, indent=2))
        return

    if args.command == "run-dynamic-sae-transfer":
        from gct.sae.dynamic_transfer import run_dynamic_sae_transfer

        queue_path = args.queue or config.runtime.output_dir / "queues" / "sae_transfer.sqlite"
        tasks = args.tasks or config.runtime.output_dir / "datasets" / "tasks.jsonl"
        raw_out = args.raw_out or config.runtime.output_dir / "sae_transfer" / "raw.jsonl"
        output = run_dynamic_sae_transfer(
            config,
            queue_path,
            tasks,
            args.anchor_solutions,
            raw_out,
            samples_per_target=args.samples_per_target,
            claim_size=args.claim_size,
            lease_seconds=args.lease_seconds,
            worker_id=args.worker_id,
            max_items=args.max_items,
        )
        print(json.dumps(output, indent=2))
        return

    if args.command == "work-queue-status":
        from gct.runtime.work_queue import SqliteWorkQueue

        queue_path = args.queue or config.runtime.output_dir / "queues" / "sae_transfer.sqlite"
        print(json.dumps(SqliteWorkQueue(queue_path).stats(), indent=2))
        return

    if args.command == "resource-report":
        out = args.out or config.runtime.output_dir / "resource_report.json"
        write_resource_report(str(out), config.model.name, config.model.dtype)
        print(json.dumps({"out": str(out), "gpus": len(detect_gpus())}, indent=2))
        return

    if args.command == "status":
        print_status(args.raw, args.expected)
        return


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=None, help="HF dataset name, e.g. EleutherAI/hendrycks_math.")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--answer-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--level-column", default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--filter-expr", default=None)
    parser.add_argument("--streaming", action="store_true")


def add_math_compat_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from gct.math_compat.utils import MATH_CONFIGS

    math_sweep = subparsers.add_parser(
        "math-sweep-sae",
        help="Run the original Competition-MATH sweep plus SAE extraction.",
    )
    math_sweep.add_argument("--dataset", default="EleutherAI/hendrycks_math")
    math_sweep.add_argument("--configs", nargs="+", default=list(MATH_CONFIGS))
    math_sweep.add_argument("--split", default="test")
    math_sweep.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    math_sweep.add_argument("--out-dir", type=Path, default=Path("results_math"))
    math_sweep.add_argument("--sae-repo", default="EleutherAI/sae-llama-3-8b-32x")
    math_sweep.add_argument("--layers", type=int, nargs="+", default=[10, 16, 24, 28, 30])
    math_sweep.add_argument("--samples-per-problem", type=int, default=3)
    math_sweep.add_argument("--top-features-per-layer", type=int, default=256)
    math_sweep.add_argument("--limit", type=int, default=None)
    math_sweep.add_argument("--num-shards", type=int, default=1)
    math_sweep.add_argument("--shard-index", type=int, default=0)
    add_math_generation_args(math_sweep, seed=314159, max_input_tokens=4096)

    merge_sweep = subparsers.add_parser(
        "math-merge-sweep-shards",
        help="Merge original Competition-MATH sweep shard outputs.",
    )
    merge_sweep.add_argument(
        "--shard-dirs",
        type=Path,
        nargs="+",
        default=[Path(f"results_math_shard{i}") for i in range(4)],
    )
    merge_sweep.add_argument("--out-dir", type=Path, default=Path("results_math"))
    merge_sweep.add_argument("--top-n", type=int, default=200)

    math_neighbors = subparsers.add_parser(
        "math-build-neighbors",
        help="Build original Competition-MATH SAE neighbor CSVs.",
    )
    math_neighbors.add_argument("--out-dir", type=Path, default=Path("results_math"))
    math_neighbors.add_argument("--top-k", type=int, default=5)

    math_transfer = subparsers.add_parser(
        "math-transfer",
        help="Run original Competition-MATH top-1 SAE-anchor transfer.",
    )
    math_transfer.add_argument("--out-dir", type=Path, default=Path("results_math"))
    math_transfer.add_argument("--neighbors", type=Path, required=True)
    math_transfer.add_argument("--raw-out", type=Path, required=True)
    math_transfer.add_argument("--summary-out", type=Path, required=True)
    math_transfer.add_argument("--samples-per-target", type=int, default=3)
    math_transfer.add_argument("--target-limit", type=int, default=None)
    math_transfer.add_argument("--num-shards", type=int, default=1)
    math_transfer.add_argument("--shard-index", type=int, default=0)
    add_math_generation_args(math_transfer, seed=271828, max_input_tokens=7600)

    merge_transfer = subparsers.add_parser(
        "math-merge-transfer-shards",
        help="Merge original Competition-MATH transfer shard summaries for ablations.",
    )
    merge_transfer.add_argument("--out-dir", type=Path, default=Path("results_math"))
    merge_transfer.add_argument("--regime", choices=["combo", "layer28"], default="combo")
    merge_transfer.add_argument("--variant", choices=["cos07", "cos01"], default="cos07")
    merge_transfer.add_argument("--summary-paths", type=Path, nargs="+", default=None)
    merge_transfer.add_argument("--summary-out", type=Path, default=None)
    merge_transfer.add_argument("--raw-paths", type=Path, nargs="+", default=None)
    merge_transfer.add_argument("--raw-out", type=Path, default=None)

    math_ablation = subparsers.add_parser(
        "math-context-ablation",
        help="Run original Competition-MATH no-context vs with-anchor ablations.",
    )
    math_ablation.add_argument("--out-dir", type=Path, default=Path("results_math"))
    math_ablation.add_argument("--transfer-summary", type=Path, required=True)
    math_ablation.add_argument("--raw-out", type=Path, required=True)
    math_ablation.add_argument("--summary-out", type=Path, required=True)
    math_ablation.add_argument("--run-groups", type=int, default=10)
    math_ablation.add_argument("--samples-per-group", type=int, default=3)
    math_ablation.add_argument("--target-limit", type=int, default=None)
    math_ablation.add_argument("--num-shards", type=int, default=1)
    math_ablation.add_argument("--shard-index", type=int, default=0)
    add_math_generation_args(math_ablation, seed=161803, max_input_tokens=7600)


def add_math_generation_args(parser: argparse.ArgumentParser, *, seed: int, max_input_tokens: int) -> None:
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=max_input_tokens)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")


def run_math_compat_command(args: argparse.Namespace) -> None:
    from gct.math_compat.commands import (
        build_math_neighbors,
        merge_math_sweep_shards,
        merge_math_transfer_shards,
        run_math_context_ablation,
        run_math_sweep_sae,
        run_math_transfer,
    )

    if args.command == "math-sweep-sae":
        run_math_sweep_sae(args)
        return
    if args.command == "math-merge-sweep-shards":
        merge_math_sweep_shards(args)
        return
    if args.command == "math-build-neighbors":
        build_math_neighbors(args)
        return
    if args.command == "math-transfer":
        run_math_transfer(args)
        return
    if args.command == "math-merge-transfer-shards":
        merge_math_transfer_shards(args)
        return
    if args.command == "math-context-ablation":
        run_math_context_ablation(args)
        return
    raise SystemExit(f"Unhandled MATH compatibility command: {args.command}")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides: dict[str, Any] = {}
    dataset_overrides: dict[str, Any] = {}
    option_map = {
        "dataset": "name",
        "dataset_config": "config_name",
        "split": "split",
        "prompt_column": "prompt_column",
        "answer_column": "answer_column",
        "id_column": "id_column",
        "level_column": "level_column",
        "category_column": "category_column",
        "max_rows": "max_rows",
        "filter_expr": "filter_expr",
    }
    for attr, key in option_map.items():
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                dataset_overrides[key] = value
    if getattr(args, "streaming", False):
        dataset_overrides["streaming"] = True
    if getattr(args, "engine", None) is not None:
        dataset_overrides["preprocessing_engine"] = args.engine
    if dataset_overrides:
        overrides["dataset"] = dataset_overrides
    model_overrides: dict[str, Any] = {}
    if args.model is not None:
        model_overrides["name"] = args.model
    if args.dtype is not None:
        model_overrides["dtype"] = args.dtype
    if model_overrides:
        overrides["model"] = model_overrides
    runtime_overrides: dict[str, Any] = {}
    if args.output_dir is not None:
        runtime_overrides["output_dir"] = args.output_dir
    if args.num_workers is not None:
        runtime_overrides["num_workers"] = args.num_workers
    if args.generation_batch_size is not None:
        runtime_overrides["generation_batch_size"] = args.generation_batch_size
    if runtime_overrides:
        overrides["runtime"] = runtime_overrides
    if args.backend is not None:
        overrides["backend"] = {"kind": args.backend}
    inference_overrides: dict[str, Any] = {}
    if args.inference_backend is not None:
        inference_overrides["backend"] = args.inference_backend
    if args.sglang_base_url is not None:
        inference_overrides["sglang_base_url"] = args.sglang_base_url
    if args.sglang_api_key is not None:
        inference_overrides["sglang_api_key"] = args.sglang_api_key
    if args.sglang_model is not None:
        inference_overrides["sglang_model"] = args.sglang_model
    if args.sglang_timeout_s is not None:
        inference_overrides["sglang_timeout_s"] = args.sglang_timeout_s
    if args.sglang_max_concurrency is not None:
        inference_overrides["sglang_max_concurrency"] = args.sglang_max_concurrency
    if inference_overrides:
        overrides["inference"] = inference_overrides
    telemetry_overrides: dict[str, Any] = {}
    if args.wandb_api_key is not None:
        telemetry_overrides["wandb_api_key"] = args.wandb_api_key
    if args.wandb_mode is not None:
        telemetry_overrides["wandb_mode"] = args.wandb_mode
    if args.wandb_project is not None:
        telemetry_overrides["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        telemetry_overrides["wandb_entity"] = args.wandb_entity
    if args.wandb_log_interval is not None:
        telemetry_overrides["wandb_log_interval"] = args.wandb_log_interval
    if telemetry_overrides:
        overrides["telemetry"] = telemetry_overrides
    return load_config(args.config, overrides)


def print_status(raw_path: Path, expected: int | None) -> None:
    rows = list(read_jsonl(raw_path))
    unique_generations = {
        (row.get("task_id"), int(row.get("sample_index", 0)))
        for row in rows
        if row.get("task_id") is not None
    }
    latencies = [float(row["latency_s"]) for row in rows if "latency_s" in row]
    payload = {
        "path": str(raw_path),
        "rows": len(rows),
        "unique_generations": len(unique_generations),
        "expected": expected,
        "completion": (len(unique_generations) / expected if expected else None),
        "mean_latency_s": (sum(latencies) / len(latencies) if latencies else None),
    }
    print(json.dumps(payload, indent=2))


def shard_plan(plan_path: Path, out_dir: Path, num_shards: int) -> None:
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    rows = list(read_jsonl(plan_path))
    for shard_index in range(num_shards):
        shard_rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
        out = out_dir / f"curriculum_shard{shard_index}of{num_shards}.jsonl"
        write_jsonl(out, shard_rows)
        print(f"{out}: {len(shard_rows)}/{len(rows)} rows")


def _target_ids_from_plan(path: Path) -> set[str]:
    return {row["task_id"] for row in read_jsonl(path)}


if __name__ == "__main__":
    main()
