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

    resource = subparsers.add_parser("resource-report", help="Detect GPU resources and estimate parallelism.")
    resource.add_argument("--out", type=Path, default=None)

    status = subparsers.add_parser("status", help="Summarize JSONL progress and latency.")
    status.add_argument("--raw", type=Path, required=True)
    status.add_argument("--expected", type=int, default=None)

    args = parser.parse_args(argv)
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
