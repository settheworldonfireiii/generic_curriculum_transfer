from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal local shells.
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable

from gct.math_compat.utils import (
    MATH_CONFIGS,
    aggregate_top_features,
    answers_match,
    append_jsonl,
    build_problem_prompt,
    build_transfer_prompt,
    combined_metric,
    extract_final_answer,
    parse_level,
    read_jsonl,
    summarize_sweep,
    task_id,
    write_csv,
    write_jsonl,
)


REGIMES = {
    "combo": [10, 16, 24, 30],
    "layer28": [28],
}
VARIANTS = ("cos07", "cos01")


def stable_seed(seed: int, *parts: object) -> int:
    key = ":".join([str(seed), *[str(part) for part in parts]]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31)


def dtype_from_name(name: str, torch: Any) -> Any:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def load_math_rows(args: Any) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install datasets") from exc
    rows = []
    wanted_levels = set(args.levels)
    global_index = 0
    for config in args.configs:
        dataset = load_dataset(args.dataset, config, split=args.split)
        for local_index, item in enumerate(dataset):
            level = parse_level(item.get("level"))
            if level not in wanted_levels:
                continue
            rows.append(
                {
                    "task_id": task_id(args.split, global_index),
                    "split": args.split,
                    "index": global_index,
                    "config": config,
                    "local_index": local_index,
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "expected_answer": extract_final_answer(item["solution"]),
                    "level": level,
                    "category": item.get("type") or config,
                }
            )
            global_index += 1
            if args.limit is not None and len(rows) >= args.limit:
                return rows
    if not rows:
        raise SystemExit("No MATH rows matched the requested split/levels.")
    return rows


def load_model_and_tokenizer(args: Any) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install torch and transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype, torch),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer, torch


def generate_once(model: Any, tokenizer: Any, prompt: str, args: Any, sample_seed: int, torch: Any) -> str:
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(
        model_device(model)
    )
    do_sample = args.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def encode_hidden_states(model: Any, tokenizer: Any, text: str, max_length: int, torch: Any) -> tuple[Any, Any]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model_device(model))
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["attention_mask"][0].bool()


def top_pooled_features(sae: Any, hidden_state: Any, attention_mask: Any, top_k: int, torch: Any) -> tuple[list[int], list[float], float, int]:
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


def completed_sae(path: Path) -> set[tuple[str, int]]:
    return {(row["task_id"], int(row["layer"])) for row in read_jsonl(path)}


def completed_raw(path: Path) -> set[tuple[str, int]]:
    return {(row["task_id"], int(row["sample_index"])) for row in read_jsonl(path)}


def run_math_sweep_sae(args: Any) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "math_sweep_raw.jsonl"
    summary_path = args.out_dir / "math_sweep_summary.csv"
    solved_path = args.out_dir / "math_solved_ids.txt"
    sae_rows_path = args.out_dir / "math_sae_task_layer_rows.jsonl"
    manifest_path = args.out_dir / "math_tasks.jsonl"

    rows = load_math_rows(args)
    _validate_shard(args.num_shards, args.shard_index)
    if args.num_shards > 1:
        before = len(rows)
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(rows)}/{before} rows")
    write_jsonl(manifest_path, rows)
    print(f"Loaded {len(rows)} Competition-MATH rows into {manifest_path}")

    try:
        from sparsify import Sae
    except ImportError as exc:
        raise SystemExit("Missing dependency eai-sparsify. Install with: pip install eai-sparsify") from exc

    model, tokenizer, torch = load_model_and_tokenizer(args)
    device = model_device(model)

    done_sae = completed_sae(sae_rows_path) if args.resume else set()
    for layer in args.layers:
        hookpoint = f"layers.{layer}"
        print(f"Loading SAE {args.sae_repo} hookpoint={hookpoint}")
        sae = Sae.load_from_hub(args.sae_repo, hookpoint=hookpoint).to(device)
        sae.eval()
        for row in tqdm(rows, desc=f"MATH SAE layer {layer}"):
            if (row["task_id"], layer) in done_sae:
                continue
            prompt = build_problem_prompt(row["problem"])
            hidden_states, attention_mask = encode_hidden_states(model, tokenizer, prompt, args.max_input_tokens, torch)
            feature_ids, values, l0, n_tokens = top_pooled_features(
                sae,
                hidden_states[layer + 1],
                attention_mask,
                args.top_features_per_layer,
                torch,
            )
            append_jsonl(
                sae_rows_path,
                {
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "index": row["index"],
                    "level": row["level"],
                    "category": row["category"],
                    "layer": layer,
                    "hookpoint": hookpoint,
                    "features": {f"L{layer}:{fid}": float(value) for fid, value in zip(feature_ids, values, strict=True)},
                    "mean_l0": float(l0),
                    "n_tokens": int(n_tokens),
                },
            )
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done_raw = completed_raw(raw_path) if args.resume else set()
    for row in tqdm(rows, desc="MATH sweep"):
        prompt = build_problem_prompt(row["problem"])
        for sample_index in range(args.samples_per_problem):
            if (row["task_id"], sample_index) in done_raw:
                continue
            raw_output = generate_once(
                model,
                tokenizer,
                prompt,
                args,
                stable_seed(args.seed, row["task_id"], sample_index),
                torch,
            )
            predicted = extract_final_answer(raw_output)
            exact = answers_match(predicted, row["expected_answer"])
            if exact:
                problem_prefix = " ".join(str(row["problem"]).split())[:180]
                print(
                    "[SOLVED] "
                    f"task_id={row['task_id']} sample={sample_index} "
                    f"level={row['level']} category={row['category']} "
                    f"predicted={predicted!r} expected={row['expected_answer']!r} "
                    f"problem={problem_prefix}",
                    flush=True,
                )
            append_jsonl(
                raw_path,
                {
                    "task_id": row["task_id"],
                    "split": row["split"],
                    "index": row["index"],
                    "level": row["level"],
                    "category": row["category"],
                    "sample_index": sample_index,
                    "exact": exact,
                    "predicted_answer": predicted,
                    "expected_answer": row["expected_answer"],
                    "raw_output": raw_output,
                },
            )

    summary = summarize_sweep(raw_path, summary_path, solved_path)
    aggregate_top_features(
        sae_rows_path,
        {task: bool(row["solved"]) for task, row in summary.items()},
        args.out_dir / "math_top_features_by_layer.csv",
        args.out_dir / "math_top_features_solved_vs_unsolved.csv",
    )


def merge_math_sweep_shards(args: Any) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    sae_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for shard_dir in args.shard_dirs:
        tasks_path = shard_dir / "math_tasks.jsonl"
        raw_path = shard_dir / "math_sweep_raw.jsonl"
        sae_path = shard_dir / "math_sae_task_layer_rows.jsonl"
        _require_file(tasks_path)
        _require_file(raw_path)
        _require_file(sae_path)
        for row in read_jsonl(tasks_path):
            tasks_by_id[row["task_id"]] = row
        for row in read_jsonl(raw_path):
            raw_by_key[(row["task_id"], int(row["sample_index"]))] = row
        for row in read_jsonl(sae_path):
            sae_by_key[(row["task_id"], int(row["layer"]))] = row

    task_rows = sorted(tasks_by_id.values(), key=lambda row: (str(row.get("split", "")), int(row["index"])))
    raw_rows = sorted(raw_by_key.values(), key=lambda row: (int(row["index"]), int(row["sample_index"])))
    sae_rows = sorted(sae_by_key.values(), key=lambda row: (int(row["index"]), int(row["layer"])))

    manifest_path = args.out_dir / "math_tasks.jsonl"
    raw_path = args.out_dir / "math_sweep_raw.jsonl"
    sae_path = args.out_dir / "math_sae_task_layer_rows.jsonl"
    write_jsonl(manifest_path, task_rows)
    write_jsonl(raw_path, raw_rows)
    write_jsonl(sae_path, sae_rows)
    print(f"Wrote {manifest_path} ({len(task_rows)} tasks)")
    print(f"Wrote {raw_path} ({len(raw_rows)} attempts)")
    print(f"Wrote {sae_path} ({len(sae_rows)} task-layer rows)")

    summary = summarize_sweep(raw_path, args.out_dir / "math_sweep_summary.csv", args.out_dir / "math_solved_ids.txt")
    aggregate_top_features(
        sae_path,
        {task: bool(row["solved"]) for task, row in summary.items()},
        args.out_dir / "math_top_features_by_layer.csv",
        args.out_dir / "math_top_features_solved_vs_unsolved.csv",
        top_n=args.top_n,
    )


def build_math_neighbors(args: Any) -> None:
    summary = _load_summary(args.out_dir / "math_sweep_summary.csv")
    rows_path = args.out_dir / "math_sae_task_layer_rows.jsonl"
    solved_ids = [task_id for task_id, row in summary.items() if row["solved"] == "True"]
    target_ids = [task_id for task_id, row in summary.items() if row["solved"] != "True"]
    print(f"Solved anchors: {len(solved_ids)}")
    print(f"Unsolved targets: {len(target_ids)}")

    for regime_name, layers in REGIMES.items():
        embeddings = _features_for_layers(rows_path, layers)
        for variant in VARIANTS:
            out_rows = []
            for target_id in target_ids:
                if target_id not in embeddings:
                    continue
                scored = []
                for anchor_id in solved_ids:
                    if anchor_id not in embeddings:
                        continue
                    score, cosine, jaccard, tanimoto = combined_metric(embeddings[target_id], embeddings[anchor_id], variant)
                    scored.append((score, anchor_id, cosine, jaccard, tanimoto))
                scored.sort(reverse=True)
                for rank, (score, anchor_id, cosine, jaccard, tanimoto) in enumerate(scored[: args.top_k], start=1):
                    target_meta = summary[target_id]
                    anchor_meta = summary[anchor_id]
                    out_rows.append(
                        {
                            "target_task_id": target_id,
                            "anchor_task_id": anchor_id,
                            "neighbor_rank": rank,
                            "score_variant": variant,
                            "layer_regime": regime_name,
                            "combined_similarity": score,
                            "cosine": cosine,
                            "weighted_jaccard": jaccard,
                            "tanimoto": tanimoto,
                            "target_level": target_meta["level"],
                            "target_category": target_meta["category"],
                            "anchor_level": anchor_meta["level"],
                            "anchor_category": anchor_meta["category"],
                        }
                    )
            out_path = args.out_dir / f"neighbors_{regime_name}_{variant}.csv"
            write_csv(
                out_path,
                out_rows,
                [
                    "target_task_id",
                    "anchor_task_id",
                    "neighbor_rank",
                    "score_variant",
                    "layer_regime",
                    "combined_similarity",
                    "cosine",
                    "weighted_jaccard",
                    "tanimoto",
                    "target_level",
                    "target_category",
                    "anchor_level",
                    "anchor_category",
                ],
            )
            print(f"Wrote {out_path}")


def merge_math_transfer_shards(args: Any) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"transfer_{args.regime}_{args.variant}"
    summary_paths = list(args.summary_paths or sorted(args.out_dir.glob(f"{prefix}_shard*of*_summary.csv")))
    if not summary_paths:
        raise SystemExit(f"No transfer shard summaries found for {prefix}")

    rows_by_target: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        _require_file(path)
        with path.open() as f:
            for row in csv.DictReader(f):
                rows_by_target[row["target_task_id"]] = row

    rows = list(rows_by_target.values())
    rows.sort(key=lambda row: (-int(str(row.get("solved", "")) == "True"), -float(row.get("success_rate", 0.0)), row["target_task_id"]))
    summary_out = args.summary_out or args.out_dir / f"{prefix}_summary.csv"
    write_csv(
        summary_out,
        rows,
        [
            "target_task_id",
            "anchor_task_id",
            "score_variant",
            "layer_regime",
            "attempts",
            "successes",
            "solved",
            "success_rate",
        ],
    )
    print(f"Wrote {summary_out}")
    print(f"Solved targets: {sum(1 for row in rows if str(row.get('solved', '')) == 'True')}/{len(rows)}")

    if args.raw_out is None:
        return
    raw_paths = list(args.raw_paths or sorted(args.out_dir.glob(f"{prefix}_shard*of*_raw.jsonl")))
    raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in raw_paths:
        _require_file(path)
        for row in read_jsonl(path):
            raw_by_key[(row["target_task_id"], int(row["sample_index"]))] = row
    raw_rows = sorted(raw_by_key.values(), key=lambda row: (row["target_task_id"], int(row["sample_index"])))
    write_jsonl(args.raw_out, raw_rows)
    print(f"Wrote {args.raw_out} ({len(raw_rows)} attempts)")


def run_math_transfer(args: Any) -> None:
    tasks = _load_tasks(args.out_dir / "math_tasks.jsonl")
    anchor_solutions = _load_successful_solutions(args.out_dir / "math_sweep_raw.jsonl")
    neighbors = _load_top1_neighbors(args.neighbors)
    _validate_shard(args.num_shards, args.shard_index)
    if args.num_shards > 1:
        before = len(neighbors)
        neighbors = [row for idx, row in enumerate(neighbors) if idx % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(neighbors)}/{before} neighbor rows")
    if args.target_limit is not None:
        neighbors = neighbors[: args.target_limit]
    done = _completed_transfer(args.raw_out) if args.resume else set()
    model, tokenizer, torch = load_model_and_tokenizer(args)

    for row in tqdm(neighbors, desc=f"MATH transfer {args.neighbors.name}"):
        target_id = row["target_task_id"]
        anchor_id = row["anchor_task_id"]
        if anchor_id not in anchor_solutions:
            continue
        target = tasks[target_id]
        anchor = tasks[anchor_id]
        prompt = build_transfer_prompt(anchor["problem"], anchor_solutions[anchor_id], target["problem"])
        for sample_index in range(args.samples_per_target):
            key = (target_id, sample_index)
            if key in done:
                continue
            raw_output = generate_once(
                model,
                tokenizer,
                prompt,
                args,
                stable_seed(args.seed, args.neighbors.name, target_id, anchor_id, sample_index),
                torch,
            )
            predicted = extract_final_answer(raw_output)
            exact = answers_match(predicted, target["expected_answer"])
            append_jsonl(
                args.raw_out,
                {
                    "target_task_id": target_id,
                    "anchor_task_id": anchor_id,
                    "sample_index": sample_index,
                    "exact": exact,
                    "predicted_answer": predicted,
                    "expected_answer": target["expected_answer"],
                    "raw_output": raw_output,
                    "score_variant": row["score_variant"],
                    "layer_regime": row["layer_regime"],
                    "combined_similarity": row["combined_similarity"],
                    "cosine": row["cosine"],
                    "weighted_jaccard": row["weighted_jaccard"],
                    "tanimoto": row["tanimoto"],
                },
            )
    _summarize_transfer(args.raw_out, args.summary_out)


def run_math_context_ablation(args: Any) -> None:
    tasks = _load_tasks(args.out_dir / "math_tasks.jsonl")
    anchor_solutions = _load_successful_solutions(args.out_dir / "math_sweep_raw.jsonl")
    pairs = _load_context_solved_pairs(args.transfer_summary)
    _validate_shard(args.num_shards, args.shard_index)
    if args.num_shards > 1:
        before = len(pairs)
        pairs = [row for idx, row in enumerate(pairs) if idx % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(pairs)}/{before} ablation pairs")
    if args.target_limit is not None:
        pairs = pairs[: args.target_limit]
    if not pairs:
        print(f"No context-solved pairs found in {args.transfer_summary}")
        return
    done = _completed_ablation(args.raw_out) if args.resume else set()
    model, tokenizer, torch = load_model_and_tokenizer(args)

    work = []
    for pair in pairs:
        target_id = pair["target_task_id"]
        anchor_id = pair["anchor_task_id"]
        if target_id not in tasks or anchor_id not in tasks or anchor_id not in anchor_solutions:
            continue
        for condition in ("no_context", "with_anchor"):
            for run_group in range(args.run_groups):
                for sample_index in range(args.samples_per_group):
                    work.append((pair, condition, run_group, sample_index))

    for pair, condition, run_group, sample_index in tqdm(work, desc=f"MATH ablation {args.transfer_summary.name}"):
        target_id = pair["target_task_id"]
        anchor_id = pair["anchor_task_id"]
        key = (target_id, condition, run_group, sample_index)
        if args.resume and key in done:
            continue
        target = tasks[target_id]
        anchor = tasks[anchor_id]
        if condition == "no_context":
            prompt = build_problem_prompt(target["problem"])
        else:
            prompt = build_transfer_prompt(anchor["problem"], anchor_solutions[anchor_id], target["problem"])
        raw_output = generate_once(
            model,
            tokenizer,
            prompt,
            args,
            stable_seed(args.seed, args.transfer_summary.name, target_id, anchor_id, condition, run_group, sample_index),
            torch,
        )
        predicted = extract_final_answer(raw_output)
        exact = answers_match(predicted, target["expected_answer"])
        append_jsonl(
            args.raw_out,
            {
                "target_task_id": target_id,
                "anchor_task_id": anchor_id if condition == "with_anchor" else "",
                "condition": condition,
                "run_group": run_group,
                "sample_index": sample_index,
                "exact": exact,
                "predicted_answer": predicted,
                "expected_answer": target["expected_answer"],
                "raw_output": raw_output,
                "score_variant": pair["score_variant"],
                "layer_regime": pair["layer_regime"],
                "combined_similarity": pair.get("combined_similarity", ""),
            },
        )
    _summarize_ablation(args.raw_out, args.summary_out)


def _load_summary(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as f:
        return {row["task_id"]: row for row in csv.DictReader(f)}


def _features_for_layers(rows_path: Path, layers: list[int]) -> dict[str, dict[str, float]]:
    by_task: dict[str, dict[str, float]] = {}
    for row in read_jsonl(rows_path):
        if int(row["layer"]) not in layers:
            continue
        features = by_task.setdefault(row["task_id"], {})
        for key, value in row["features"].items():
            features[key] = float(value)
    return by_task


def _load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in read_jsonl(path)}


def _load_successful_solutions(raw_path: Path) -> dict[str, str]:
    solutions = {}
    for row in read_jsonl(raw_path):
        if row["exact"] and row["task_id"] not in solutions:
            solutions[row["task_id"]] = row["raw_output"]
    return solutions


def _load_top1_neighbors(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = [row for row in csv.DictReader(f) if int(row["neighbor_rank"]) == 1]
    rows.sort(key=lambda row: (-float(row["combined_similarity"]), row["target_task_id"]))
    return rows


def _load_context_solved_pairs(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = [row for row in csv.DictReader(f) if row.get("solved") == "True"]
    rows.sort(key=lambda row: (-float(row.get("success_rate", 0.0)), row["target_task_id"]))
    return rows


def _completed_transfer(path: Path) -> set[tuple[str, int]]:
    return {(row["target_task_id"], int(row["sample_index"])) for row in read_jsonl(path)}


def _completed_ablation(path: Path) -> set[tuple[str, str, int, int]]:
    return {
        (row["target_task_id"], row["condition"], int(row["run_group"]), int(row["sample_index"]))
        for row in read_jsonl(path)
    }


def _summarize_transfer(raw_path: Path, summary_path: Path) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        current = by_task.setdefault(
            row["target_task_id"],
            {
                "target_task_id": row["target_task_id"],
                "anchor_task_id": row["anchor_task_id"],
                "attempts": 0,
                "successes": 0,
                "score_variant": row["score_variant"],
                "layer_regime": row["layer_regime"],
            },
        )
        current["attempts"] += 1
        current["successes"] += int(bool(row["exact"]))
    rows = []
    for row in by_task.values():
        row["solved"] = row["successes"] > 0
        row["success_rate"] = row["successes"] / row["attempts"] if row["attempts"] else 0.0
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["solved"]), -float(row["success_rate"]), row["target_task_id"]))
    write_csv(
        summary_path,
        rows,
        [
            "target_task_id",
            "anchor_task_id",
            "score_variant",
            "layer_regime",
            "attempts",
            "successes",
            "solved",
            "success_rate",
        ],
    )
    print(f"Wrote {summary_path}")
    print(f"Solved targets: {sum(1 for row in rows if row['solved'])}/{len(rows)}")


def _summarize_ablation(raw_path: Path, summary_path: Path) -> None:
    group_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        key = (row["target_task_id"], row["condition"], int(row["run_group"]))
        current = group_rows.setdefault(
            key,
            {
                "target_task_id": row["target_task_id"],
                "anchor_task_id": row["anchor_task_id"],
                "condition": row["condition"],
                "run_group": int(row["run_group"]),
                "score_variant": row["score_variant"],
                "layer_regime": row["layer_regime"],
                "attempts": 0,
                "successes": 0,
            },
        )
        current["attempts"] += 1
        current["successes"] += int(bool(row["exact"]))

    rows = []
    for row in group_rows.values():
        row["solved"] = row["successes"] > 0
        row["success_rate"] = row["successes"] / row["attempts"] if row["attempts"] else 0.0
        rows.append(row)
    rows.sort(key=lambda row: (row["target_task_id"], row["condition"], row["run_group"]))
    write_csv(
        summary_path,
        rows,
        [
            "target_task_id",
            "anchor_task_id",
            "condition",
            "run_group",
            "score_variant",
            "layer_regime",
            "attempts",
            "successes",
            "solved",
            "success_rate",
        ],
    )
    print(f"Wrote {summary_path}")
    for target_id in sorted({row["target_task_id"] for row in rows}):
        for condition in ("no_context", "with_anchor"):
            subset = [row for row in rows if row["target_task_id"] == target_id and row["condition"] == condition]
            solved_groups = sum(1 for row in subset if row["solved"])
            successes = sum(int(row["successes"]) for row in subset)
            attempts = sum(int(row["attempts"]) for row in subset)
            print(
                f"{target_id} {condition}: solved_groups={solved_groups}/{len(subset)} "
                f"attempt_successes={successes}/{attempts}"
            )


def _validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required shard file: {path}")


def default_math_configs() -> list[str]:
    return list(MATH_CONFIGS)
