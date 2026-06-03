# ETL Pipeline

## 1. Raw Data

Input is a Hugging Face dataset selected by CLI:

```bash
gct prepare-data --dataset DATASET --split SPLIT --prompt-column COL --answer-column COL
```

The default config uses `EleutherAI/hendrycks_math`.

## 2. HF Or Spark Preprocessing

HF datasets is the default path. Use it for normal experiment sizes, streaming
loads, and simple filters.

Spark is optional. Use it when data requires distributed filtering, joins,
aggregation, or repartitioning. The scaffold expects HF-normalized JSONL first;
dataset-specific Spark jobs can then extend `gct/etl/spark.py`.

## 3. Saved Dataset Format

The saved canonical format is JSONL:

```json
{"task_id":"task_...","prompt":"...","answer":"...","metadata":{"level":"2","category":"Algebra"}}
```

This format is intentionally easy to copy between SLURM and cloud machines.

## 4. PyTorch Dataset And DataLoader

`JsonlTaskDataset` loads the saved JSONL. `build_dataloader` controls
`batch_size`, `num_workers`, prefetching, and pinning.

## 5. Training Or Ablation Loop

`run-sweep` performs baseline generation and writes `sweep/raw.jsonl`,
`sweep/summary.csv`, and `sweep/solved_ids.txt`.

`run-train` performs supervised causal-LM training on prompt+answer text.
`run-ablation` performs batched generation and writes resumable raw JSONL.

## 6. Observability

Metrics are appended to JSONL asynchronously. W&B can mirror selected metrics
in offline or online mode.
