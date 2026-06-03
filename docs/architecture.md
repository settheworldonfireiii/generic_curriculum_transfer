# Architecture

## Design Choice

This is a hybrid scaffold: runnable local/SLURM core with optional production
interfaces. The core commands are plain Python CLIs. Airflow, SLURM, and cloud
providers call those commands instead of duplicating experiment logic.

## Units

- `gct.config`: dataclass config schema and YAML loading.
- `gct.data`: Hugging Face row normalization and PyTorch `Dataset`/`DataLoader`.
- `gct.etl`: local HF preparation plus optional Spark preprocessing.
- `gct.schedulers`: context-length-aware curriculum planning and knapsack
  allocation.
- `gct.sae`: SAE feature extraction, sparse similarity metrics, solved-anchor
  neighbor construction, and top-1 SAE transfer.
- `gct.runtime`: GPU detection, model-parallel estimates, queue/backlog
  controller, and execution backends.
- `gct.training`: model training loop and batched ablation/generation loop.
- `gct.telemetry`: asynchronous latency/throughput metrics and optional W&B.
- `airflow/dags`: orchestration DAG that submits work, not GPU compute.
- `slurm`: batch wrappers that call the same CLI.

## Airflow Role

Airflow should handle dependencies, retries, scheduling, and auditability. It is
not an accelerator and should not directly run long GPU inference loops. In this
design, Airflow triggers ETL/planning and submits SLURM/cloud work. The GPU job
then writes resumable JSONL outputs; Airflow later summarizes those outputs.

## Backpressure And Scaling

`QueueController` tracks arrival rate and service rate from cheap timestamp
samples. If utilization approaches the scale threshold and backlog exists, the
orchestrator can request additional shards/workers before saturation. If
utilization approaches the throttle threshold, producers should slow down or
reduce batch admission.

## Knapsack Allocation

Context length is treated as the resource cost. Priority can encode difficulty,
expected value, solved-anchor quality, or deadline. The planner selects work
under a token budget so the first sweep does not admit a set of long prompts
that saturates GPU memory and leaves workers backlogged.

Knapsack allocation controls which target tasks are admitted. It does not choose
anchors. SAE neighbor construction chooses anchors from solved tasks using
feature-space similarity.

## Model Parallelism

The scaffold estimates model memory for Llama-class models from parameter count
and dtype. For Llama 3 8B, fp16/bf16 weights plus overhead usually fit on a
single 24GB+ GPU; smaller GPUs or larger models should use tensor parallelism
or mixed tensor/pipeline parallelism. The estimate is intentionally explicit so
the chosen backend can map it to Accelerate, DeepSpeed, vLLM, or another
launcher.
