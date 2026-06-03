# Generic Curriculum Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate, dataset-agnostic curriculum transfer scaffold with HF/PyTorch data loading, optional Spark/Airflow/SLURM/cloud integrations, telemetry, W&B, context-length allocation, and model-parallelism estimates.

**Architecture:** Plain Python CLI commands form the stable core. Orchestration layers call the CLI rather than duplicating logic. GPU-heavy work writes resumable JSONL artifacts so jobs can move between SLURM and cloud machines.

**Tech Stack:** Python, HF datasets, PyTorch DataLoader, Transformers, optional Spark, optional Airflow, optional W&B, SLURM shell wrappers.

---

### Task 1: Package Skeleton

**Files:**
- Create `pyproject.toml`
- Create `gct/config/*.py`
- Create `configs/*.yaml`

- [x] Define package metadata and optional extras.
- [x] Add dataclass config schema for dataset, model, runtime, backend, telemetry.
- [x] Add default Competitive MATH config and an example arbitrary HF dataset config.

### Task 2: Data And ETL

**Files:**
- Create `gct/data/hf_loader.py`
- Create `gct/data/torch_dataset.py`
- Create `gct/etl/local.py`
- Create `gct/etl/spark.py`

- [x] Normalize arbitrary HF rows into `task_id`, `prompt`, `answer`, `metadata`.
- [x] Save canonical JSONL.
- [x] Build PyTorch Dataset/DataLoader with multiple workers.
- [x] Add optional Spark preprocessing entrypoint.

### Task 3: Scheduling And Resources

**Files:**
- Create `gct/schedulers/knapsack.py`
- Create `gct/schedulers/curriculum.py`
- Create `gct/runtime/resources.py`
- Create `gct/runtime/queue_control.py`

- [x] Add context-length knapsack allocator.
- [x] Add curriculum plan writer.
- [x] Add GPU/resource detection and Llama-class memory estimates.
- [x] Add arrival/service utilization controller.

### Task 4: Execution Loops

**Files:**
- Create `gct/training/loop.py`
- Create `gct/training/ablation.py`
- Create `gct/telemetry/*.py`

- [x] Add supervised CausalLM training loop.
- [x] Add batched ablation/generation loop for intra-GPU utilization.
- [x] Add async JSONL metrics and optional W&B.

### Task 5: Orchestration

**Files:**
- Create `gct/cli.py`
- Create `slurm/*.sbatch`
- Create `airflow/dags/gct_pipeline_dag.py`

- [x] Add CLI commands for data prep, planning, sharding, training, ablation, resource report, and status.
- [x] Add SLURM pipeline and array wrappers.
- [x] Add Airflow DAG that orchestrates CLI/SLURM work.

### Task 6: Docs And Tests

**Files:**
- Create `README.md`
- Create `docs/architecture.md`
- Create `docs/etl.md`
- Create `tests/*.py`

- [x] Explain Airflow role, ETL flow, SLURM/cloud usage, telemetry, and parallelism.
- [x] Test config overrides, knapsack allocation, queue decisions, and sharding.

