# Generic Curriculum Transfer

Hybrid scaffold for dataset-agnostic curriculum transfer experiments. It uses
the current ARC/MATH workflow as the basis, but separates data preparation,
curriculum planning, execution, telemetry, and orchestration so the same run can
move between SLURM and cloud machines.

## What Airflow Does

Airflow is the workflow scheduler and audit log. It should decide that
`prepare-data` runs before `plan-curriculum`, and that ablation jobs start only
after shard manifests exist. It should not own the GPU process. GPU-heavy work
is submitted to SLURM or a cloud VM, then Airflow records status and retries
failed orchestration steps.

## Pipeline

```text
Raw HF dataset
  ↓
HF datasets loader, or optional Spark stage for large preprocessing
  ↓
Saved normalized dataset: task_id / prompt / answer / metadata
  ↓
PyTorch Dataset + DataLoader
  ↓
Model sweep and generic solved/unsolved summary
  ↓
Curriculum planner with context-length knapsack allocation
  ↓
Optional SAE feature extraction and solved-anchor neighbor schedule
  ↓
GPU runners: SAE transfer, training, or batched ablation/generation
  ↓
Async telemetry + W&B + JSONL summaries
```

## Install

```bash
cd generic_curriculum_transfer
python -m pip install -e ".[wandb,sae,spark]"
```

For SGLang serving support on a GPU host:

```bash
python -m pip install -e ".[sglang]"
```

For cloud:

```bash
chmod +x scripts/install_cloud_env.sh
./scripts/install_cloud_env.sh
conda activate arc-sae-sweep
```

To include SGLang in the cloud environment:

```bash
INSTALL_SGLANG=1 ./scripts/install_cloud_env.sh
```

## Default Competitive MATH Run

```bash
gct --config configs/competitive_math.yaml prepare-data
gct --config configs/competitive_math.yaml run-sweep --samples-per-task 3
gct --config configs/competitive_math.yaml resource-report
gct --config configs/competitive_math.yaml plan-curriculum --token-capacity 32000
gct --config configs/competitive_math.yaml shard-plan --num-shards 4
gct --config configs/competitive_math.yaml run-ablation \
  --tasks runs/competitive_math/plans/curriculum_shard0of4.jsonl \
  --raw-out runs/competitive_math/ablation/shard0of4_raw.jsonl
```

## Exact First-Project MATH Compatibility

Use these commands when you want the same Competition-MATH SAE workflow as the
original `arc_sae_transfer_experiment`: same dataset/config list, prompt
templates, answer extraction, solved-anchor neighbor formulas, transfer prompt,
raw/summary filenames, static sharding, and `--resume` behavior.

These commands are independent of YAML config files.

Sharded sweep plus SAE extraction:

```bash
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard gct math-sweep-sae \
    --out-dir "results_math_shard${shard}" \
    --num-shards 4 \
    --shard-index "$shard" \
    --resume \
    > "sweep${shard}.log" 2>&1 &
done
wait
```

Merge the sweep shards and build all four neighbor files:

```bash
gct math-merge-sweep-shards \
  --shard-dirs results_math_shard0 results_math_shard1 results_math_shard2 results_math_shard3 \
  --out-dir results_math

gct math-build-neighbors --out-dir results_math --top-k 5
```

Run transfer for one regime/variant on four GPUs:

```bash
run_math_transfer_variant () {
  local regime="$1"
  local variant="$2"
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard gct math-transfer \
      --out-dir results_math \
      --neighbors "results_math/neighbors_${regime}_${variant}.csv" \
      --raw-out "results_math/transfer_${regime}_${variant}_shard${shard}of4_raw.jsonl" \
      --summary-out "results_math/transfer_${regime}_${variant}_shard${shard}of4_summary.csv" \
      --num-shards 4 \
      --shard-index "$shard" \
      --resume \
      > "transfer_${regime}_${variant}_${shard}.log" 2>&1 &
  done
  wait
  gct math-merge-transfer-shards \
    --out-dir results_math \
    --regime "$regime" \
    --variant "$variant"
}

run_math_transfer_variant combo cos07
run_math_transfer_variant combo cos01
run_math_transfer_variant layer28 cos07
run_math_transfer_variant layer28 cos01
```

Run context ablations for a merged transfer summary:

```bash
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard gct math-context-ablation \
    --out-dir results_math \
    --transfer-summary results_math/transfer_combo_cos07_summary.csv \
    --raw-out "results_math/ablation_combo_cos07_shard${shard}of4_raw.jsonl" \
    --summary-out "results_math/ablation_combo_cos07_shard${shard}of4_summary.csv" \
    --run-groups 10 \
    --samples-per-group 3 \
    --num-shards 4 \
    --shard-index "$shard" \
    --resume \
    >> "ablate_combo_cos07_${shard}.log" 2>&1 &
done
wait
```

The exact compatibility outputs are:

```text
results_math/math_tasks.jsonl
results_math/math_sweep_raw.jsonl
results_math/math_sweep_summary.csv
results_math/math_solved_ids.txt
results_math/math_sae_task_layer_rows.jsonl
results_math/neighbors_{combo,layer28}_{cos07,cos01}.csv
results_math/transfer_<regime>_<variant>_summary.csv
results_math/ablation_<regime>_<variant>_shard<idx>of<num>_raw.jsonl
```

## W&B

W&B is optional and defaults to offline mode. To log online, pass the API key as
a global argument before the subcommand:

```bash
gct --config configs/competitive_math.yaml \
  --wandb-api-key "$WANDB_API_KEY" \
  --wandb-mode online \
  --wandb-project generic-curriculum-transfer \
  run-sweep --samples-per-task 3
```

The same global flags work for every W&B-enabled command:

```text
run-train
run-sweep
run-ablation
run-sae-transfer
```

For shell wrappers, export the key and project:

```bash
export WANDB_API_KEY=...
export WANDB_MODE=online
export WANDB_PROJECT=generic-curriculum-transfer
./scripts/run_cloud_sae_pipeline.sh
```

`--wandb-log-interval N` controls how often generation loops log per-attempt
metrics. The default is every 20 generations.

## SAE Transfer Path

The SAE path mirrors the previous ARC/MATH experiment:

1. Run a sweep to produce solved anchors and unsolved targets.
2. Extract pooled SAE task features at configured layers.
3. Build nearest-neighbor schedules from unsolved targets to solved anchors.
4. Run transfer with the top-1 solved anchor.

Knapsack allocation is used only to decide which **target tasks** enter the run.
It does not choose anchors. Anchors are still chosen by SAE similarity.

Expected summary input:

```csv
task_id,solved,level,category
task_a,True,2,Algebra
task_b,False,3,Geometry
```

Commands:

```bash
gct --config configs/competitive_math.yaml prepare-data
gct --config configs/competitive_math.yaml run-sweep --samples-per-task 3
gct --config configs/competitive_math.yaml plan-curriculum --token-capacity 32000
gct --config configs/competitive_math.yaml extract-sae

gct --config configs/competitive_math.yaml build-sae-neighbors \
  --summary runs/competitive_math/sweep/summary.csv \
  --target-plan runs/competitive_math/plans/curriculum.jsonl \
  --variant cos07 \
  --layer-regime combo

gct --config configs/competitive_math.yaml run-sae-transfer \
  --neighbors runs/competitive_math/sae/neighbors_combo_cos07.csv \
  --anchor-solutions runs/competitive_math/sweep/raw.jsonl
```

## SGLang Inference Backend

`run-sweep`, `run-ablation`, and `run-sae-transfer` can use an SGLang
OpenAI-compatible server instead of loading a local Hugging Face model in each
runner. SAE extraction still uses local torch/HF because it needs hidden states
and SAE activations.

Start a single-GPU SGLang server:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Then point GCT at it:

```bash
gct --config configs/competitive_math.yaml \
  --inference-backend sglang \
  --sglang-base-url http://127.0.0.1:30000/v1 \
  run-sweep --samples-per-task 3
```

For replicated serving, launch SGLang with its data-parallel/gateway mode and
keep the same GCT client flags. For shell wrappers:

```bash
export INFERENCE_BACKEND=sglang
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
./scripts/run_cloud_sae_pipeline.sh
```

## Dynamic SAE-Transfer Queue

Static sharding is still available, but SAE transfer can now run from a SQLite
work queue. This is useful with SGLang because many lightweight workers can
claim target-anchor rows dynamically while the SGLang server handles GPU-side
batching/routing.

```bash
gct --config configs/competitive_math.yaml init-transfer-work-queue \
  --neighbors runs/competitive_math/sae/neighbors_combo_cos07.csv

gct --config configs/competitive_math.yaml work-queue-status

gct --config configs/competitive_math.yaml \
  --inference-backend sglang \
  --sglang-base-url http://127.0.0.1:30000/v1 \
  run-dynamic-sae-transfer \
  --anchor-solutions runs/competitive_math/sweep/raw.jsonl \
  --claim-size 4 \
  --lease-seconds 900
```

Run multiple workers against the same queue to balance uneven target lengths:

```bash
for worker in 0 1 2 3; do
  gct --config configs/competitive_math.yaml \
    --inference-backend sglang \
    --sglang-base-url http://127.0.0.1:30000/v1 \
    run-dynamic-sae-transfer \
    --anchor-solutions runs/competitive_math/sweep/raw.jsonl \
    --worker-id "worker-$worker" \
    >> "logs/dynamic-transfer-$worker.log" 2>&1 &
done
wait
```

Default SAE scoring variants:

```text
cos07:        0.70 * SAE cosine + 0.15 * weighted_jaccard + 0.15 * tanimoto
cos01:        0.10 * SAE cosine + 0.45 * weighted_jaccard + 0.45 * tanimoto
arc_weighted: 0.70 * SAE cosine + 0.20 * weighted_jaccard + 0.10 * metadata structure cosine
```

## Command Catalog

All commands use the same entrypoint:

```bash
gct --config configs/competitive_math.yaml <command>
```

Dataset preparation:

```bash
gct --config configs/competitive_math.yaml prepare-data
gct --config configs/competitive_math.yaml prepare-data --engine spark
```

Sweep/evaluation:

```bash
gct --config configs/competitive_math.yaml run-sweep --samples-per-task 3
gct --config configs/competitive_math.yaml run-sweep \
  --tasks runs/competitive_math/datasets/tasks.jsonl \
  --raw-out runs/competitive_math/sweep/raw.jsonl \
  --summary-out runs/competitive_math/sweep/summary.csv \
  --solved-ids-out runs/competitive_math/sweep/solved_ids.txt \
  --samples-per-task 3
```

Curriculum planning and sharding:

```bash
gct --config configs/competitive_math.yaml plan-curriculum --token-capacity 32000
gct --config configs/competitive_math.yaml shard-plan --num-shards 4
```

SAE extraction and neighbor construction:

```bash
gct --config configs/competitive_math.yaml extract-sae
gct --config configs/competitive_math.yaml extract-sae \
  --layers 10 16 24 28 30 \
  --top-features-per-layer 256

gct --config configs/competitive_math.yaml build-sae-neighbors \
  --summary runs/competitive_math/sweep/summary.csv \
  --target-plan runs/competitive_math/plans/curriculum.jsonl \
  --variant cos07 \
  --layer-regime combo
```

SAE transfer:

```bash
gct --config configs/competitive_math.yaml run-sae-transfer \
  --neighbors runs/competitive_math/sae/neighbors_combo_cos07.csv \
  --anchor-solutions runs/competitive_math/sweep/raw.jsonl \
  --samples-per-target 3
```

Generic ablations:

```bash
gct --config configs/competitive_math.yaml run-ablation \
  --tasks runs/competitive_math/plans/curriculum.jsonl \
  --raw-out runs/competitive_math/ablation/raw.jsonl \
  --samples-per-task 3
```

Training loop:

```bash
gct --config configs/competitive_math.yaml run-train \
  --tasks runs/competitive_math/plans/curriculum.jsonl \
  --max-steps 100
```

Resource/status commands:

```bash
gct --config configs/competitive_math.yaml resource-report
gct status --raw runs/competitive_math/sweep/raw.jsonl
gct --config configs/competitive_math.yaml work-queue-status
```

Cloud/SLURM wrappers:

```bash
sbatch slurm/run_pipeline.sbatch
sbatch slurm/run_sae_pipeline.sbatch
sbatch --array=0-3 slurm/run_ablation_array.sbatch

./scripts/run_cloud_ablation.sh
./scripts/run_cloud_sae_pipeline.sh
```

## Any Hugging Face Dataset

Pass the dataset, split, and column mappings:

```bash
gct --config configs/example_any_hf_dataset.yaml prepare-data \
  --dataset gsm8k \
  --dataset-config main \
  --split test \
  --prompt-column question \
  --answer-column answer \
  --max-rows 100
```

The same run can override model and runtime parameters:

```bash
gct --config configs/example_any_hf_dataset.yaml \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype fp16 \
  --output-dir runs/gsm8k_llama3 \
  prepare-data \
  --dataset gsm8k \
  --dataset-config main \
  --split test \
  --prompt-column question \
  --answer-column answer
```

The phrase "any dataset" means any HF dataset whose rows can be mapped into
`prompt` and `answer` fields. If a dataset has images, audio, multiple labels,
or nested conversations, add a small normalizer instead of forcing it through
the two-column interface.

## SLURM

Single pipeline:

```bash
sbatch slurm/run_pipeline.sbatch
```

Array ablations after `shard-plan`:

```bash
sbatch --array=0-3 slurm/run_ablation_array.sbatch
```

## Airflow

Place `airflow/dags/gct_pipeline_dag.py` in your Airflow DAG folder or set:

```bash
export GCT_PROJECT_DIR=$PWD
export GCT_CONFIG=configs/competitive_math.yaml
```

The DAG runs ETL, resource reporting, curriculum planning, submits the SLURM
array, and then summarizes raw JSONL outputs.

## Telemetry

Hot paths enqueue metrics into `runs/<name>/metrics/*.jsonl`; a background
thread flushes aggregates every few seconds. This measures latency,
arrival/service rates, and throughput without forcing the GPU loop to block on
logging. W&B is optional and defaults to offline mode.

Dataset loading is CPU-side by design: filesystems, JSON/HF decoding, Python
normalization, tokenization, worker prefetching, and pinned-memory staging all
happen before tensors are transferred to GPU. The GPU should spend its time on
model compute, not blocking on filesystem and Python parsing work.

## API Keys

Currently supported keys:

```text
WANDB_API_KEY / --wandb-api-key
  Required only for online W&B logging.

HF_TOKEN / huggingface-cli login
  Required for gated or private Hugging Face models/datasets, including
  Meta-Llama-3 checkpoints if your environment has not already authenticated.
```

No other API key is required by the current code path. IBM Cloud API keys are
only needed if you automate instance creation with IBM Cloud CLI; the included
cloud scripts assume the machine already exists and you are running over SSH.

## Parallelism

- Dataset loading uses PyTorch `DataLoader` workers.
- Ablations use batched generation via `runtime.generation_batch_size`.
- SLURM array jobs parallelize across GPUs/nodes.
- `resource-report` estimates whether the model likely fits on one GPU or
  should use tensor/pipeline parallelism. Actual TP/PP launchers can be wired to
  vLLM, DeepSpeed, Accelerate, or Megatron as a backend-specific extension.

## Spark

Spark is optional and intended for data too large for one process:

```bash
python -m pip install -e ".[spark]"
gct --config configs/competitive_math.yaml prepare-data --engine spark
```

The current Spark stage expects normalized JSONL from HF first, then performs
filtering/repartitioning. Extend `gct/etl/spark.py` for joins and aggregations
specific to a dataset.
