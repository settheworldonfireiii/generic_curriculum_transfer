#!/usr/bin/env bash
#SBATCH --job-name=gct-ablate
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gct-ablate-%A-%a.log

set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
SHARDS="${SHARDS:-4}"

mkdir -p logs
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-arc-sae-sweep}"

TASKS="${TASKS:-runs/competitive_math/plans/curriculum_shard${SLURM_ARRAY_TASK_ID}of${SHARDS}.jsonl}"
RAW_OUT="${RAW_OUT:-runs/competitive_math/ablation/shard${SLURM_ARRAY_TASK_ID}of${SHARDS}_raw.jsonl}"

INFERENCE_ARGS=()
if [[ -n "${INFERENCE_BACKEND:-}" ]]; then
  INFERENCE_ARGS+=(--inference-backend "$INFERENCE_BACKEND")
fi
if [[ -n "${SGLANG_BASE_URL:-}" ]]; then
  INFERENCE_ARGS+=(--sglang-base-url "$SGLANG_BASE_URL")
fi
if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  INFERENCE_ARGS+=(--sglang-api-key "$SGLANG_API_KEY")
fi
if [[ -n "${SGLANG_MODEL:-}" ]]; then
  INFERENCE_ARGS+=(--sglang-model "$SGLANG_MODEL")
fi

gct --config "$CONFIG" "${INFERENCE_ARGS[@]}" run-ablation \
  --tasks "$TASKS" \
  --raw-out "$RAW_OUT" \
  --samples-per-task "${SAMPLES_PER_TASK:-3}"
