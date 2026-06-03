#!/usr/bin/env bash
#SBATCH --job-name=gct-pipeline
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/gct-pipeline-%j.log

set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
TOKEN_CAPACITY="${TOKEN_CAPACITY:-32000}"

mkdir -p logs
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-arc-sae-sweep}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(--wandb-api-key "$WANDB_API_KEY" --wandb-mode "${WANDB_MODE:-online}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

gct --config "$CONFIG" "${WANDB_ARGS[@]}" prepare-data
gct --config "$CONFIG" "${WANDB_ARGS[@]}" resource-report
gct --config "$CONFIG" "${WANDB_ARGS[@]}" plan-curriculum --token-capacity "$TOKEN_CAPACITY"
gct --config "$CONFIG" "${WANDB_ARGS[@]}" shard-plan --num-shards "${SHARDS:-4}"
gct --config "$CONFIG" "${WANDB_ARGS[@]}" run-ablation --samples-per-task "${SAMPLES_PER_TASK:-3}"
