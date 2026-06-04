#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
SHARDS="${SHARDS:-4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-3}"

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

for shard in $(seq 0 $((SHARDS - 1))); do
  gpu=$shard
  tasks="runs/competitive_math/plans/curriculum_shard${shard}of${SHARDS}.jsonl"
  raw_out="runs/competitive_math/ablation/shard${shard}of${SHARDS}_raw.jsonl"
  log="logs/cloud-ablate-shard${shard}of${SHARDS}.log"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES="$gpu" gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" run-ablation \
    --tasks "$tasks" \
    --raw-out "$raw_out" \
    --samples-per-task "$SAMPLES_PER_TASK" \
    >> "$log" 2>&1 &
done

wait
