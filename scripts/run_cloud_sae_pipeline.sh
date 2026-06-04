#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
SUMMARY="${SUMMARY:-runs/competitive_math/sweep/summary.csv}"
ANCHOR_SOLUTIONS="${ANCHOR_SOLUTIONS:-runs/competitive_math/sweep/raw.jsonl}"
TOKEN_CAPACITY="${TOKEN_CAPACITY:-32000}"
VARIANT="${VARIANT:-cos07}"
LAYER_REGIME="${LAYER_REGIME:-combo}"

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

gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" prepare-data
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" run-sweep --samples-per-task "${SWEEP_SAMPLES_PER_TASK:-3}"
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" resource-report
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" plan-curriculum --token-capacity "$TOKEN_CAPACITY"
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" extract-sae
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" build-sae-neighbors \
  --summary "$SUMMARY" \
  --target-plan runs/competitive_math/plans/curriculum.jsonl \
  --variant "$VARIANT" \
  --layer-regime "$LAYER_REGIME"
gct --config "$CONFIG" "${WANDB_ARGS[@]}" "${INFERENCE_ARGS[@]}" run-sae-transfer \
  --neighbors "runs/competitive_math/sae/neighbors_${LAYER_REGIME}_${VARIANT}.csv" \
  --anchor-solutions "$ANCHOR_SOLUTIONS" \
  --samples-per-target "${SAMPLES_PER_TARGET:-3}"
