#!/usr/bin/env bash
#SBATCH --job-name=gct-sae
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/gct-sae-%j.log

set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
SUMMARY="${SUMMARY:-runs/competitive_math/sweep/summary.csv}"
ANCHOR_SOLUTIONS="${ANCHOR_SOLUTIONS:-runs/competitive_math/sweep/raw.jsonl}"
TOKEN_CAPACITY="${TOKEN_CAPACITY:-32000}"
VARIANT="${VARIANT:-cos07}"
LAYER_REGIME="${LAYER_REGIME:-combo}"

mkdir -p logs
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-arc-sae-sweep}"

gct --config "$CONFIG" prepare-data
gct --config "$CONFIG" run-sweep --samples-per-task "${SWEEP_SAMPLES_PER_TASK:-3}"
gct --config "$CONFIG" plan-curriculum --token-capacity "$TOKEN_CAPACITY"
gct --config "$CONFIG" extract-sae
gct --config "$CONFIG" build-sae-neighbors \
  --summary "$SUMMARY" \
  --target-plan runs/competitive_math/plans/curriculum.jsonl \
  --variant "$VARIANT" \
  --layer-regime "$LAYER_REGIME"
gct --config "$CONFIG" run-sae-transfer \
  --neighbors "runs/competitive_math/sae/neighbors_${LAYER_REGIME}_${VARIANT}.csv" \
  --anchor-solutions "$ANCHOR_SOLUTIONS" \
  --samples-per-target "${SAMPLES_PER_TARGET:-3}"
