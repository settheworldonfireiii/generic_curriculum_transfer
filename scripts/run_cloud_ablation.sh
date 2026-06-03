#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/competitive_math.yaml}"
SHARDS="${SHARDS:-4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-3}"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-arc-sae-sweep}"

for shard in $(seq 0 $((SHARDS - 1))); do
  gpu=$shard
  tasks="runs/competitive_math/plans/curriculum_shard${shard}of${SHARDS}.jsonl"
  raw_out="runs/competitive_math/ablation/shard${shard}of${SHARDS}_raw.jsonl"
  log="logs/cloud-ablate-shard${shard}of${SHARDS}.log"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES="$gpu" gct --config "$CONFIG" run-ablation \
    --tasks "$tasks" \
    --raw-out "$raw_out" \
    --samples-per-task "$SAMPLES_PER_TASK" \
    >> "$log" 2>&1 &
done

wait

