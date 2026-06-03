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

gct --config "$CONFIG" prepare-data
gct --config "$CONFIG" resource-report
gct --config "$CONFIG" plan-curriculum --token-capacity "$TOKEN_CAPACITY"
gct --config "$CONFIG" shard-plan --num-shards "${SHARDS:-4}"
gct --config "$CONFIG" run-ablation --samples-per-task "${SAMPLES_PER_TASK:-3}"
