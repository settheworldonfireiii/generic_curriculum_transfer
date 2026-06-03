from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = os.environ.get("GCT_PROJECT_DIR", os.path.expanduser("~/generic_curriculum_transfer"))
CONFIG = os.environ.get("GCT_CONFIG", "configs/competitive_math.yaml")
SUMMARY = os.environ.get("GCT_SUMMARY", "runs/competitive_math/sweep/summary.csv")
ANCHOR_SOLUTIONS = os.environ.get("GCT_ANCHOR_SOLUTIONS", "runs/competitive_math/sweep/raw.jsonl")
VARIANT = os.environ.get("GCT_SAE_VARIANT", "cos07")
LAYER_REGIME = os.environ.get("GCT_LAYER_REGIME", "combo")

default_args = {"owner": "gct", "retries": 1}

with DAG(
    dag_id="generic_curriculum_transfer_sae",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["gpu", "sae", "curriculum", "transfer"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} prepare-data",
    )

    run_sweep = BashOperator(
        task_id="run_sweep",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} run-sweep --samples-per-task 3",
    )

    plan_curriculum = BashOperator(
        task_id="plan_curriculum",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} plan-curriculum --token-capacity 32000",
    )

    extract_sae = BashOperator(
        task_id="extract_sae",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} extract-sae",
    )

    build_neighbors = BashOperator(
        task_id="build_neighbors",
        bash_command=(
            f"cd {PROJECT_DIR} && gct --config {CONFIG} build-sae-neighbors "
            f"--summary {SUMMARY} "
            " --target-plan runs/competitive_math/plans/curriculum.jsonl "
            f"--variant {VARIANT} --layer-regime {LAYER_REGIME}"
        ),
    )

    run_transfer = BashOperator(
        task_id="run_sae_transfer",
        bash_command=(
            f"cd {PROJECT_DIR} && gct --config {CONFIG} run-sae-transfer "
            f"--neighbors runs/competitive_math/sae/neighbors_{LAYER_REGIME}_{VARIANT}.csv "
            f"--anchor-solutions {ANCHOR_SOLUTIONS}"
        ),
    )

    prepare_data >> run_sweep >> plan_curriculum >> extract_sae >> build_neighbors >> run_transfer
