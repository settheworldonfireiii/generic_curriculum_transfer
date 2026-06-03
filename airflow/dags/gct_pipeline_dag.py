from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = os.environ.get("GCT_PROJECT_DIR", os.path.expanduser("~/generic_curriculum_transfer"))
CONFIG = os.environ.get("GCT_CONFIG", "configs/competitive_math.yaml")

default_args = {"owner": "gct", "retries": 1}

with DAG(
    dag_id="generic_curriculum_transfer",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["gpu", "curriculum", "transfer"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} prepare-data",
    )

    resource_report = BashOperator(
        task_id="resource_report",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} resource-report",
    )

    plan_curriculum = BashOperator(
        task_id="plan_curriculum",
        bash_command=f"cd {PROJECT_DIR} && gct --config {CONFIG} plan-curriculum --token-capacity 32000",
    )

    submit_slurm_ablation = BashOperator(
        task_id="submit_slurm_ablation",
        bash_command=f"cd {PROJECT_DIR} && sbatch slurm/run_ablation_array.sbatch",
    )

    summarize_status = BashOperator(
        task_id="summarize_status",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "for f in runs/competitive_math/ablation/*_raw.jsonl; do "
            "gct status --raw \"$f\" || true; "
            "done"
        ),
    )

    prepare_data >> resource_report >> plan_curriculum >> submit_slurm_ablation >> summarize_status

