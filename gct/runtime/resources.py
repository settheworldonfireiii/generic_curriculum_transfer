from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int


@dataclass(frozen=True)
class ParallelismPlan:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    reason: str


def detect_gpus() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(parts[2]),
                memory_free_mb=int(parts[3]),
            )
        )
    return gpus


def estimate_model_memory_gb(model_name: str, dtype: str) -> float:
    lower = model_name.lower()
    if "70b" in lower:
        params_b = 70
    elif "13b" in lower:
        params_b = 13
    elif "8b" in lower or "llama-3" in lower:
        params_b = 8
    elif "7b" in lower:
        params_b = 7
    else:
        params_b = 8
    bytes_per_param = 4 if dtype == "fp32" else 2
    # Weights plus conservative KV/cache/runtime overhead.
    return params_b * 1e9 * bytes_per_param / 1e9 * 1.35


def estimate_parallelism(model_name: str, dtype: str, gpus: list[GpuInfo] | None = None) -> ParallelismPlan:
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        return ParallelismPlan(1, 1, "No GPUs detected; use single-process CPU/dev mode.")
    required_gb = estimate_model_memory_gb(model_name, dtype)
    free_per_gpu_gb = max(gpu.memory_free_mb for gpu in gpus) / 1024
    if required_gb < free_per_gpu_gb * 0.80:
        return ParallelismPlan(1, 1, f"Estimated {required_gb:.1f}GB fits on one GPU.")
    tp = min(len(gpus), max(1, int(required_gb // max(1.0, free_per_gpu_gb * 0.75)) + 1))
    if tp <= len(gpus):
        return ParallelismPlan(tp, 1, f"Estimated {required_gb:.1f}GB needs tensor parallelism across {tp} GPUs.")
    return ParallelismPlan(len(gpus), 2, "Model likely needs mixed tensor/pipeline parallelism.")


def write_resource_report(path: str, model_name: str, dtype: str) -> None:
    gpus = detect_gpus()
    plan = estimate_parallelism(model_name, dtype, gpus)
    with open(path, "w") as f:
        json.dump(
            {
                "gpus": [gpu.__dict__ for gpu in gpus],
                "parallelism": plan.__dict__,
                "estimated_model_memory_gb": estimate_model_memory_gb(model_name, dtype),
            },
            f,
            indent=2,
        )

