from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "EleutherAI/hendrycks_math"
    config_name: str | None = None
    split: str = "test"
    streaming: bool = False
    prompt_column: str = "problem"
    answer_column: str = "solution"
    id_column: str | None = None
    level_column: str | None = "level"
    category_column: str | None = "type"
    max_rows: int | None = None
    filter_expr: str | None = None
    preprocessing_engine: str = "hf"
    saved_format: str = "jsonl"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bf16"
    max_input_tokens: int = 7600
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path = Path("runs/default")
    num_workers: int = 4
    batch_size: int = 1
    generation_batch_size: int = 1
    prefetch_factor: int = 2
    resume: bool = True
    seed: int = 1729


@dataclass(frozen=True)
class BackendConfig:
    kind: str = "slurm"
    slurm_partition: str = "interactive-gpu"
    slurm_gres: str = "gpu:a40:4"
    slurm_time: str = "08:00:00"
    cloud_provider: str | None = None
    cloud_profile: str | None = None


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    flush_interval_s: float = 5.0
    wandb_project: str | None = "generic-curriculum-transfer"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"


@dataclass(frozen=True)
class SaeConfig:
    enabled: bool = True
    repo: str = "EleutherAI/sae-llama-3-8b-32x"
    layers: tuple[int, ...] = (10, 16, 24, 28, 30)
    combo_layers: tuple[int, ...] = (10, 16, 24, 30)
    top_features_per_layer: int = 256
    top_k_neighbors: int = 5
    default_variant: str = "cos07"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    sae: SaeConfig = field(default_factory=SaeConfig)

    @staticmethod
    def from_mapping(raw: dict[str, Any]) -> "ExperimentConfig":
        dataset = DatasetConfig(**raw.get("dataset", {}))
        model = ModelConfig(**raw.get("model", {}))
        runtime_raw = dict(raw.get("runtime", {}))
        if "output_dir" in runtime_raw:
            runtime_raw["output_dir"] = Path(runtime_raw["output_dir"])
        runtime = RuntimeConfig(**runtime_raw)
        backend = BackendConfig(**raw.get("backend", {}))
        telemetry = TelemetryConfig(**raw.get("telemetry", {}))
        sae_raw = dict(raw.get("sae", {}))
        for key in ("layers", "combo_layers"):
            if key in sae_raw and isinstance(sae_raw[key], list):
                sae_raw[key] = tuple(sae_raw[key])
        sae = SaeConfig(**sae_raw)
        return ExperimentConfig(
            dataset=dataset,
            model=model,
            runtime=runtime,
            backend=backend,
            telemetry=telemetry,
            sae=sae,
        )
