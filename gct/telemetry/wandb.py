from __future__ import annotations

from typing import Any

from gct.config.schema import ExperimentConfig


class WandbRun:
    def __init__(self, config: ExperimentConfig, run_name: str) -> None:
        self.config = config
        self.run_name = run_name
        self.run: Any | None = None

    def __enter__(self) -> "WandbRun":
        if not self.config.telemetry.wandb_project:
            return self
        try:
            import wandb
        except ImportError:
            return self
        self.run = wandb.init(
            project=self.config.telemetry.wandb_project,
            entity=self.config.telemetry.wandb_entity,
            name=self.run_name,
            mode=self.config.telemetry.wandb_mode,
            config={
                "dataset": self.config.dataset.name,
                "model": self.config.model.name,
                "backend": self.config.backend.kind,
            },
        )
        return self

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def __exit__(self, *args: object) -> None:
        if self.run is not None:
            self.run.finish()

