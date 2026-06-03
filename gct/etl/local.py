from __future__ import annotations

import time
from pathlib import Path

from gct.config.schema import ExperimentConfig
from gct.data.hf_loader import load_hf_rows
from gct.utils.jsonl import write_jsonl


def prepare_dataset_local(config: ExperimentConfig) -> Path:
    start = time.perf_counter()
    output = config.runtime.output_dir / "datasets" / "tasks.jsonl"
    rows = list(load_hf_rows(config.dataset))
    write_jsonl(output, rows)
    manifest = config.runtime.output_dir / "datasets" / "manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            [
                f"dataset={config.dataset.name}",
                f"split={config.dataset.split}",
                f"rows={len(rows)}",
                f"format={config.dataset.saved_format}",
                f"elapsed_s={time.perf_counter() - start:.3f}",
            ]
        )
        + "\n"
    )
    return output

