from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from gct.config.schema import ExperimentConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal bootstrap envs
    yaml = None


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with path.open() as f:
            if yaml is not None:
                raw = yaml.safe_load(f) or {}
            else:
                raw = _simple_yaml_load(f.read())
    if overrides:
        raw = deep_update(raw, overrides)
    return ExperimentConfig.from_mapping(raw)


def dump_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("Install pyyaml to dump config files.")
    with path.open("w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Tiny two-level YAML reader for bootstrap configs.

    This keeps CLI/help/status usable before optional dependencies are installed.
    Full YAML support is provided by PyYAML in the normal environment.
    """

    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            root[key] = {}
            current = root[key]
            continue
        if current is None:
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
