from pathlib import Path

from gct.config.load import load_config


def test_load_config_with_nested_overrides() -> None:
    config = load_config(
        Path("configs/competitive_math.yaml"),
        {"dataset": {"name": "gsm8k", "prompt_column": "question"}, "runtime": {"batch_size": 2}},
    )

    assert config.dataset.name == "gsm8k"
    assert config.dataset.prompt_column == "question"
    assert config.dataset.answer_column == "solution"
    assert config.runtime.batch_size == 2

