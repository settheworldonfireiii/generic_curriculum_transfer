import sys
from types import SimpleNamespace

from gct.config.schema import ExperimentConfig, TelemetryConfig
from gct.telemetry.wandb import WandbRun


def test_wandb_run_logs_in_with_api_key_without_exposing_key(monkeypatch) -> None:
    calls = {"login": [], "init": []}

    class FakeRun:
        def __init__(self) -> None:
            self.logged = []
            self.finished = False

        def log(self, payload, step=None):
            self.logged.append((payload, step))

        def finish(self):
            self.finished = True

    fake_run = FakeRun()

    fake_wandb = SimpleNamespace(
        login=lambda key=None, relogin=False: calls["login"].append({"key": key, "relogin": relogin}),
        init=lambda **kwargs: (calls["init"].append(kwargs), fake_run)[1],
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = ExperimentConfig(
        telemetry=TelemetryConfig(
            wandb_project="project",
            wandb_entity="entity",
            wandb_mode="online",
            wandb_api_key="secret-key",
        )
    )

    with WandbRun(config, "sweep") as run:
        run.log({"x": 1}, step=2)

    assert calls["login"] == [{"key": "secret-key", "relogin": True}]
    assert calls["init"][0]["project"] == "project"
    assert calls["init"][0]["entity"] == "entity"
    assert calls["init"][0]["name"] == "sweep"
    assert "secret-key" not in str(calls["init"][0]["config"])
    assert fake_run.logged == [({"x": 1}, 2)]
    assert fake_run.finished

