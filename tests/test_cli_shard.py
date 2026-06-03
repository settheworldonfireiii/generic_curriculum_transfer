from pathlib import Path

from gct.cli import shard_plan
from gct.utils.jsonl import read_jsonl, write_jsonl


def test_shard_plan_uses_stable_modulo_assignment(tmp_path: Path) -> None:
    plan = tmp_path / "plan.jsonl"
    rows = [{"task_id": f"task_{idx}", "prompt": "p", "answer": "a"} for idx in range(5)]
    write_jsonl(plan, rows)

    shard_plan(plan, tmp_path, num_shards=2)

    shard0 = list(read_jsonl(tmp_path / "curriculum_shard0of2.jsonl"))
    shard1 = list(read_jsonl(tmp_path / "curriculum_shard1of2.jsonl"))
    assert [row["task_id"] for row in shard0] == ["task_0", "task_2", "task_4"]
    assert [row["task_id"] for row in shard1] == ["task_1", "task_3"]

