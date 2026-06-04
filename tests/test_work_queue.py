from __future__ import annotations

from pathlib import Path

from gct.runtime.work_queue import SqliteWorkQueue


def test_sqlite_work_queue_claims_by_priority_and_completes(tmp_path: Path) -> None:
    queue = SqliteWorkQueue(tmp_path / "work.sqlite")
    inserted = queue.enqueue_many(
        [
            {"target_task_id": "low", "combined_similarity": "0.1"},
            {"target_task_id": "high", "combined_similarity": "0.9"},
        ],
        id_field="target_task_id",
        priority_field="combined_similarity",
    )

    assert inserted == 2
    leases = queue.claim(limit=1, worker_id="worker", lease_seconds=60)
    assert [lease.item_id for lease in leases] == ["high"]

    queue.complete("high")
    stats = queue.stats()
    assert stats["done"] == 1
    assert stats["pending"] == 1


def test_sqlite_work_queue_reclaims_expired_leases(tmp_path: Path) -> None:
    queue = SqliteWorkQueue(tmp_path / "work.sqlite")
    queue.enqueue_many([{"target_task_id": "x"}], id_field="target_task_id")

    assert queue.claim(limit=1, worker_id="first", lease_seconds=-1)[0].item_id == "x"
    assert queue.claim(limit=1, worker_id="second", lease_seconds=60)[0].item_id == "x"
