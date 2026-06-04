from __future__ import annotations

import json
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkLease:
    item_id: str
    payload: dict[str, Any]
    attempts: int


class SqliteWorkQueue:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS work_items (
                    item_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority REAL NOT NULL DEFAULT 0.0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_until REAL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_work_status ON work_items(status, priority)")
            conn.commit()

    def enqueue_many(
        self,
        rows: list[dict[str, Any]],
        id_field: str,
        priority_field: str | None = None,
    ) -> int:
        self.initialize()
        now = time.time()
        inserted = 0
        with self._connect() as conn:
            for row in rows:
                item_id = str(row[id_field])
                priority = _float_or_zero(row.get(priority_field)) if priority_field else 0.0
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO work_items
                    (item_id, payload_json, status, priority, attempts, created_at, updated_at)
                    VALUES (?, ?, 'pending', ?, 0, ?, ?)
                    """,
                    (item_id, json.dumps(row, ensure_ascii=True), priority, now, now),
                )
                inserted += cursor.rowcount
            conn.commit()
        return inserted

    def claim(self, limit: int, worker_id: str, lease_seconds: float) -> list[WorkLease]:
        self.initialize()
        now = time.time()
        lease_until = now + lease_seconds
        limit = max(1, limit)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT item_id, payload_json, attempts
                FROM work_items
                WHERE status = 'pending'
                   OR (status = 'leased' AND lease_until IS NOT NULL AND lease_until < ?)
                ORDER BY priority DESC, attempts ASC, created_at ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            leases: list[WorkLease] = []
            for item_id, payload_json, attempts in rows:
                conn.execute(
                    """
                    UPDATE work_items
                    SET status = 'leased',
                        attempts = attempts + 1,
                        lease_owner = ?,
                        lease_until = ?,
                        updated_at = ?
                    WHERE item_id = ?
                    """,
                    (worker_id, lease_until, now, item_id),
                )
                leases.append(WorkLease(item_id=item_id, payload=json.loads(payload_json), attempts=int(attempts) + 1))
            conn.commit()
        return leases

    def complete(self, item_id: str) -> None:
        self._update_status(item_id, "done", None)

    def fail(self, item_id: str, error: str, max_attempts: int = 3) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT attempts FROM work_items WHERE item_id = ?", (item_id,)).fetchone()
            status = "failed" if row and int(row[0]) >= max_attempts else "pending"
            now = time.time()
            conn.execute(
                """
                UPDATE work_items
                SET status = ?, lease_owner = NULL, lease_until = NULL, error = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (status, error[-2000:], now, item_id),
            )
            conn.commit()

    def stats(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as conn:
            counts = {
                status: count
                for status, count in conn.execute(
                    "SELECT status, COUNT(*) FROM work_items GROUP BY status"
                ).fetchall()
            }
            total = sum(counts.values())
            leased_expired = conn.execute(
                "SELECT COUNT(*) FROM work_items WHERE status = 'leased' AND lease_until < ?",
                (time.time(),),
            ).fetchone()[0]
        return {
            "queue": str(self.path),
            "total": total,
            "pending": counts.get("pending", 0),
            "leased": counts.get("leased", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "expired_leases": int(leased_expired),
        }

    def _update_status(self, item_id: str, status: str, error: str | None) -> None:
        with self._connect() as conn:
            now = time.time()
            conn.execute(
                """
                UPDATE work_items
                SET status = ?, lease_owner = NULL, lease_until = NULL, error = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (status, error, now, item_id),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
