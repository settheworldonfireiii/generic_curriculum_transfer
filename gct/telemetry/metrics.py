from __future__ import annotations

import json
import queue
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


@dataclass
class Timer:
    metrics: "AsyncMetrics | NullMetrics"
    name: str
    tags: dict[str, Any] | None = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.metrics.observe(self.name, time.perf_counter() - self.start, self.tags)


class NullMetrics:
    def incr(self, name: str, value: float = 1.0, tags: dict[str, Any] | None = None) -> None:
        return None

    def observe(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        return None

    def timer(self, name: str, tags: dict[str, Any] | None = None) -> Timer:
        return Timer(self, name, tags)

    def close(self) -> None:
        return None


class AsyncMetrics:
    """Low-overhead metrics sink.

    Hot paths enqueue counters/observations only. A background thread aggregates
    and flushes JSONL records so latency measurement does not serialize GPU work.
    """

    def __init__(self, path: Path, flush_interval_s: float = 5.0) -> None:
        self.path = path
        self.flush_interval_s = flush_interval_s
        self.events: queue.SimpleQueue[tuple[str, str, float, dict[str, Any]]] = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def incr(self, name: str, value: float = 1.0, tags: dict[str, Any] | None = None) -> None:
        self.events.put(("counter", name, value, tags or {}))

    def observe(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        self.events.put(("timer", name, value, tags or {}))

    def timer(self, name: str, tags: dict[str, Any] | None = None) -> Timer:
        return Timer(self, name, tags)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.flush_interval_s + 1.0)
        self._flush_once()

    def _run(self) -> None:
        while not self.stop_event.wait(self.flush_interval_s):
            self._flush_once()

    def _flush_once(self) -> None:
        counters: dict[tuple[str, str], float] = defaultdict(float)
        observations: dict[tuple[str, str], list[float]] = defaultdict(list)
        while True:
            try:
                kind, name, value, tags = self.events.get_nowait()
            except queue.Empty:
                break
            key = (name, json.dumps(tags, sort_keys=True))
            if kind == "counter":
                counters[key] += value
            else:
                observations[key].append(value)

        if not counters and not observations:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        with self.path.open("a") as f:
            for (name, tags_json), value in counters.items():
                f.write(
                    json.dumps(
                        {
                            "ts": now,
                            "kind": "counter",
                            "name": name,
                            "value": value,
                            "tags": json.loads(tags_json),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            for (name, tags_json), values in observations.items():
                f.write(
                    json.dumps(
                        {
                            "ts": now,
                            "kind": "timer",
                            "name": name,
                            "count": len(values),
                            "mean": statistics.fmean(values),
                            "p50": statistics.median(values),
                            "max": max(values),
                            "tags": json.loads(tags_json),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

