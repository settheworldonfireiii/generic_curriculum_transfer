from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class QueueDecision:
    arrival_rate: float
    service_rate: float
    utilization: float
    backlog: int
    should_scale_out: bool
    should_throttle: bool


class QueueController:
    def __init__(self, window: int = 128, scale_threshold: float = 0.78, throttle_threshold: float = 0.95) -> None:
        self.arrivals: deque[float] = deque(maxlen=window)
        self.services: deque[float] = deque(maxlen=window)
        self.scale_threshold = scale_threshold
        self.throttle_threshold = throttle_threshold

    def record_arrival(self, timestamp: float) -> None:
        self.arrivals.append(timestamp)

    def record_service(self, timestamp: float) -> None:
        self.services.append(timestamp)

    def decide(self, backlog: int) -> QueueDecision:
        arrival_rate = _rate(self.arrivals)
        service_rate = _rate(self.services)
        utilization = arrival_rate / service_rate if service_rate > 0 else 0.0
        return QueueDecision(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            utilization=utilization,
            backlog=backlog,
            should_scale_out=backlog > 0 and utilization >= self.scale_threshold,
            should_throttle=utilization >= self.throttle_threshold,
        )


def _rate(samples: deque[float]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1] - samples[0]
    if elapsed <= 0:
        return 0.0
    return (len(samples) - 1) / elapsed

