from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnapsackItem:
    item_id: str
    value: float
    cost: int
    priority: float = 1.0


def allocate_knapsack(items: list[KnapsackItem], capacity: int) -> list[KnapsackItem]:
    if capacity <= 0:
        return []
    scaled = [
        KnapsackItem(
            item_id=item.item_id,
            value=item.value * item.priority,
            cost=max(1, item.cost),
            priority=item.priority,
        )
        for item in items
    ]
    dp = [0.0] * (capacity + 1)
    keep: list[list[int]] = [[] for _ in range(capacity + 1)]
    for idx, item in enumerate(scaled):
        for cap in range(capacity, item.cost - 1, -1):
            candidate = dp[cap - item.cost] + item.value
            if candidate > dp[cap]:
                dp[cap] = candidate
                keep[cap] = keep[cap - item.cost] + [idx]
    selected_ids = set(keep[capacity])
    return [items[idx] for idx in keep[capacity] if idx in selected_ids]

