from gct.schedulers.knapsack import KnapsackItem, allocate_knapsack


def test_allocate_knapsack_respects_capacity_and_priority() -> None:
    items = [
        KnapsackItem("long_low_priority", value=10, cost=10, priority=0.1),
        KnapsackItem("short_high_priority", value=3, cost=2, priority=3.0),
        KnapsackItem("medium", value=4, cost=3, priority=1.0),
    ]

    selected = allocate_knapsack(items, capacity=5)

    assert {item.item_id for item in selected} == {"short_high_priority", "medium"}
    assert sum(item.cost for item in selected) <= 5

