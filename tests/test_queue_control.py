from gct.runtime.queue_control import QueueController


def test_queue_controller_scales_before_saturation() -> None:
    controller = QueueController(scale_threshold=0.75, throttle_threshold=0.95)
    for ts in [0.0, 1.0, 2.0, 3.0]:
        controller.record_arrival(ts)
    for ts in [0.0, 0.833, 1.666, 2.5]:
        controller.record_service(ts)

    decision = controller.decide(backlog=10)

    assert decision.utilization > 0.75
    assert decision.should_scale_out
    assert not decision.should_throttle
