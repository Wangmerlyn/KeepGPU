from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController


def test_macm_unknown_utilization_backs_off_when_threshold_enabled():
    assert MacMGPUController._should_run_batch(None, 10) is False


def test_macm_unknown_utilization_runs_when_backoff_disabled():
    assert MacMGPUController._should_run_batch(None, -1) is True
