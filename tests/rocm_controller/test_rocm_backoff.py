from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


def test_rocm_unknown_utilization_backs_off_when_threshold_enabled():
    assert RocmGPUController._should_run_batch(None, 10) is False


def test_rocm_unknown_utilization_runs_when_backoff_disabled():
    assert RocmGPUController._should_run_batch(None, -1) is True
