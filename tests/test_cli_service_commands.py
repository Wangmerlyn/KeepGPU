from typer.testing import CliRunner

from keep_gpu import cli

runner = CliRunner()


def test_start_command_uses_rpc(monkeypatch):
    called = {}

    def fake_ensure(host, port, auto_start=True):
        called["ensure"] = (host, port, auto_start)

    def fake_rpc(method, params, host, port):
        called["rpc"] = (method, params, host, port)
        return {"job_id": "job-123"}

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(
        cli.app,
        [
            "start",
            "--gpu-ids",
            "0,1",
            "--vram",
            "2GiB",
            "--interval",
            "60",
            "--busy-threshold",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert called["ensure"] == (
        cli.DEFAULT_SERVICE_HOST,
        cli.DEFAULT_SERVICE_PORT,
        True,
    )

    method, params, host, port = called["rpc"]
    assert method == "start_keep"
    assert params["gpu_ids"] == [0, 1]
    assert params["vram"] == "2GiB"
    assert params["interval"] == 60
    assert params["busy_threshold"] == 30
    assert host == cli.DEFAULT_SERVICE_HOST
    assert port == cli.DEFAULT_SERVICE_PORT


def test_stop_requires_job_id_or_all():
    result = runner.invoke(cli.app, ["stop"])
    assert result.exit_code == 1
    assert "Provide --job-id or use --all" in result.output


def test_blocking_mode_remains_default(monkeypatch):
    called = {}

    def fake_run(interval, gpu_ids, vram, legacy_threshold, busy_threshold):
        called["args"] = (interval, gpu_ids, vram, legacy_threshold, busy_threshold)

    monkeypatch.setattr(cli, "_run_blocking", fake_run)
    result = runner.invoke(
        cli.app,
        ["--interval", "120", "--gpu-ids", "0", "--vram", "1GiB"],
    )

    assert result.exit_code == 0
    assert called["args"] == (120, "0", "1GiB", None, -1)


def test_start_prints_dashboard_and_stop_hints(monkeypatch):
    def fake_ensure(host, port, auto_start=True):
        return True

    def fake_rpc(method, params, host, port):
        return {"job_id": "job-abc"}

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 0
    assert "Auto-started KeepGPU service" in result.output
    assert "Dashboard:" in result.output
    assert "keep-gpu service-stop" in result.output


def test_service_stop_requires_managed_pid(monkeypatch):
    monkeypatch.setattr(cli, "_service_available", lambda host, port: False)
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: False)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "No managed daemon PID found" in result.output


def test_service_stop_refuses_active_sessions_without_force(monkeypatch):
    monkeypatch.setattr(cli, "_service_available", lambda host, port: True)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda method, params, host, port, timeout=8.0: (
            {"active_jobs": [{"job_id": "j1"}]}
            if method == "status"
            else {"stopped": []}
        ),
    )
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: True)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "Active keep sessions detected" in result.output


def test_stop_handles_service_timeout_without_traceback(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        raise RuntimeError(
            "Cannot reach KeepGPU service at http://127.0.0.1:8765/rpc: timed out"
        )

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: None)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 1
    assert "Cannot reach KeepGPU service" in result.output
    assert "service-stop --force" in result.output
    assert "Traceback" not in result.output


def test_cli_module_avoids_eager_gpu_imports():
    assert not hasattr(cli, "GlobalGPUController")
    assert not hasattr(cli, "KeepGPUServer")


def test_stop_all_fallback_force_stops_managed_daemon(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        raise RuntimeError("timed out")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 1234)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: True)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 0
    assert "force-stopped local daemon" in result.output


def test_service_stop_force_skips_rpc(monkeypatch):
    called = {"rpc": 0}

    def fake_rpc(*args, **kwargs):
        called["rpc"] += 1
        return {}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: True)

    result = runner.invoke(cli.app, ["service-stop", "--force"])

    assert result.exit_code == 0
    assert "Force-stopped KeepGPU service daemon" in result.output
    assert called["rpc"] == 0


def test_http_json_request_wraps_timeout(monkeypatch):
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    try:
        cli._http_json_request("GET", "http://127.0.0.1:8765/health")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Cannot reach KeepGPU service" in str(exc)


def test_http_json_request_wraps_non_json_response(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: DummyResponse())

    try:
        cli._http_json_request("GET", "http://127.0.0.1:8765/health")
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Non-JSON response from service endpoint" in str(exc)


def test_stop_service_process_rejects_unmanaged_pid(monkeypatch):
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 4321)
    monkeypatch.setattr(cli, "_is_managed_keepgpu_pid", lambda pid: False)
    monkeypatch.setattr(cli, "_clear_service_pid", lambda host, port: None)
    monkeypatch.setattr(
        cli.os,
        "kill",
        lambda pid, sig: (_ for _ in ()).throw(
            AssertionError("os.kill should not be called")
        ),
    )

    stopped = cli._stop_service_process("127.0.0.1", 8765)
    assert stopped is False
