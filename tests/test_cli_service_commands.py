import json

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


def test_start_command_rejects_negative_gpu_ids(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_ensure_service_running",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("service should not be started")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RPC should not be called")
        ),
    )

    result = runner.invoke(cli.app, ["start", "--gpu-ids", "0,-1"])

    assert result.exit_code == 1
    assert "non-negative integers" in result.output


def test_start_command_rejects_non_positive_interval(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RPC should not be called")
        ),
    )

    result = runner.invoke(cli.app, ["start", "--interval", "0"])

    assert result.exit_code == 1
    assert "interval must be positive" in result.output


def test_start_command_rejects_busy_threshold_above_percent_range(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_ensure_service_running",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("service should not be started")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RPC should not be called")
        ),
    )

    result = runner.invoke(cli.app, ["start", "--busy-threshold", "101"])

    assert result.exit_code == 1
    assert "busy_threshold must be -1 or an integer between 0 and 100" in result.output


def test_stop_requires_job_id_or_all():
    result = runner.invoke(cli.app, ["stop"])
    assert result.exit_code == 1
    assert "Provide --job-id or use --all" in result.output


def test_status_forwards_explicit_empty_job_id_to_service(monkeypatch):
    called = {}

    def fake_rpc(method, params, host, port):
        called["rpc"] = (method, params, host, port)
        raise RuntimeError("job_id must be a URL-path-safe non-empty string")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", ""])

    assert result.exit_code == 1
    assert "job_id must be a URL-path-safe non-empty string" in result.output
    method, params, _host, _port = called["rpc"]
    assert method == "status"
    assert params == {"job_id": ""}


def test_stop_forwards_explicit_empty_job_id_to_service(monkeypatch):
    called = {}

    def fake_rpc(method, params, host, port, timeout=8.0):
        called["rpc"] = (method, params, host, port, timeout)
        raise RuntimeError("job_id must be a URL-path-safe non-empty string")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", ""])

    assert result.exit_code == 1
    assert "job_id must be a URL-path-safe non-empty string" in result.output
    method, params, _host, _port, timeout = called["rpc"]
    assert method == "stop_keep"
    assert params == {"job_id": ""}
    assert timeout == 45.0


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


def test_blocking_mode_rejects_non_positive_interval(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocking runner should not be called")
        ),
    )

    result = runner.invoke(cli.app, ["--interval", "0"])

    assert result.exit_code == 1
    assert "interval must be positive" in result.output


def test_blocking_mode_rejects_duplicate_gpu_ids():
    result = runner.invoke(cli.app, ["--gpu-ids", "0,1,0"])

    assert result.exit_code == 1
    assert "gpu_ids must not contain duplicate values" in result.output
    assert "Traceback" not in result.output


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
    assert "Tracked keep sessions detected" in result.output


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
    payload = json.loads(json.loads(result.output))
    assert payload["stopped"] == []
    assert payload["timed_out"] == []
    assert payload["failed"] == []
    assert payload["errors"] == {}


def test_stop_all_does_not_fallback_for_rpc_application_error(monkeypatch):
    called = {"stop_process": False}

    def fake_rpc(method, params, host, port, timeout=8.0):
        raise RuntimeError("validation failed")

    def fake_stop_process(host, port):
        called["stop_process"] = True
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 1234)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 1
    assert "validation failed" in result.output
    assert "force-stopped local daemon" not in result.output
    assert called["stop_process"] is False


def test_stop_all_fallback_requires_stop_process_success(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        raise RuntimeError(
            "Cannot reach KeepGPU service at http://127.0.0.1:8765/rpc: timed out"
        )

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 1234)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: False)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 1
    assert "Cannot reach KeepGPU service" in result.output
    assert "ownership-verified daemon could be force-stopped" in result.output
    assert "force-stopped local daemon" not in result.output


def test_process_uid_uses_ps_when_proc_uid_is_unavailable(monkeypatch):
    class MissingProcPath:
        def __init__(self, _path):
            pass

        def stat(self):
            raise OSError("no proc")

    monkeypatch.setattr(cli, "Path", MissingProcPath)
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *args, **kwargs: "1001\n",
    )

    assert cli._process_uid(4321) == 1001


def test_process_uid_returns_none_when_target_uid_is_unknown(monkeypatch):
    class MissingProcPath:
        def __init__(self, _path):
            pass

        def stat(self):
            raise OSError("no proc")

    monkeypatch.setattr(cli, "Path", MissingProcPath)
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ps failed")),
    )

    assert cli._process_uid(4321) is None


def test_stop_service_process_requires_structured_ownership_record(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    cli._service_pid_path("127.0.0.1", 8765).write_text("4321", encoding="utf-8")
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == []


def test_stop_service_process_rejects_record_missing_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
    }
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == []


def test_stop_service_process_rejects_unknown_current_start_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: None)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == []


def test_stop_service_process_stops_matching_owned_daemon(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )

    alive = {"value": True}
    kills = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))
        alive["value"] = False

    monkeypatch.setattr(cli, "_pid_alive", lambda pid: alive["value"])
    monkeypatch.setattr(cli.os, "kill", fake_kill)

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0.5)

    assert stopped is True
    assert kills == [(4321, cli.signal.SIGTERM)]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


def test_stop_service_process_rechecks_ownership_before_sigkill(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)

    cmdline = {"value": cli._service_command("127.0.0.1", 8765)}
    monkeypatch.setattr(cli, "_process_cmdline", lambda pid: cmdline["value"])
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))
        if sig == cli.signal.SIGTERM:
            cmdline["value"] = ["python", "other.py"]

    monkeypatch.setattr(cli.os, "kill", fake_kill)

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == [(4321, cli.signal.SIGTERM)]


def test_stop_service_process_confirms_sigkill_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == [(4321, cli.signal.SIGTERM), (4321, cli.signal.SIGKILL)]
    assert cli._service_pid_path("127.0.0.1", 8765).exists()


def test_write_service_pid_stores_ownership_record(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)

    cli._write_service_pid("127.0.0.1", 8765, 4321)

    payload = json.loads(
        cli._service_pid_path("127.0.0.1", 8765).read_text(encoding="utf-8")
    )
    assert payload["pid"] == 4321
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8765
    assert payload["argv"] == cli._service_command("127.0.0.1", 8765)
    assert payload["uid"] == cli._process_uid(4321)
    assert "start_time" in payload
    assert "created_at" in payload


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


def test_stop_service_process_rejects_mismatched_record(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(cli, "_process_cmdline", lambda pid: ["python", "other.py"])
    monkeypatch.setattr(
        cli.os,
        "kill",
        lambda pid, sig: (_ for _ in ()).throw(
            AssertionError("os.kill should not be called")
        ),
    )

    stopped = cli._stop_service_process("127.0.0.1", 8765)
    assert stopped is False
