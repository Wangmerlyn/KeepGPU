import builtins
import json
import re
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from keep_gpu import cli

runner = CliRunner()
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _single_decoded_json_object(output):
    payload = json.loads(output)
    assert isinstance(payload, dict)
    return payload


def _status_session_record(state="active"):
    return {
        "job_id": "job-1",
        "params": {
            "gpu_ids": [0],
            "vram": "1GiB",
            "interval": 60,
            "busy_threshold": 25,
        },
        "state": state,
        "last_error": None,
    }


def _force_color_console(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=True, color_system="truecolor"),
    )
    return output


def test_interval_help_uses_number_metavar():
    root_help = runner.invoke(cli.app, ["--help"])
    start_help = runner.invoke(cli.app, ["start", "--help"])
    root_output = ANSI_PATTERN.sub("", root_help.output)
    start_output = ANSI_PATTERN.sub("", start_help.output)

    assert root_help.exit_code == 0
    root_interval_line = next(
        line for line in root_output.splitlines() if "--interval" in line
    )
    assert "NUMBER" in root_interval_line
    assert start_help.exit_code == 0
    start_interval_line = next(
        line for line in start_output.splitlines() if "--interval" in line
    )
    assert "NUMBER" in start_interval_line


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
            "--job-id",
            "custom-job",
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
    assert params["job_id"] == "custom-job"
    assert host == cli.DEFAULT_SERVICE_HOST
    assert port == cli.DEFAULT_SERVICE_PORT


@pytest.mark.parametrize(
    ("rpc_result", "expected_message"),
    [
        ({}, "Malformed JSON-RPC response: start_keep result must include job_id"),
        (
            {"job_id": None},
            "Malformed JSON-RPC response: start_keep result must include job_id",
        ),
        (
            {"job_id": 123},
            "Malformed JSON-RPC response: job_id must be a URL-path-safe non-empty string",
        ),
        (
            {"job_id": ""},
            "Malformed JSON-RPC response: job_id must be a URL-path-safe non-empty string",
        ),
        (
            {"job_id": "   "},
            "Malformed JSON-RPC response: job_id must be a URL-path-safe non-empty string",
        ),
        (
            {"job_id": "bad id"},
            "Malformed JSON-RPC response: job_id must be a URL-path-safe non-empty string",
        ),
    ],
)
def test_start_command_rejects_malformed_job_id_result(
    monkeypatch, rpc_result, expected_message
):
    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_rpc_call", lambda *args, **kwargs: rpc_result)

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert expected_message in " ".join(result.output.split())
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "root_args",
    [
        ["--gpu-ids", "0"],
        ["--vram", "2GiB"],
        ["--interval", "60"],
        ["--busy-threshold", "30"],
        ["--util-threshold", "30"],
        ["--threshold", "30"],
    ],
)
def test_start_rejects_root_blocking_options_before_service_calls(
    monkeypatch, root_args
):
    called = {"ensure": False, "rpc": False}

    def fake_ensure(host, port, auto_start=True):
        called["ensure"] = True
        return False

    def fake_rpc(method, params, host, port):
        called["rpc"] = True
        return {"job_id": "job-root-options"}

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, [*root_args, "start"])

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "Omit blocking-mode root options before service subcommands" in (
        normalized_output
    )
    assert "keep-gpu start --gpu-ids 0" in normalized_output
    assert called == {"ensure": False, "rpc": False}


def test_status_rejects_root_blocking_options_before_service_rpc(monkeypatch):
    called = {"rpc": False}

    def fake_rpc(method, params, host, port):
        called["rpc"] = True
        return {"active_jobs": []}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["--gpu-ids", "0", "status"])

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "`status` service subcommand" in normalized_output
    assert "Omit blocking-mode root options before service subcommands" in (
        normalized_output
    )
    assert called == {"rpc": False}


def test_root_option_source_helper_accepts_raw_commandline_value():
    assert cli._is_commandline_parameter_source("COMMANDLINE") is True


def test_start_command_accepts_fractional_interval(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: None)

    def fake_rpc(method, params, host, port):
        called["params"] = params
        return {"job_id": "job-fractional"}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start", "--interval", "0.5"])

    assert result.exit_code == 0
    assert called["params"]["interval"] == 0.5


def test_start_command_defaults_to_eco_safe_busy_threshold(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: None)

    def fake_rpc(method, params, host, port):
        called["params"] = params
        return {"job_id": "job-default"}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 0
    assert called["params"]["busy_threshold"] == 25


def test_start_command_preserves_explicit_unconditional_busy_threshold(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: None)

    def fake_rpc(method, params, host, port):
        called["params"] = params
        return {"job_id": "job-unconditional"}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start", "--busy-threshold", "-1"])

    assert result.exit_code == 0
    assert called["params"]["busy_threshold"] == -1


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


@pytest.mark.parametrize("gpu_ids", ["", "   "])
def test_start_command_rejects_empty_gpu_ids_before_auto_start(monkeypatch, gpu_ids):
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

    result = runner.invoke(cli.app, ["start", "--gpu-ids", gpu_ids])

    assert result.exit_code == 1
    assert "gpu_ids must not be empty" in result.output


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


def test_start_command_rejects_non_integer_busy_threshold_before_auto_start(
    monkeypatch,
):
    called = {"ensure": False, "rpc": False}

    def fake_ensure(*args, **kwargs):
        called["ensure"] = True
        raise AssertionError("service should not be started")

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        raise AssertionError("RPC should not be called")

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start", "--busy-threshold", "abc"])

    assert result.exit_code == 1
    assert "busy_threshold must be -1 or an integer between 0 and 100" in result.output
    assert "Usage:" not in result.output
    assert called == {"ensure": False, "rpc": False}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--job-id", "bad/id"], "job_id must be a URL-path-safe non-empty string"),
        (["--vram", "not-a-size"], "invalid format"),
        (["--vram", ("9" * 500) + "GiB"], "vram must be no more than"),
        (["--interval", str(10**1000)], "interval must be no more than"),
        (["--interval", f"+{10**1000}"], "interval must be no more than"),
        (["--interval", "NaN"], "interval must be finite and positive"),
        (["--interval", "Infinity"], "interval must be finite and positive"),
    ],
)
def test_start_command_rejects_local_inputs_before_auto_start(
    monkeypatch, args, message
):
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

    result = runner.invoke(cli.app, ["start", *args])

    assert result.exit_code == 1
    assert message in result.output


def test_start_command_rejects_invalid_host_before_auto_start(monkeypatch):
    called = {"ensure": False, "rpc": False}

    def fake_ensure(*args, **kwargs):
        called["ensure"] = True
        return False

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {"job_id": "job-host"}

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start", "--host", "bad host"])

    assert result.exit_code == 1
    assert "host must be a DNS hostname or IPv4 address" in result.output
    assert "Traceback" not in result.output
    assert called == {"ensure": False, "rpc": False}


def test_start_command_rejects_invalid_port_before_auto_start(monkeypatch):
    called = {"ensure": False, "rpc": False}

    monkeypatch.setattr(
        cli,
        "_ensure_service_running",
        lambda *args, **kwargs: called.__setitem__("ensure", True),
    )
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: called.__setitem__("rpc", True),
    )

    result = runner.invoke(cli.app, ["start", "--port", "0"])

    assert result.exit_code == 1
    assert "port must be an integer between 1 and 65535" in result.output
    assert called == {"ensure": False, "rpc": False}


def test_start_command_rejects_non_integer_port_before_auto_start(monkeypatch):
    called = {"ensure": False, "rpc": False}

    monkeypatch.setattr(
        cli,
        "_ensure_service_running",
        lambda *args, **kwargs: called.__setitem__("ensure", True),
    )
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: called.__setitem__("rpc", True),
    )

    result = runner.invoke(cli.app, ["start", "--port", "abc"])

    assert result.exit_code == 1
    assert "port must be an integer between 1 and 65535" in result.output
    assert "Traceback" not in result.output
    assert called == {"ensure": False, "rpc": False}


@pytest.mark.parametrize("invalid_port", ["abc", "0"])
def test_serve_rejects_invalid_port_before_server_import(monkeypatch, invalid_port):
    original_import = builtins.__import__
    imported = {"server": False}

    def guarded_import(name, *args, **kwargs):
        if name == "keep_gpu.mcp.server":
            imported["server"] = True
            raise AssertionError("server should not be imported before validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = runner.invoke(cli.app, ["serve", "--port", invalid_port])

    assert result.exit_code == 1
    assert "port must be an integer between 1 and 65535" in result.output
    assert "Traceback" not in result.output
    assert imported["server"] is False


def test_validate_cli_service_host_delegates_to_shared_validator():
    assert cli._validate_cli_service_host("localhost") == "localhost"

    with pytest.raises(
        cli.typer.BadParameter,
        match="host must be a DNS hostname or IPv4 address",
    ):
        cli._validate_cli_service_host("bad host")


def test_start_command_rejects_non_whitespace_malformed_host_before_auto_start(
    monkeypatch,
):
    called = {"ensure": False, "rpc": False}

    def fake_ensure(*args, **kwargs):
        called["ensure"] = True
        return False

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {"job_id": "job-host"}

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["start", "--host", "%"])

    assert result.exit_code == 1
    assert "host must be a DNS hostname or IPv4 address" in result.output
    assert called == {"ensure": False, "rpc": False}


def test_stop_requires_job_id_or_all():
    result = runner.invoke(cli.app, ["stop"])
    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "Provide --job-id or use --all."


def test_stop_rejects_job_id_with_all_before_rpc(monkeypatch):
    called = {"rpc": False, "stop_all": False}

    def fake_rpc(method, params, host, port, timeout=8.0):
        called["rpc"] = True
        return {"stopped": ["job-1"], "timed_out": [], "failed": [], "errors": {}}

    def fake_stop_all(host, port):
        called["stop_all"] = True
        return {"stopped": ["job-1"], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_all_sessions_with_fallback", fake_stop_all)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1", "--all"])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert "Use either --job-id or --all" in payload["error"]
    assert called["rpc"] is False
    assert called["stop_all"] is False


@pytest.mark.parametrize("job_id", ["", "   ", "bad/id"])
def test_status_rejects_invalid_job_id_before_rpc(monkeypatch, job_id):
    def fake_rpc(*args, **kwargs):
        raise AssertionError("RPC should not be called")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", job_id])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "job_id must be a URL-path-safe non-empty string"


@pytest.mark.parametrize("job_id", ["", "   ", "bad/id"])
def test_stop_rejects_invalid_job_id_before_rpc_or_fallback(monkeypatch, job_id):
    def fake_rpc(*args, **kwargs):
        raise AssertionError("RPC should not be called")

    def fake_stop_all(*args, **kwargs):
        raise AssertionError("stop-all fallback should not be called")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_all_sessions_with_fallback", fake_stop_all)

    result = runner.invoke(cli.app, ["stop", "--job-id", job_id])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "job_id must be a URL-path-safe non-empty string"


@pytest.mark.parametrize(
    ("command", "called_key"),
    [
        (["status", "--host", "bad host"], "rpc"),
        (["stop", "--all", "--host", "bad host"], "stop_all"),
        (["list-gpus", "--host", "bad host"], "rpc"),
    ],
)
def test_service_json_commands_reject_invalid_host_before_rpc_or_fallback(
    monkeypatch, command, called_key
):
    called = {"rpc": False, "stop_all": False}

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {}

    def fake_stop_all(*args, **kwargs):
        called["stop_all"] = True
        return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_all_sessions_with_fallback", fake_stop_all)

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "host must be a DNS hostname or IPv4 address"
    assert called[called_key] is False


def test_service_json_command_rejects_invalid_port_before_rpc(monkeypatch):
    called = {"rpc": False}

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--port", "70000"])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "port must be an integer between 1 and 65535"
    assert called["rpc"] is False


@pytest.mark.parametrize(
    ("command", "called_key"),
    [
        (["status", "--port", "abc"], "rpc"),
        (["status", "--port", "+8765"], "rpc"),
        (["stop", "--all", "--port", "abc"], "stop_all"),
        (["stop", "--all", "--port", "8_765"], "stop_all"),
        (["list-gpus", "--port", "abc"], "rpc"),
        (["list-gpus", "--port", "１２３"], "rpc"),
    ],
)
def test_service_json_commands_reject_non_integer_port_as_json_before_rpc_or_fallback(
    monkeypatch, command, called_key
):
    called = {"rpc": False, "stop_all": False}

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {}

    def fake_stop_all(*args, **kwargs):
        called["stop_all"] = True
        return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_all_sessions_with_fallback", fake_stop_all)

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "port must be an integer between 1 and 65535"
    assert called[called_key] is False


def test_service_stop_rejects_invalid_host_before_daemon_operations(monkeypatch):
    called = {"available": False, "stop": False}

    def fake_available(*args, **kwargs):
        called["available"] = True
        return False

    def fake_stop(*args, **kwargs):
        called["stop"] = True
        return True

    monkeypatch.setattr(cli, "_service_available", fake_available)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop)

    result = runner.invoke(cli.app, ["service-stop", "--host", "bad host", "--force"])

    assert result.exit_code == 1
    assert "host must be a DNS hostname or IPv4 address" in result.output
    assert called == {"available": False, "stop": False}


def test_service_stop_rejects_non_integer_port_before_daemon_operations(monkeypatch):
    called = {"rpc": False, "stop": False}

    def fake_rpc(*args, **kwargs):
        called["rpc"] = True
        return {}

    def fake_stop(*args, **kwargs):
        called["stop"] = True
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop)

    result = runner.invoke(cli.app, ["service-stop", "--port", "abc", "--force"])

    assert result.exit_code == 1
    assert "port must be an integer between 1 and 65535" in result.output
    assert "Traceback" not in result.output
    assert called == {"rpc": False, "stop": False}


def test_http_json_request_wraps_malformed_url_as_service_unreachable():
    with pytest.raises(cli.ServiceUnreachableError, match="Cannot reach KeepGPU"):
        cli._http_json_request("GET", "http://bad host:8765/health")


def test_http_json_request_reports_invalid_utf8_as_service_response_error(monkeypatch):
    class InvalidUtf8Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"\xff"

    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: InvalidUtf8Response())

    with pytest.raises(cli.ServiceResponseError, match="Invalid UTF-8 response"):
        cli._http_json_request("GET", "http://127.0.0.1:8765/health")


def test_status_outputs_json_error_for_nonstandard_json_constant_response(monkeypatch):
    class NaNResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"jsonrpc":"2.0","id":1000,'
                b'"result":{"active_jobs":[],"future_extra":NaN}}'
            )

    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: NaNResponse())

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert "Non-JSON response from service endpoint" in payload["error"]
    assert result.exception is not None
    assert not isinstance(result.exception, ValueError)


def test_ensure_service_running_stops_auto_started_process_on_health_timeout(
    monkeypatch, tmp_path
):
    stopped = []

    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_service_available", lambda host, port: False)
    monkeypatch.setattr(cli, "_start_service_process", lambda host, port: 4321)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)

    def fake_stop(host, port, timeout=3.0):
        stopped.append((host, port, timeout))

    monkeypatch.setattr(cli, "_stop_service_process", fake_stop)

    with pytest.raises(RuntimeError, match="Failed to auto-start KeepGPU service"):
        cli._ensure_service_running("127.0.0.1", 8765)

    assert stopped == [("127.0.0.1", 8765, 1.0)]


def test_ensure_service_running_clears_dead_pid_record_before_auto_start(
    monkeypatch, tmp_path
):
    starts = []

    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_service_available", lambda host, port: len(starts) > 0)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        cli, "_start_service_process", lambda host, port: starts.append((host, port))
    )

    auto_started = cli._ensure_service_running("127.0.0.1", 8765)

    assert auto_started is True
    assert starts == [("127.0.0.1", 8765)]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


def test_ensure_service_running_refuses_to_spawn_over_live_managed_pid(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_service_available", lambda host, port: False)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    pid_path = cli._service_pid_path("127.0.0.1", 8765)
    original_record = pid_path.read_text(encoding="utf-8")
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: pid == 4321)
    monkeypatch.setattr(
        cli,
        "_process_cmdline",
        lambda pid: cli._service_command("127.0.0.1", 8765),
    )
    monkeypatch.setattr(
        cli,
        "_start_service_process",
        lambda host, port: (_ for _ in ()).throw(
            AssertionError("auto-start must not overwrite a live PID record")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_stop_service_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("timeout cleanup should not run without auto-start")
        ),
    )

    with pytest.raises(RuntimeError, match="health check failed") as exc_info:
        cli._ensure_service_running("127.0.0.1", 8765)

    assert str(cli._service_log_path("127.0.0.1", 8765)) in str(exc_info.value)
    assert pid_path.read_text(encoding="utf-8") == original_record


def test_status_outputs_single_decoded_json_object(monkeypatch):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {}
        return {
            "active": True,
            "active_jobs": [
                {
                    "job_id": "job-1",
                    "params": {"gpu_ids": [0]},
                    "state": "active",
                    "last_error": None,
                }
            ],
        }

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload["active"] is True
    assert payload["active_jobs"][0]["job_id"] == "job-1"


@pytest.mark.parametrize(
    ("command", "expected_payload"),
    [
        (["status"], {"active_jobs": []}),
        (
            ["stop", "--job-id", "job-1"],
            {"stopped": ["job-1"], "timed_out": [], "failed": [], "errors": {}},
        ),
        (["list-gpus"], {"gpus": []}),
    ],
)
def test_service_json_commands_stay_plain_json_when_console_color_is_enabled(
    monkeypatch, command, expected_payload
):
    output = _force_color_console(monkeypatch)

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        if method == "stop_keep":
            return {"stopped": ["job-1"], "timed_out": [], "failed": [], "errors": {}}
        if method == "list_gpus":
            return {"gpus": []}
        raise AssertionError(f"unexpected RPC method: {method}")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, command)

    rendered = output.getvalue()
    assert result.exit_code == 0
    assert "\x1b[" not in rendered
    assert _single_decoded_json_object(rendered) == expected_payload


def test_service_json_errors_stay_plain_json_when_console_color_is_enabled(
    monkeypatch,
):
    output = _force_color_console(monkeypatch)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("telemetry service unavailable")
        ),
    )

    result = runner.invoke(cli.app, ["list-gpus"])

    rendered = output.getvalue()
    assert result.exit_code == 1
    assert "\x1b[" not in rendered
    assert _single_decoded_json_object(rendered) == {
        "error": "telemetry service unavailable"
    }


def test_status_job_outputs_single_decoded_json_object(monkeypatch):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {"job_id": "job-1"}
        return {
            "active": True,
            "job_id": "job-1",
            "params": {"gpu_ids": [0]},
            "state": "active",
            "last_error": None,
        }

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", "job-1"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload["job_id"] == "job-1"


@pytest.mark.parametrize("payload", [{}, {"active_jobs": {}}])
def test_status_rejects_malformed_all_session_payloads(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "active_jobs must be a list" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {"active_jobs": [1]},
        {
            "active_jobs": [
                {
                    "job_id": 1,
                    "params": {},
                    "state": "active",
                    "last_error": None,
                }
            ]
        },
        {
            "active_jobs": [
                {
                    "job_id": "bad/id",
                    "params": {},
                    "state": "active",
                    "last_error": None,
                }
            ]
        },
        {
            "active_jobs": [
                {
                    "params": {},
                    "state": "active",
                    "last_error": None,
                }
            ]
        },
        {
            "active_jobs": [
                {
                    "job_id": "job-1",
                    "state": "active",
                    "last_error": None,
                }
            ]
        },
        {
            "active_jobs": [
                {
                    "job_id": "job-1",
                    "params": {},
                    "last_error": None,
                }
            ]
        },
        {
            "active_jobs": [
                {
                    "job_id": "job-1",
                    "params": {},
                    "state": "active",
                    "last_error": 1,
                }
            ]
        },
    ],
)
def test_status_rejects_malformed_active_job_entries(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("command", "expected_params", "payload", "field"),
    [
        (
            ["status"],
            {},
            {"active_jobs": [_status_session_record("weird")]},
            "active_jobs[0].state",
        ),
        (
            ["status", "--job-id", "job-1"],
            {"job_id": "job-1"},
            {"active": True, **_status_session_record("weird")},
            "result.state",
        ),
    ],
)
def test_status_rejects_unknown_session_state(
    monkeypatch, command, expected_params, payload, field
):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == expected_params
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert f"{field} must be one of" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "session_params",
    [
        {"gpu_ids": "all"},
        {"gpu_ids": []},
        {"interval": -1},
        {"busy_threshold": 101},
        {"vram": []},
    ],
)
def test_status_rejects_malformed_known_session_params(monkeypatch, session_params):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {}
        return {
            "active_jobs": [
                {
                    "job_id": "job-1",
                    "params": session_params,
                    "state": "active",
                    "last_error": None,
                }
            ]
        }

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert "active_jobs[0].params" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"active": "yes", "job_id": "job-1"},
        {"active": True, "job_id": 1},
        {"active": False, "job_id": "bad/id"},
        {
            "active": True,
            "job_id": "bad/id",
            "params": {},
            "state": "active",
            "last_error": None,
        },
    ],
)
def test_status_job_rejects_malformed_payloads(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {"job_id": "job-1"}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {"active": True, "job_id": "job-1"},
        {
            "active": True,
            "job_id": "job-1",
            "params": [],
            "state": "active",
            "last_error": None,
        },
        {
            "active": True,
            "job_id": "job-1",
            "params": {},
            "last_error": None,
        },
        {
            "active": True,
            "job_id": "job-1",
            "params": {},
            "state": "active",
            "last_error": 1,
        },
    ],
)
def test_status_job_rejects_active_payload_missing_session_fields(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {"job_id": "job-1"}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert "Traceback" not in result.output


def test_status_job_allows_inactive_payload_without_session_fields(monkeypatch):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {"job_id": "missing-job"}
        return {"active": False, "job_id": "missing-job"}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", "missing-job"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload == {"active": False, "job_id": "missing-job"}


@pytest.mark.parametrize(
    "payload",
    [
        {"active": False, "job_id": "other-job"},
        {
            "active": True,
            "job_id": "other-job",
            "params": {},
            "state": "active",
            "last_error": None,
        },
    ],
)
def test_status_job_rejects_mismatched_job_id_result(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "status"
        assert params == {"job_id": "job-1"}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["status", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed status response" in decoded["error"]
    assert "result.job_id must match requested job_id" in decoded["error"]
    assert "Traceback" not in result.output


def test_stop_job_outputs_single_decoded_json_object(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "job-1"}
        assert timeout == 45.0
        return {"stopped": ["job-1"], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload["stopped"] == ["job-1"]


def test_stop_job_allows_empty_not_found_result(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "missing-job"}
        assert timeout == 45.0
        return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "missing-job"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload == {"stopped": [], "timed_out": [], "failed": [], "errors": {}}


@pytest.mark.parametrize(
    "payload",
    [
        {"stopped": ["other-job"], "timed_out": [], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": ["other-job"], "failed": [], "errors": {}},
        {
            "stopped": [],
            "timed_out": [],
            "failed": ["other-job"],
            "errors": {"other-job": "release failed"},
        },
    ],
)
def test_stop_job_rejects_mismatched_job_id_result(monkeypatch, payload):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "job-1"}
        assert timeout == 45.0
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert "outcome job ids must match requested job_id" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"stopped": "job-1", "timed_out": [], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": {}, "failed": [], "errors": {}},
        {"stopped": [], "timed_out": [], "failed": None, "errors": {}},
        {"stopped": [], "timed_out": [], "failed": [], "errors": []},
        {
            "stopped": [],
            "timed_out": [],
            "failed": [],
            "errors": {},
            "message": {"text": "bad"},
        },
        {
            "stopped": [],
            "timed_out": [],
            "failed": [],
            "errors": {},
            "message": None,
        },
    ],
)
def test_stop_job_rejects_malformed_payloads(monkeypatch, payload):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "job-1"}
        assert timeout == 45.0
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {"stopped": [1], "timed_out": [], "failed": [], "errors": {}},
        {"stopped": ["bad/id"], "timed_out": [], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": [None], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": [], "failed": [False], "errors": {}},
        {
            "stopped": [],
            "timed_out": [],
            "failed": ["bad/id"],
            "errors": {"bad/id": "boom"},
        },
        {
            "stopped": [],
            "timed_out": [],
            "failed": [],
            "errors": {"bad/id": "boom"},
        },
        {"stopped": [], "timed_out": [], "failed": [], "errors": {"job-1": 1}},
    ],
)
def test_stop_job_rejects_malformed_job_id_lists_and_errors(monkeypatch, payload):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "job-1"}
        assert timeout == 45.0
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {"stopped": ["job-1", "job-1"], "timed_out": [], "failed": [], "errors": {}},
        {"stopped": ["job-1"], "timed_out": ["job-1"], "failed": [], "errors": {}},
        {
            "stopped": ["job-1"],
            "timed_out": [],
            "failed": ["job-1"],
            "errors": {"job-1": "boom"},
        },
        {"stopped": [], "timed_out": [], "failed": ["job-1"], "errors": {}},
        {
            "stopped": [],
            "timed_out": [],
            "failed": [],
            "errors": {"job-1": "boom"},
        },
    ],
)
def test_stop_job_rejects_inconsistent_outcome_payloads(monkeypatch, payload):
    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {"job_id": "job-1"}
        assert timeout == 45.0
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["stop", "--job-id", "job-1"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert "Traceback" not in result.output


def test_stop_all_outputs_single_decoded_json_object(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_stop_all_sessions_with_fallback",
        lambda host, port: {
            "stopped": ["job-1"],
            "timed_out": [],
            "failed": [],
            "errors": {},
        },
    )

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload["stopped"] == ["job-1"]


def test_stop_all_rejects_malformed_payload(monkeypatch):
    called = {"stop_process": False}

    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {}
        assert timeout == 45.0
        return {"stopped": []}

    def fake_stop_process(host, port):
        called["stop_process"] = True
        raise AssertionError("_stop_service_process must not be called")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 1234)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert called["stop_process"] is False
    assert "Traceback" not in result.output


def test_stop_all_rejects_inconsistent_payload_before_stopping_daemon(monkeypatch):
    called = {"stop_process": False}

    def fake_rpc(method, params, host, port, timeout=8.0):
        assert method == "stop_keep"
        assert params == {}
        assert timeout == 45.0
        return {
            "stopped": ["job-1"],
            "timed_out": [],
            "failed": ["job-1"],
            "errors": {"job-1": "boom"},
        }

    def fake_stop_process(host, port):
        called["stop_process"] = True
        raise AssertionError("_stop_service_process must not be called")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_read_service_pid", lambda host, port: 1234)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["stop", "--all"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed stop_keep response" in decoded["error"]
    assert called["stop_process"] is False
    assert "Traceback" not in result.output


def test_list_gpus_outputs_single_decoded_json_object(monkeypatch):
    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        return {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        }

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 0
    payload = _single_decoded_json_object(result.output)
    assert payload["gpus"][0]["id"] == 0


@pytest.mark.parametrize("payload", [{}, {"gpus": {}}, {"gpus": [1]}])
def test_list_gpus_rejects_malformed_payloads(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed list_gpus response" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize("missing_field", ["memory_total", "memory_used"])
def test_list_gpus_reports_missing_memory_fields(monkeypatch, missing_field):
    gpu = {
        "id": 0,
        "visible_id": 0,
        "platform": "cuda",
        "name": "GPU 0",
        "memory_total": 1024,
        "memory_used": 512,
        "utilization": 12.5,
    }
    del gpu[missing_field]

    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        return {"gpus": [gpu]}

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert f"missing '{missing_field}'" in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("memory_total", "memory_used", "message_fragment"),
    [
        ("1024", 512, "memory_total must be a non-negative integer or null"),
        (-1, 0, "memory_total must be a non-negative integer or null"),
        (1024, -1, "memory_used must be a non-negative integer or null"),
        (1024, 2048, "memory_used must not exceed memory_total"),
    ],
)
def test_list_gpus_reports_invalid_memory_fields(
    monkeypatch, memory_total, memory_used, message_fragment
):
    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        return {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": memory_total,
                    "memory_used": memory_used,
                    "utilization": 12.5,
                }
            ]
        }

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert message_fragment in decoded["error"]
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": True,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": "0",
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 1,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": -1,
                    "visible_id": -1,
                    "platform": "cuda",
                    "name": "GPU hidden",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                },
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU alias",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                },
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": 1,
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": 0,
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": "1024",
                    "memory_used": 512,
                    "utilization": 12.5,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": -1,
                    "memory_used": 0,
                    "utilization": None,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": -1,
                    "utilization": None,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 2048,
                    "utilization": None,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": False,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": -1,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": 101,
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": float("nan"),
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": float("inf"),
                }
            ]
        },
        {
            "gpus": [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "cuda",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 512,
                    "utilization": float("-inf"),
                }
            ]
        },
    ],
)
def test_list_gpus_rejects_malformed_gpu_records(monkeypatch, payload):
    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        return payload

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 1
    decoded = _single_decoded_json_object(result.output)
    assert "Malformed list_gpus response" in decoded["error"]
    assert "Traceback" not in result.output


def test_list_gpus_error_outputs_single_decoded_json_object(monkeypatch):
    def fake_rpc(method, params, host, port):
        assert method == "list_gpus"
        assert params == {}
        raise RuntimeError("telemetry service unavailable")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)

    result = runner.invoke(cli.app, ["list-gpus"])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert payload["error"] == "telemetry service unavailable"


def test_rpc_call_sends_explicit_jsonrpc_request_version(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    captured = {}

    def fake_http_json_request(method, url, payload, timeout=8.0):
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = payload
        return {"jsonrpc": "2.0", "id": payload["id"], "result": {}}

    monkeypatch.setattr(cli, "_http_json_request", fake_http_json_request)

    assert cli._rpc_call("status", None, "127.0.0.1", 8765) == {}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8765/rpc"
    assert captured["payload"] == {
        "jsonrpc": "2.0",
        "id": 1000,
        "method": "status",
        "params": {},
    }


def test_rpc_call_rejects_success_envelope_without_result(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)

    def fake_http_json_request(method, url, payload, timeout=8.0):
        assert method == "POST"
        assert url == "http://127.0.0.1:8765/rpc"
        assert payload["id"] == 1000
        return {"jsonrpc": "2.0", "id": 1000}

    monkeypatch.setattr(cli, "_http_json_request", fake_http_json_request)

    with pytest.raises(cli.ServiceResponseError, match="missing result"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_success_envelope_with_non_object_result(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": 1000, "result": []},
    )

    with pytest.raises(cli.ServiceResponseError, match="result must be an object"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


@pytest.mark.parametrize("version", [None, "1.0"])
def test_rpc_call_rejects_success_envelope_with_invalid_jsonrpc_version(
    monkeypatch, version
):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    response = {"id": 1000, "result": {}}
    if version is not None:
        response["jsonrpc"] = version
    monkeypatch.setattr(cli, "_http_json_request", lambda *args, **kwargs: response)

    with pytest.raises(cli.ServiceResponseError, match="jsonrpc must be 2.0"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_success_envelope_with_mismatched_id(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": 999, "result": {}},
    )

    with pytest.raises(cli.ServiceResponseError, match="mismatched id"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_error_envelope_with_non_object_error(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": 1000, "error": "bad"},
    )

    with pytest.raises(cli.ServiceResponseError, match="error must be an object"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


@pytest.mark.parametrize(
    "error",
    [
        {"code": -32603},
        {"code": -32603, "message": None},
        {"code": -32603, "message": []},
    ],
)
def test_rpc_call_rejects_error_envelope_without_string_message(monkeypatch, error):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": 1000, "error": error},
    )

    with pytest.raises(
        cli.ServiceResponseError, match=re.escape("error.message must be a string")
    ):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_error_envelope_with_null_id(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        },
    )

    with pytest.raises(cli.ServiceResponseError, match="mismatched id"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


@pytest.mark.parametrize("body", [{"result": {}}, {"error": {"message": "boom"}}])
@pytest.mark.parametrize(
    ("request_time", "response_id"),
    [
        (1.0, 1000.0),
        (0.001, True),
    ],
)
def test_rpc_call_rejects_envelope_with_invalid_id_type(
    monkeypatch, body, request_time, response_id
):
    monkeypatch.setattr(cli.time, "time", lambda: request_time)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": response_id, **body},
    )

    with pytest.raises(cli.ServiceResponseError, match="mismatched id"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_error_envelope_without_id_member(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Requests must include an id."},
        },
    )

    with pytest.raises(cli.ServiceResponseError, match="missing id"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_envelope_with_both_error_and_result(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {
            "jsonrpc": "2.0",
            "id": 1000,
            "error": {"code": -32000, "message": "service failed"},
            "result": {},
        },
    )

    with pytest.raises(
        cli.ServiceResponseError, match="both error and result members are present"
    ):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


def test_rpc_call_rejects_non_object_jsonrpc_response(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(cli, "_http_json_request", lambda *args, **kwargs: [])

    with pytest.raises(cli.ServiceResponseError, match="response must be an object"):
        cli._rpc_call("status", {}, "127.0.0.1", 8765)


@pytest.mark.parametrize(
    ("command", "method", "params"),
    [
        (["status"], "status", {}),
        (["stop", "--job-id", "job-1"], "stop_keep", {"job_id": "job-1"}),
        (["list-gpus"], "list_gpus", {}),
    ],
)
def test_service_json_commands_output_json_error_for_non_object_rpc_response(
    monkeypatch, command, method, params
):
    def fake_http_json_request(http_method, url, payload, timeout=8.0):
        assert payload["method"] == method
        assert payload["params"] == params
        return []

    monkeypatch.setattr(cli, "_http_json_request", fake_http_json_request)

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert "response must be an object" in payload["error"]


def test_status_outputs_json_error_for_malformed_rpc_success_envelope(monkeypatch):
    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {"jsonrpc": "2.0", "id": 1000},
    )

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 1
    payload = _single_decoded_json_object(result.output)
    assert "missing result" in payload["error"]


def test_blocking_mode_defaults_to_eco_safe_busy_threshold(monkeypatch):
    called = {}

    def fake_run(interval, gpu_ids, vram, legacy_threshold, busy_threshold):
        called["args"] = (interval, gpu_ids, vram, legacy_threshold, busy_threshold)

    monkeypatch.setattr(cli, "_run_blocking", fake_run)
    result = runner.invoke(
        cli.app,
        ["--interval", "120", "--gpu-ids", "0", "--vram", "1GiB"],
    )

    assert result.exit_code == 0
    assert called["args"] == (120, "0", "1GiB", None, 25)


def test_blocking_mode_preserves_explicit_unconditional_busy_threshold(monkeypatch):
    called = {}

    def fake_run(interval, gpu_ids, vram, legacy_threshold, busy_threshold):
        called["args"] = (interval, gpu_ids, vram, legacy_threshold, busy_threshold)

    monkeypatch.setattr(cli, "_run_blocking", fake_run)
    result = runner.invoke(
        cli.app,
        [
            "--interval",
            "120",
            "--gpu-ids",
            "0",
            "--vram",
            "1GiB",
            "--busy-threshold",
            "-1",
        ],
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


def test_blocking_mode_rejects_non_integer_busy_threshold_without_usage(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocking runner should not be called")
        ),
    )

    result = runner.invoke(cli.app, ["--busy-threshold", "abc"])

    assert result.exit_code == 1
    assert "busy_threshold must be -1 or an integer between 0 and 100" in result.output
    assert "Usage:" not in result.output


def test_invalid_root_interval_before_subcommand_is_rejected(monkeypatch):
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

    result = runner.invoke(cli.app, ["--interval", "not-a-number", "start"])

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "--interval" in normalized_output
    assert "Omit blocking-mode root options before service subcommands" in (
        normalized_output
    )


def test_invalid_root_busy_threshold_before_subcommand_is_rejected(monkeypatch):
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

    result = runner.invoke(cli.app, ["--busy-threshold", "abc", "start"])

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "--busy-threshold" in normalized_output
    assert "Omit blocking-mode root options before service subcommands" in (
        normalized_output
    )
    assert "Usage:" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--vram", ("9" * 500) + "GiB"], "vram must be no more than"),
        (["--vram", "not-a-size"], "invalid format"),
        (["--threshold", ("9" * 500) + "GiB"], "vram must be no more than"),
    ],
)
def test_blocking_mode_rejects_invalid_vram_without_raw_exception(args, message):
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 1
    assert message in result.output


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
    assert "job_id=job-abc" in result.output
    assert "Dashboard:" in result.output
    assert "keep-gpu status --job-id job-abc" in result.output
    assert "keep-gpu stop --job-id job-abc" in result.output
    assert "keep-gpu service-stop" in result.output


def test_start_rolls_back_auto_started_service_on_startup_unavailable(monkeypatch):
    stopped = []

    def fake_ensure(host, port, auto_start=True):
        return True

    def fake_rpc(method, params, host, port):
        raise cli.ServiceRPCError(
            "No usable visible GPUs", code=cli.JSONRPC_STARTUP_UNAVAILABLE
        )

    def fake_stop(host, port):
        stopped.append((host, port))
        return True

    monkeypatch.setattr(cli, "_ensure_service_running", fake_ensure)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop)

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "No usable visible GPUs" in result.output
    assert stopped == [(cli.DEFAULT_SERVICE_HOST, cli.DEFAULT_SERVICE_PORT)]


def test_start_does_not_stop_auto_started_service_for_non_startup_rpc_error(
    monkeypatch,
):
    stopped = []

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            cli.ServiceRPCError("Internal error", code=-32603)
        ),
    )
    monkeypatch.setattr(
        cli, "_stop_service_process", lambda host, port: stopped.append((host, port))
    )

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "Internal error" in result.output
    assert stopped == []


def test_start_does_not_stop_already_running_service_on_startup_unavailable(
    monkeypatch,
):
    stopped = []

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            cli.ServiceRPCError(
                "No usable visible GPUs", code=cli.JSONRPC_STARTUP_UNAVAILABLE
            )
        ),
    )
    monkeypatch.setattr(
        cli, "_stop_service_process", lambda host, port: stopped.append((host, port))
    )

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "No usable visible GPUs" in result.output
    assert stopped == []


def test_start_does_not_stop_auto_started_service_for_malformed_success_payload(
    monkeypatch,
):
    stopped = []

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_rpc_call", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli, "_stop_service_process", lambda host, port: stopped.append((host, port))
    )

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "start_keep result must include job_id" in result.output
    assert stopped == []


def test_start_does_not_stop_auto_started_service_for_malformed_error_envelope(
    monkeypatch,
):
    stopped = []

    monkeypatch.setattr(cli.time, "time", lambda: 1.0)
    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli,
        "_http_json_request",
        lambda *args, **kwargs: {
            "jsonrpc": "2.0",
            "id": 1000,
            "error": {"code": cli.JSONRPC_STARTUP_UNAVAILABLE, "message": []},
        },
    )
    monkeypatch.setattr(
        cli, "_stop_service_process", lambda host, port: stopped.append((host, port))
    )

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "error.message must be a string" in result.output
    assert stopped == []


def test_start_rollback_stop_failure_preserves_startup_error(monkeypatch):
    def fail_stop(host, port):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(cli, "_ensure_service_running", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            cli.ServiceRPCError(
                "No usable visible GPUs", code=cli.JSONRPC_STARTUP_UNAVAILABLE
            )
        ),
    )
    monkeypatch.setattr(cli, "_stop_service_process", fail_stop)

    result = runner.invoke(cli.app, ["start"])

    assert result.exit_code == 1
    assert "No usable visible GPUs" in result.output
    assert "stop failed" not in result.output


def test_service_stop_requires_managed_pid(monkeypatch):
    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}

    monkeypatch.setattr(cli, "_service_available", lambda host, port: True)
    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", lambda host, port: False)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "No managed daemon PID found" in result.output


@pytest.mark.parametrize("payload", [{}, {"active_jobs": {}}])
def test_service_stop_rejects_malformed_status_before_side_effects(
    monkeypatch, payload
):
    calls = {"stop_keep": 0, "stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return payload
        if method == "stop_keep":
            calls["stop_keep"] += 1
            return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "active_jobs must be a list" in result.output
    assert calls == {"stop_keep": 0, "stop_process": 0}
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"stopped": [], "timed_out": [], "failed": [], "errors": []},
    ],
)
def test_service_stop_rejects_malformed_stop_keep_before_stopping_daemon(
    monkeypatch, payload
):
    calls = {"stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        if method == "stop_keep":
            return payload
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "Malformed stop_keep response" in result.output
    assert calls["stop_process"] == 0
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("payload", "job_id"),
    [
        (
            {
                "stopped": [],
                "timed_out": ["slow-job"],
                "failed": [],
                "errors": {},
            },
            "slow-job",
        ),
        (
            {
                "stopped": [],
                "timed_out": [],
                "failed": ["bad-job"],
                "errors": {"bad-job": "release failed"},
            },
            "bad-job",
        ),
    ],
)
def test_service_stop_rejects_incomplete_stop_keep_before_stopping_daemon(
    monkeypatch, payload, job_id
):
    calls = {"stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        if method == "stop_keep":
            return payload
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert job_id in result.output
    assert "keep-gpu" in result.output
    assert "service-stop --force" in result.output
    assert calls["stop_process"] == 0
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("message", "expected_fragment"),
    [
        (
            "Timed out while stopping some sessions.",
            "Timed out while stopping some sessions.",
        ),
        (" ", "' '"),
    ],
)
def test_service_stop_rejects_stop_keep_message_before_stopping_daemon(
    monkeypatch, message, expected_fragment
):
    calls = {"stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        if method == "stop_keep":
            return {
                "stopped": [],
                "timed_out": [],
                "failed": [],
                "errors": {},
                "message": message,
            }
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert expected_fragment in result.output
    assert "service-stop --force" in result.output
    assert calls["stop_process"] == 0
    assert "Traceback" not in result.output


def test_service_stop_rejects_newly_stopped_session_before_stopping_daemon(
    monkeypatch,
):
    calls = {"stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            return {"active_jobs": []}
        if method == "stop_keep":
            return {
                "stopped": ["late-job"],
                "timed_out": [],
                "failed": [],
                "errors": {},
            }
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "late-job" in result.output
    assert "service-stop --force" in result.output
    assert calls["stop_process"] == 0
    assert "Traceback" not in result.output


def test_service_stop_rechecks_status_after_clean_stop_keep_before_stopping_daemon(
    monkeypatch,
):
    calls = {"status": 0, "stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            calls["status"] += 1
            if calls["status"] == 1:
                return {"active_jobs": []}
            return {
                "active_jobs": [
                    {
                        "job_id": "late-job",
                        "params": {},
                        "state": "active",
                        "last_error": None,
                    }
                ]
            }
        if method == "stop_keep":
            return {
                "stopped": [],
                "timed_out": [],
                "failed": [],
                "errors": {},
            }
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert calls == {"status": 2, "stop_process": 0}
    assert "Tracked keep sessions detected" in result.output
    assert "late-job" in result.output
    assert "Traceback" not in result.output


def test_service_stop_rejects_malformed_final_status_before_stopping_daemon(
    monkeypatch,
):
    calls = {"status": 0, "stop_process": 0}

    def fake_rpc(method, params, host, port, timeout=8.0):
        if method == "status":
            calls["status"] += 1
            if calls["status"] == 1:
                return {"active_jobs": []}
            return {"active_jobs": {}}
        if method == "stop_keep":
            return {
                "stopped": [],
                "timed_out": [],
                "failed": [],
                "errors": {},
            }
        raise AssertionError(f"unexpected method {method}")

    def fake_stop_process(host, port):
        calls["stop_process"] += 1
        return True

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fake_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert calls == {"status": 2, "stop_process": 0}
    assert "active_jobs must be a list" in result.output
    assert "Traceback" not in result.output


def test_service_stop_requires_live_status_without_force(monkeypatch):
    called = {"stop_process": False}

    def fail_stop_process(host, port):
        called["stop_process"] = True
        raise AssertionError("_stop_service_process must not be called")

    def fake_rpc(*args, **kwargs):
        raise cli.ServiceUnreachableError("mocked unreachable")

    monkeypatch.setattr(cli, "_rpc_call", fake_rpc)
    monkeypatch.setattr(cli, "_stop_service_process", fail_stop_process)

    result = runner.invoke(cli.app, ["service-stop"])

    assert result.exit_code == 1
    assert "service-stop --force" in result.output
    assert called["stop_process"] is False
    assert "Traceback" not in result.output


def test_service_stop_refuses_active_sessions_without_force(monkeypatch):
    monkeypatch.setattr(cli, "_service_available", lambda host, port: True)
    monkeypatch.setattr(
        cli,
        "_rpc_call",
        lambda method, params, host, port, timeout=8.0: (
            {
                "active_jobs": [
                    {
                        "job_id": "j1",
                        "params": {},
                        "state": "active",
                        "last_error": None,
                    }
                ]
            }
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
    payload = _single_decoded_json_object(result.output)
    assert "Cannot reach KeepGPU service" in payload["error"]
    assert "service-stop --force" in payload["error"]
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
    payload = _single_decoded_json_object(result.output)
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


def test_process_start_identity_uses_ps_when_proc_stat_is_unavailable(monkeypatch):
    class MissingProcStatPath:
        def __init__(self, _path):
            pass

        def exists(self):
            return False

    calls = []

    def fake_check_output(args, **kwargs):
        calls.append((args, kwargs))
        return "Wed Jul  1 05:00:00 2026\n"

    monkeypatch.setattr(cli, "Path", MissingProcStatPath)
    monkeypatch.setattr(cli.subprocess, "check_output", fake_check_output)

    assert cli._process_start_identity(4321) == "Wed Jul  1 05:00:00 2026"
    assert calls == [
        (
            ["ps", "-p", "4321", "-o", "lstart="],
            {"text": True},
        )
    ]


@pytest.mark.parametrize("ps_output", ["\n", "   \n"])
def test_process_start_identity_returns_none_for_empty_ps_output(
    monkeypatch, ps_output
):
    class MissingProcStatPath:
        def __init__(self, _path):
            pass

        def exists(self):
            return False

    monkeypatch.setattr(cli, "Path", MissingProcStatPath)
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *args, **kwargs: ps_output,
    )

    assert cli._process_start_identity(4321) is None


def test_process_start_identity_returns_none_when_target_start_time_is_unknown(
    monkeypatch,
):
    class MissingProcStatPath:
        def __init__(self, _path):
            pass

        def exists(self):
            return False

    monkeypatch.setattr(cli, "Path", MissingProcStatPath)
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ps failed")),
    )

    assert cli._process_start_identity(4321) is None


def test_stop_service_process_uses_ps_start_identity_when_proc_is_unavailable(
    monkeypatch, tmp_path
):
    class MissingProcPath:
        def __init__(self, _path):
            pass

        def exists(self):
            return False

        def stat(self):
            raise OSError("no proc")

    def fake_check_output(args, **kwargs):
        if args == ["ps", "-p", "4321", "-o", "uid="]:
            return "1001\n"
        if args == ["ps", "-p", "4321", "-o", "lstart="]:
            return "Wed Jul  1 05:00:00 2026\n"
        raise AssertionError(f"unexpected ps command: {args!r}")

    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "Path", MissingProcPath)
    monkeypatch.setattr(cli.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )

    cli._write_service_pid("127.0.0.1", 8765, 4321)

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


def test_stop_service_process_uses_monotonic_deadlines(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": 1000,
        "start_time": "12345",
        "created_at": 1.0,
    }
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")

    alive = iter([True, False])
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: next(alive, False))
    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monotonic_calls = [10.0, 10.1]
    monkeypatch.setattr(
        cli.time,
        "monotonic",
        lambda: (
            monotonic_calls.pop(0) if len(monotonic_calls) > 1 else monotonic_calls[0]
        ),
    )
    monkeypatch.setattr(
        cli.time,
        "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
    )

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0.5)

    assert stopped is True
    assert kills == [(4321, cli.signal.SIGTERM)]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


def test_stop_service_process_sigkill_wait_uses_monotonic_deadline(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": 1000,
        "start_time": "12345",
        "created_at": 1.0,
    }
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")

    alive = {"value": True}
    kills = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))
        if sig == cli.signal.SIGKILL:
            alive["value"] = False

    monkeypatch.setattr(cli, "_pid_alive", lambda pid: alive["value"])
    monkeypatch.setattr(cli.os, "kill", fake_kill)
    monotonic_calls = [10.0, 11.0, 11.0, 11.1]
    monkeypatch.setattr(
        cli.time,
        "monotonic",
        lambda: (
            monotonic_calls.pop(0) if len(monotonic_calls) > 1 else monotonic_calls[0]
        ),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli.time,
        "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
    )

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0.5)

    assert stopped is True
    assert kills == [(4321, cli.signal.SIGTERM), (4321, cli.signal.SIGKILL)]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


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
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("pid", 4321.9),
        ("pid", True),
        ("port", 8765.0),
        ("port", True),
    ],
)
def test_stop_service_process_rejects_coerced_ownership_record_numbers(
    monkeypatch, tmp_path, field, invalid_value
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": 1000,
        "start_time": "12345",
    }
    payload[field] = invalid_value
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")

    alive = {"value": True}
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: alive["value"])
    kills = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))
        alive["value"] = False

    monkeypatch.setattr(cli.os, "kill", fake_kill)

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0.5)

    assert stopped is False
    assert kills == []


@pytest.mark.parametrize(
    ("recorded_uid", "current_uid", "recorded_start", "current_start"),
    [
        (1000.0, 1000, "12345", "12345"),
        (True, 1, "12345", "12345"),
        (1000, 1000, "", ""),
        (1000, 1000, 12345, 12345),
        (1000, 1000, True, True),
    ],
)
def test_stop_service_process_rejects_malformed_identity_record_components(
    monkeypatch,
    tmp_path,
    recorded_uid,
    current_uid,
    recorded_start,
    current_start,
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": recorded_uid,
        "start_time": recorded_start,
    }
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: current_uid)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: current_start)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == []
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


@pytest.mark.parametrize("invalid_port", [0, -1])
def test_read_service_pid_record_rejects_non_positive_port(
    monkeypatch, tmp_path, invalid_port
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": invalid_port,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": 1000,
        "start_time": "12345",
    }
    cli._service_pid_path("127.0.0.1", 8765).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert cli._read_service_pid_record("127.0.0.1", 8765) is None


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
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


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


@pytest.mark.parametrize(
    ("uid", "start_time"),
    [
        (None, "12345"),
        (1000, None),
    ],
)
def test_stop_service_process_rejects_unknown_recorded_identity_components(
    monkeypatch, tmp_path, uid, start_time
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    payload = {
        "pid": 4321,
        "host": "127.0.0.1",
        "port": 8765,
        "argv": cli._service_command("127.0.0.1", 8765),
        "uid": uid,
        "start_time": start_time,
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
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


@pytest.mark.parametrize(
    ("current_uid", "current_start_time"),
    [
        (None, "12345"),
        (1000, None),
    ],
)
def test_stop_service_process_rejects_unknown_current_identity_components(
    monkeypatch, tmp_path, current_uid, current_start_time
):
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")
    cli._write_service_pid("127.0.0.1", 8765, 4321)
    monkeypatch.setattr(
        cli, "_process_cmdline", lambda pid: cli._service_command("127.0.0.1", 8765)
    )
    monkeypatch.setattr(cli, "_process_uid", lambda pid: current_uid)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: current_start_time)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    kills = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    stopped = cli._stop_service_process("127.0.0.1", 8765, timeout=0)

    assert stopped is False
    assert kills == []
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


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
    monkeypatch.setattr(cli, "_process_uid", lambda pid: 1000)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: "12345")

    cli._write_service_pid("127.0.0.1", 8765, 4321)

    payload = json.loads(
        cli._service_pid_path("127.0.0.1", 8765).read_text(encoding="utf-8")
    )
    assert payload["pid"] == 4321
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8765
    assert payload["argv"] == cli._service_command("127.0.0.1", 8765)
    assert payload["uid"] == 1000
    assert payload["start_time"] == "12345"
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


def test_start_service_process_terminates_spawned_process_when_pid_write_fails(
    monkeypatch, tmp_path
):
    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.terminated = 0
            self.killed = 0
            self.wait_timeouts = []

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)

        def kill(self):
            self.killed += 1

    process = FakeProcess()
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        cli,
        "_write_service_pid",
        lambda host, port, pid: (_ for _ in ()).throw(OSError("pid write failed")),
    )

    with pytest.raises(OSError, match="pid write failed"):
        cli._start_service_process("127.0.0.1", 8765)

    assert process.terminated == 1
    assert process.killed == 0
    assert process.wait_timeouts == [1.0]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()


def test_start_service_process_preserves_existing_pid_record_when_pid_write_fails(
    monkeypatch, tmp_path
):
    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.terminated = 0
            self.wait_timeouts = []

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)

        def kill(self):
            raise AssertionError("clean terminate should not need kill")

    process = FakeProcess()
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    pid_path = cli._service_pid_path("127.0.0.1", 8765)
    pid_path.write_text('{"pid": 1111, "legacy": true}', encoding="utf-8")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        cli,
        "_write_service_pid",
        lambda host, port, pid: (_ for _ in ()).throw(OSError("pid write failed")),
    )

    with pytest.raises(OSError, match="pid write failed"):
        cli._start_service_process("127.0.0.1", 8765)

    assert process.terminated == 1
    assert process.wait_timeouts == [1.0]
    assert pid_path.read_text(encoding="utf-8") == '{"pid": 1111, "legacy": true}'


def test_start_service_process_aborts_before_spawn_when_pid_snapshot_fails(
    monkeypatch, tmp_path
):
    class UnreadablePidPath:
        def read_bytes(self):
            raise OSError("snapshot failed")

        def write_bytes(self, data):
            raise AssertionError("unknown prior PID record should not be overwritten")

        def unlink(self, missing_ok=False):
            raise AssertionError("unknown prior PID record should not be cleared")

    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli, "_service_pid_path", lambda host, port: UnreadablePidPath()
    )
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("service should not spawn when PID snapshot fails")
        ),
    )

    with pytest.raises(OSError, match="snapshot failed"):
        cli._start_service_process("127.0.0.1", 8765)


def test_start_service_process_waits_when_terminate_signal_fails(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.terminate_calls = 0
            self.wait_timeouts = []
            self.killed = 0

        def terminate(self):
            self.terminate_calls += 1
            raise OSError("already exited")

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)

        def kill(self):
            self.killed += 1

    process = FakeProcess()
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        cli,
        "_write_service_pid",
        lambda host, port, pid: (_ for _ in ()).throw(OSError("pid write failed")),
    )

    with pytest.raises(OSError, match="pid write failed"):
        cli._start_service_process("127.0.0.1", 8765)

    assert process.terminate_calls == 1
    assert process.wait_timeouts == [1.0]
    assert process.killed == 0


@pytest.mark.parametrize(
    ("uid", "start_time"),
    [(None, "12345"), (1000, None), (True, "12345"), (1000, "")],
)
def test_start_service_process_terminates_spawned_process_when_identity_is_unknown(
    monkeypatch, tmp_path, uid, start_time
):
    class FakeProcess:
        pid = 4321

        def __init__(self):
            self.terminated = 0
            self.wait_timeouts = []

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)

        def kill(self):
            raise AssertionError("clean terminate should not need kill")

    process = FakeProcess()
    monkeypatch.setattr(cli, "_runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(cli, "_process_uid", lambda pid: uid)
    monkeypatch.setattr(cli, "_process_start_identity", lambda pid: start_time)

    with pytest.raises(
        RuntimeError,
        match="Cannot verify KeepGPU service daemon ownership",
    ):
        cli._start_service_process("127.0.0.1", 8765)

    assert process.terminated == 1
    assert process.wait_timeouts == [1.0]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()
