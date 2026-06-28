"""CLI entrypoint for KeepGPU."""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from http.client import InvalidURL
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer
from rich.console import Console

from keep_gpu.utilities.humanized_input import parse_vram_to_elements
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
    validate_job_id,
)

DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8765
SERVICE_HOST_ERROR = "host must be a DNS hostname or IPv4 address"
SERVICE_PORT_ERROR = "port must be an integer between 1 and 65535"
ROOT_BLOCKING_OPTION_LABELS = {
    "gpu_ids": "--gpu-ids",
    "vram": "--vram",
    "legacy_threshold": "--threshold",
    "busy_threshold": "--busy-threshold/--util-threshold",
    "interval": "--interval",
}

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Keep GPUs active with blocking or service-driven workflows.",
)
console = Console()
logger = setup_logger(__name__)


class ServiceUnreachableError(RuntimeError):
    """Raised when the local service cannot be reached."""


class ServiceResponseError(RuntimeError):
    """Raised when the local service responds with an invalid HTTP payload."""


class ServiceRPCError(RuntimeError):
    """Raised when the local service returns a JSON-RPC error."""


def _runtime_dir() -> Path:
    runtime_dir = Path.home() / ".keepgpu"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _service_log_path(host: str, port: int) -> Path:
    return _runtime_dir() / f"service-{host.replace('.', '_')}-{port}.log"


def _service_pid_path(host: str, port: int) -> Path:
    return _runtime_dir() / f"service-{host.replace('.', '_')}-{port}.pid"


def _service_command(host: str, port: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        "keep_gpu.mcp.server",
        "--mode",
        "http",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _process_start_identity(pid: int) -> Optional[str]:
    try:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists():
            return None
        raw_stat = stat_path.read_text(encoding="utf-8", errors="replace")
        after_comm = raw_stat.rsplit(")", 1)[1].strip().split()
        return after_comm[19]
    except Exception:
        return None


def _process_uid(pid: int) -> Optional[int]:
    try:
        return Path(f"/proc/{pid}").stat().st_uid
    except Exception:
        try:
            out = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "uid="],
                text=True,
            )
            return int(out.strip())
        except Exception:
            return None


def _process_cmdline(pid: int) -> List[str]:
    try:
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.exists():
            raw = proc_cmdline.read_bytes()
            return [
                part.decode("utf-8", errors="replace")
                for part in raw.split(b"\x00")
                if part
            ]
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
        ).strip()
        return shlex.split(command)
    except Exception:
        return []


def _is_keepgpu_service_argv(
    argv: List[str], host: Optional[str] = None, port: Optional[int] = None
) -> bool:
    expected_tail = [
        "-m",
        "keep_gpu.mcp.server",
        "--mode",
        "http",
    ]
    if host is not None and port is not None:
        expected_tail.extend(["--host", host, "--port", str(port)])
    if len(argv) != len(expected_tail) + 1:
        return False
    return argv[1:] == expected_tail


def _build_service_pid_record(host: str, port: int, pid: int) -> Dict[str, Any]:
    return {
        "pid": pid,
        "host": host,
        "port": port,
        "argv": _service_command(host, port),
        "uid": _process_uid(pid),
        "start_time": _process_start_identity(pid),
        "created_at": time.time(),
    }


def _read_service_pid_record(host: str, port: int) -> Optional[Dict[str, Any]]:
    pid_path = _service_pid_path(host, port)
    if not pid_path.exists():
        return None
    try:
        raw = pid_path.read_text(encoding="utf-8").strip()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(payload, int):
        return {"pid": payload, "legacy": True}
    if not isinstance(payload, dict):
        return None
    try:
        payload["pid"] = int(payload["pid"])
        payload["port"] = int(payload["port"])
    except (KeyError, TypeError, ValueError):
        return None
    return payload


def _read_service_pid(host: str, port: int) -> Optional[int]:
    record = _read_service_pid_record(host, port)
    if record is None:
        return None
    return record["pid"]


def _write_service_pid(host: str, port: int, pid: int) -> None:
    record = _build_service_pid_record(host, port, pid)
    _service_pid_path(host, port).write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )


def _clear_service_pid(host: str, port: int) -> None:
    _service_pid_path(host, port).unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _apply_legacy_threshold(
    vram_value: str, legacy_threshold: Optional[str], busy_threshold: int
) -> Tuple[str, int, Optional[str]]:
    """
    Interpret the deprecated --threshold flag.

    - If the value parses as int, treat it as a busy-threshold override.
    - Otherwise treat it as a VRAM override.

    Returns:
        (vram, busy_threshold, mode) where mode is "busy", "vram", or None.
    """
    if legacy_threshold is None:
        return vram_value, busy_threshold, None

    try:
        parsed_threshold = int(legacy_threshold)
    except ValueError:
        return legacy_threshold, busy_threshold, "vram"
    return vram_value, parsed_threshold, "busy"


def _parse_gpu_ids(gpu_ids: Optional[str]) -> Optional[List[int]]:
    if gpu_ids is None:
        return None
    if gpu_ids.strip() == "":
        raise typer.BadParameter(
            "gpu_ids must not be empty; omit --gpu-ids to use all visible GPUs"
        )
    try:
        parsed = [int(i.strip()) for i in gpu_ids.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid characters in --gpu-ids '{gpu_ids}'. "
            "Use comma-separated visible ordinals."
        ) from exc
    try:
        return validate_gpu_ids(parsed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_cli_interval(interval: Any) -> Union[int, float]:
    if isinstance(interval, bool):
        raise typer.BadParameter("interval must be finite and positive")
    if isinstance(interval, str):
        normalized = interval.strip()
        if not normalized:
            raise typer.BadParameter("interval must be finite and positive")
        try:
            try:
                interval = int(normalized)
            except ValueError:
                interval = float(normalized)
        except ValueError as exc:
            raise typer.BadParameter("interval must be finite and positive") from exc
    if isinstance(interval, float) and interval.is_integer():
        interval = int(interval)
    try:
        return validate_interval(interval)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_cli_vram(vram: str) -> str:
    try:
        parse_vram_to_elements(vram)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return vram


def _validate_cli_busy_threshold(busy_threshold: int) -> int:
    try:
        return validate_busy_threshold(busy_threshold)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_cli_job_id(job_id: Optional[str]) -> Optional[str]:
    try:
        return validate_job_id(job_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_cli_service_host(host: str) -> str:
    def _is_dns_hostname(value: str) -> bool:
        if len(value) > 253 or value.endswith("."):
            return False
        labels = value.split(".")
        if labels[-1].isdigit():
            return False
        for label in labels:
            if not 1 <= len(label) <= 63:
                return False
            if label.startswith("-") or label.endswith("-"):
                return False
            if not all(
                char.isascii() and (char.isalnum() or char == "-") for char in label
            ):
                return False
        return True

    if not isinstance(host, str) or host.strip() != host or not host:
        raise typer.BadParameter(SERVICE_HOST_ERROR)
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        if not _is_dns_hostname(host):
            raise typer.BadParameter(SERVICE_HOST_ERROR) from None
    else:
        if parsed_ip.version != 4:
            raise typer.BadParameter(SERVICE_HOST_ERROR)
    return host


def _validate_cli_service_port(port: int) -> int:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise typer.BadParameter(SERVICE_PORT_ERROR)
    return port


def _validate_cli_service_endpoint(host: str, port: int) -> Tuple[str, int]:
    return _validate_cli_service_host(host), _validate_cli_service_port(port)


def _is_commandline_parameter_source(source: Any) -> bool:
    # Typer can return its vendored Click enum or a string; compare semantics.
    return getattr(source, "name", source) == "COMMANDLINE"


def _reject_root_blocking_options_before_subcommand(ctx: typer.Context) -> None:
    explicit_options = [
        option_label
        for param_name, option_label in ROOT_BLOCKING_OPTION_LABELS.items()
        if _is_commandline_parameter_source(ctx.get_parameter_source(param_name))
    ]
    if not explicit_options:
        return

    option_text = ", ".join(explicit_options)
    raise typer.BadParameter(
        f"Root blocking option(s) {option_text} were placed before the "
        f"`{ctx.invoked_subcommand}` service subcommand. Omit blocking-mode "
        "root options before service subcommands; for start options, place "
        "them after `start`, for example: `keep-gpu start --gpu-ids 0`."
    )


def _service_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _http_json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    data = None
    headers = {"content-type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    try:
        request = Request(url=url, data=data, headers=headers, method=method)
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            body = response.read().decode("utf-8")
            if not body:
                return {}
            try:
                return json.loads(body)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ServiceResponseError(
                    f"Non-JSON response from service endpoint: {url}"
                ) from exc
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = body or str(exc)
        raise ServiceResponseError(f"Service HTTP error: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ServiceUnreachableError(
            f"Cannot reach KeepGPU service at {url}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ServiceResponseError(
            f"Invalid UTF-8 response from service endpoint: {url}"
        ) from exc
    except (InvalidURL, ValueError) as exc:
        raise ServiceUnreachableError(
            f"Cannot reach KeepGPU service at {url}: {exc}"
        ) from exc


def _service_available(host: str, port: int) -> bool:
    try:
        payload = _http_json_request("GET", f"{_service_base_url(host, port)}/health")
        return bool(payload.get("ok"))
    except Exception:
        return False


def _start_service_process(host: str, port: int) -> int:
    log_path = _service_log_path(host, port)
    with log_path.open("ab") as log_file:
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": log_file,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(_service_command(host, port), **popen_kwargs)
    _write_service_pid(host, port, process.pid)
    return process.pid


def _ensure_service_running(host: str, port: int, auto_start: bool = True) -> bool:
    if _service_available(host, port):
        return False

    stale_pid = _read_service_pid(host, port)
    if stale_pid and not _pid_alive(stale_pid):
        _clear_service_pid(host, port)

    if not auto_start:
        raise RuntimeError(
            f"KeepGPU service is unavailable at {host}:{port}. Start it with `keep-gpu serve`."
        )

    _start_service_process(host, port)
    for _ in range(30):
        if _service_available(host, port):
            return True
        time.sleep(0.2)

    raise RuntimeError(
        f"Failed to auto-start KeepGPU service at {host}:{port}. Try `keep-gpu serve` manually."
    )


def _record_matches_running_process(
    record: Dict[str, Any], host: str, port: int
) -> bool:
    if record.get("legacy"):
        return False
    if "uid" not in record or "start_time" not in record:
        return False
    if record.get("host") != host or record.get("port") != port:
        return False
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
        return False
    if not _is_keepgpu_service_argv(argv, host, port):
        return False
    pid = record["pid"]
    if _process_cmdline(pid) != argv:
        return False

    recorded_uid = record.get("uid")
    current_uid = _process_uid(pid)
    if recorded_uid != current_uid:
        return False

    recorded_start = record.get("start_time")
    current_start = _process_start_identity(pid)
    if recorded_start != current_start:
        return False

    return True


def _stop_service_process(host: str, port: int, timeout: float = 3.0) -> bool:
    record = _read_service_pid_record(host, port)
    if record is None:
        return False
    pid = record["pid"]

    if not _record_matches_running_process(record, host, port):
        _clear_service_pid(host, port)
        return False

    if not _pid_alive(pid):
        _clear_service_pid(host, port)
        return True

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _clear_service_pid(host, port)
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            _clear_service_pid(host, port)
            return True
        time.sleep(0.1)

    if not _record_matches_running_process(record, host, port):
        _clear_service_pid(host, port)
        return False

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        _clear_service_pid(host, port)
        return False
    deadline = time.time() + max(0.5, min(timeout, 3.0))
    while time.time() < deadline:
        if not _pid_alive(pid):
            _clear_service_pid(host, port)
            return True
        if not _record_matches_running_process(record, host, port):
            _clear_service_pid(host, port)
            return False
        time.sleep(0.1)
    return False


def _rpc_call(
    method: str,
    params: Optional[Dict[str, Any]],
    host: str,
    port: int,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    payload = {
        "id": int(time.time() * 1000),
        "method": method,
        "params": params or {},
    }
    response = _http_json_request(
        "POST", f"{_service_base_url(host, port)}/rpc", payload, timeout=timeout
    )
    if not isinstance(response, dict):
        raise ServiceResponseError(
            "Malformed JSON-RPC response: response must be an object"
        )
    if response.get("jsonrpc") != "2.0":
        raise ServiceResponseError("Malformed JSON-RPC response: jsonrpc must be 2.0")
    if "error" in response and "result" in response:
        raise ServiceResponseError(
            "Malformed JSON-RPC response: both error and result members are present"
        )
    if "error" in response:
        error = response["error"]
        if not isinstance(error, dict):
            raise ServiceResponseError(
                "Malformed JSON-RPC response: error must be an object"
            )
        if "id" not in response:
            raise ServiceResponseError("Malformed JSON-RPC response: missing id")
        if response.get("id") not in (payload["id"], None):
            raise ServiceResponseError("Malformed JSON-RPC response: mismatched id")
        raise ServiceRPCError(error.get("message", str(error)))
    if response.get("id") != payload["id"]:
        raise ServiceResponseError("Malformed JSON-RPC response: mismatched id")
    if "result" not in response:
        raise ServiceResponseError("Malformed JSON-RPC response: missing result")
    result = response["result"]
    if not isinstance(result, dict):
        raise ServiceResponseError(
            "Malformed JSON-RPC response: result must be an object"
        )
    return result


def _malformed_method_result(method: str, detail: str) -> ServiceResponseError:
    return ServiceResponseError(f"Malformed {method} response: {detail}")


def _require_list_field(result: Dict[str, Any], field: str, method: str) -> List[Any]:
    value = result.get(field)
    if not isinstance(value, list):
        raise _malformed_method_result(method, f"{field} must be a list")
    return value


def _require_dict_field(
    result: Dict[str, Any], field: str, method: str
) -> Dict[str, Any]:
    value = result.get(field)
    if not isinstance(value, dict):
        raise _malformed_method_result(method, f"{field} must be an object")
    return value


def _validate_status_result(
    result: Dict[str, Any], *, single_job: bool
) -> Dict[str, Any]:
    if single_job:
        if not isinstance(result.get("active"), bool):
            raise _malformed_method_result("status", "active must be a bool")
        if not isinstance(result.get("job_id"), str):
            raise _malformed_method_result("status", "job_id must be a string")
    else:
        _require_list_field(result, "active_jobs", "status")
    return result


def _validate_stop_keep_result(result: Dict[str, Any]) -> Dict[str, Any]:
    for field in ("stopped", "timed_out", "failed"):
        _require_list_field(result, field, "stop_keep")
    _require_dict_field(result, "errors", "stop_keep")
    message = result.get("message")
    if message is not None and not isinstance(message, str):
        raise _malformed_method_result("stop_keep", "message must be a string")
    return result


def _validate_list_gpus_result(result: Dict[str, Any]) -> Dict[str, Any]:
    gpus = _require_list_field(result, "gpus", "list_gpus")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            raise _malformed_method_result(
                "list_gpus", f"gpus[{index}] must be an object"
            )
    return result


def _is_service_unreachable_error(exc: RuntimeError) -> bool:
    if isinstance(exc, ServiceUnreachableError):
        return True
    if isinstance(exc, (ServiceRPCError, ServiceResponseError)):
        return False
    message = str(exc).lower()
    return "cannot reach keepgpu service" in message or "timed out" in message


def _stop_all_sessions_with_fallback(host: str, port: int) -> Dict[str, Any]:
    try:
        result = _rpc_call("stop_keep", {}, host, port, timeout=45.0)
        return _validate_stop_keep_result(result)
    except RuntimeError as exc:
        if not _is_service_unreachable_error(exc):
            raise
        managed_pid = _read_service_pid(host, port)
        if managed_pid and _pid_alive(managed_pid):
            if _stop_service_process(host, port):
                result = {
                    "stopped": [],
                    "timed_out": [],
                    "failed": [],
                    "errors": {},
                    "message": (
                        "Service stop RPC timed out; force-stopped local daemon "
                        f"pid={managed_pid}. Reserved VRAM should be released by process exit."
                    ),
                }
                return _validate_stop_keep_result(result)
            raise RuntimeError(
                f"{exc}. No ownership-verified daemon could be force-stopped."
            ) from exc
        raise RuntimeError(
            f"{exc}. If service is unresponsive, run `keep-gpu service-stop --force`."
        ) from exc


def _run_blocking(
    interval: Union[int, float],
    gpu_ids: Optional[str],
    vram: str,
    legacy_threshold: Optional[str],
    busy_threshold: int,
) -> None:
    vram, busy_threshold, legacy_mode = _apply_legacy_threshold(
        vram, legacy_threshold, busy_threshold
    )
    interval = _validate_cli_interval(interval)
    vram = _validate_cli_vram(vram)
    if legacy_mode == "vram":
        console.print(
            "[yellow]`--threshold` for VRAM is deprecated; use `--vram`.[/yellow]"
        )
    elif legacy_mode == "busy":
        console.print(
            "[yellow]`--threshold` for utilization is deprecated; use `--busy-threshold`.[/yellow]"
        )
    busy_threshold = _validate_cli_busy_threshold(busy_threshold)

    gpu_id_list = _parse_gpu_ids(gpu_ids)

    import torch

    from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

    if gpu_id_list is not None:
        logger.info("Using specified visible GPU ordinals: %s", gpu_id_list)
        gpu_count = len(gpu_id_list)
    else:
        gpu_count = torch.cuda.device_count()
        logger.info("Using all available GPUs")

    logger.info("GPU count: %s", gpu_count)
    logger.info("VRAM to keep occupied: %s", vram)
    logger.info("Check interval: %s seconds", interval)
    if busy_threshold == -1:
        logger.info("Busy threshold: unconditional (utilization backoff disabled)")
    else:
        logger.info("Busy threshold: %s%%", busy_threshold)

    global_controller = GlobalGPUController(
        gpu_ids=gpu_id_list,
        interval=interval,
        vram_to_keep=vram,
        busy_threshold=busy_threshold,
    )

    with global_controller:
        logger.info("Keeping GPUs awake. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Interruption received. Releasing GPUs...")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    interval: str = typer.Option(
        "300",
        metavar="NUMBER",
        help="Interval in seconds between GPU usage checks (blocking mode only).",
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated visible GPU ordinals for blocking mode (default: all).",
    ),
    vram: str = typer.Option(
        "1GiB",
        "--vram",
        help="Amount of VRAM to keep occupied (blocking mode).",
    ),
    legacy_threshold: Optional[str] = typer.Option(
        None,
        "--threshold",
        hidden=True,
        help="Deprecated alias: numeric maps to busy-threshold, string maps to vram.",
    ),
    busy_threshold: int = typer.Option(
        DEFAULT_BUSY_THRESHOLD,
        "--busy-threshold",
        "--util-threshold",
        help=(
            "Back off when utilization is above this 0..100 percent threshold "
            "or telemetry is unavailable; -1 disables utilization backoff "
            "(blocking mode)."
        ),
    ),
):
    """Run blocking keep-alive mode when no subcommand is provided."""
    try:
        if ctx.invoked_subcommand is not None:
            _reject_root_blocking_options_before_subcommand(ctx)
            return
        interval = _validate_cli_interval(interval)
        _parse_gpu_ids(gpu_ids)
        _run_blocking(interval, gpu_ids, vram, legacy_threshold, busy_threshold)
    except typer.BadParameter as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@app.command("serve")
def serve(
    host: str = typer.Option(
        DEFAULT_SERVICE_HOST,
        help="Host interface for KeepGPU local service.",
    ),
    port: int = typer.Option(
        DEFAULT_SERVICE_PORT,
        help="Port for KeepGPU local service.",
    ),
):
    """Run KeepGPU local service (HTTP + JSON-RPC + dashboard)."""
    from keep_gpu.mcp.server import KeepGPUServer, run_http

    try:
        host, port = _validate_cli_service_endpoint(host, port)
    except typer.BadParameter as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold cyan]Service URL:[/bold cyan] http://{host}:{port}/")
    console.print(
        "[dim]Press Ctrl+C to stop the foreground service, or use `keep-gpu service-stop` for auto-started daemons.[/dim]"
    )
    run_http(KeepGPUServer(), host=host, port=port)


@app.command("start")
def start(
    gpu_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated visible GPU ordinals.",
    ),
    vram: str = typer.Option("1GiB", "--vram", help="VRAM to keep per GPU."),
    interval: str = typer.Option(
        "300", metavar="NUMBER", help="Interval in seconds between checks."
    ),
    busy_threshold: int = typer.Option(
        DEFAULT_BUSY_THRESHOLD,
        "--busy-threshold",
        "--util-threshold",
        help=(
            "Back off when utilization is above this 0..100 percent threshold "
            "or telemetry is unavailable; -1 disables utilization backoff."
        ),
    ),
    job_id: Optional[str] = typer.Option(
        None,
        help="Optional custom job id. Auto-generated when omitted.",
    ),
    host: str = typer.Option(
        DEFAULT_SERVICE_HOST,
        "--host",
        help="KeepGPU service host.",
    ),
    port: int = typer.Option(
        DEFAULT_SERVICE_PORT,
        "--port",
        help="KeepGPU service port.",
    ),
    auto_start: bool = typer.Option(
        True,
        "--auto-start/--no-auto-start",
        help="Auto-start local service when unavailable.",
    ),
):
    """Start a non-blocking keep session and return a job id.

    Use `keep-gpu stop --job-id <id>` to release this session and
    `keep-gpu service-stop` to stop the local service daemon.
    """
    try:
        host, port = _validate_cli_service_endpoint(host, port)
        interval = _validate_cli_interval(interval)
        busy_threshold = _validate_cli_busy_threshold(busy_threshold)
        parsed_gpu_ids = _parse_gpu_ids(gpu_ids)
        _validate_cli_vram(vram)
        job_id = _validate_cli_job_id(job_id)
        auto_started = _ensure_service_running(host, port, auto_start=auto_start)
        result = _rpc_call(
            "start_keep",
            {
                "gpu_ids": parsed_gpu_ids,
                "vram": vram,
                "interval": interval,
                "busy_threshold": busy_threshold,
                "job_id": job_id,
            },
            host,
            port,
        )
        result_job_id = result.get("job_id")
        if result_job_id is None:
            raise ServiceResponseError(
                "Malformed JSON-RPC response: start_keep result must include job_id"
            )
        try:
            result_job_id = validate_job_id(result_job_id)
        except ValueError as exc:
            raise ServiceResponseError(f"Malformed JSON-RPC response: {exc}") from exc
        if auto_started:
            console.print(
                f"[bold cyan]Auto-started KeepGPU service[/bold cyan] at http://{host}:{port}/"
            )
        console.print(
            f"[bold green]Started keep session[/bold green] job_id={result_job_id}"
        )
        console.print(f"[cyan]Dashboard:[/cyan] http://{host}:{port}/")
        console.print(
            f"[dim]Next: keep-gpu status --job-id {result_job_id} | keep-gpu stop --job-id {result_job_id}[/dim]"
        )
        console.print(
            "[dim]When all sessions are done, stop daemon with: keep-gpu service-stop[/dim]"
        )
    except (RuntimeError, typer.BadParameter) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@app.command("status")
def status(
    job_id: Optional[str] = typer.Option(None, help="Session id to inspect."),
    host: str = typer.Option(DEFAULT_SERVICE_HOST, "--host", help="Service host."),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port", help="Service port."),
):
    """Show session status from KeepGPU local service."""
    try:
        host, port = _validate_cli_service_endpoint(host, port)
        job_id = _validate_cli_job_id(job_id)
        result = _rpc_call(
            "status",
            {} if job_id is None else {"job_id": job_id},
            host,
            port,
        )
        result = _validate_status_result(result, single_job=job_id is not None)
        console.print_json(data=result)
    except (RuntimeError, typer.BadParameter) as exc:
        console.print_json(data={"error": str(exc)})
        raise typer.Exit(code=1) from exc


@app.command("stop")
def stop(
    job_id: Optional[str] = typer.Option(
        None,
        help="Session id to stop. Omit with --all to stop every session.",
    ),
    all_sessions: bool = typer.Option(
        False,
        "--all",
        help="Stop all sessions.",
    ),
    host: str = typer.Option(DEFAULT_SERVICE_HOST, "--host", help="Service host."),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port", help="Service port."),
):
    """Stop one session or all sessions."""
    try:
        if job_id is not None and all_sessions:
            raise RuntimeError("Use either --job-id or --all, not both.")
        if job_id is None and not all_sessions:
            raise RuntimeError("Provide --job-id or use --all.")
        host, port = _validate_cli_service_endpoint(host, port)
        job_id = _validate_cli_job_id(job_id)
        if all_sessions:
            result = _stop_all_sessions_with_fallback(host, port)
        else:
            result = _rpc_call(
                "stop_keep",
                {"job_id": job_id},
                host,
                port,
                timeout=45.0,
            )
            result = _validate_stop_keep_result(result)
        console.print_json(data=result)
    except (RuntimeError, typer.BadParameter) as exc:
        console.print_json(data={"error": str(exc)})
        raise typer.Exit(code=1) from exc


@app.command("list-gpus")
def list_gpus(
    host: str = typer.Option(DEFAULT_SERVICE_HOST, "--host", help="Service host."),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port", help="Service port."),
):
    """List GPU telemetry from local service."""
    try:
        host, port = _validate_cli_service_endpoint(host, port)
        result = _rpc_call("list_gpus", {}, host, port)
        result = _validate_list_gpus_result(result)
        console.print_json(data=result)
    except (RuntimeError, typer.BadParameter) as exc:
        console.print_json(data={"error": str(exc)})
        raise typer.Exit(code=1) from exc


@app.command("service-stop")
def service_stop(
    host: str = typer.Option(DEFAULT_SERVICE_HOST, "--host", help="Service host."),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port", help="Service port."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Stop service even if tracked sessions exist.",
    ),
):
    """Stop local KeepGPU service daemon started by auto-start logic."""
    try:
        host, port = _validate_cli_service_endpoint(host, port)
        if force:
            stopped = _stop_service_process(host, port)
            if not stopped:
                raise RuntimeError(
                    "No managed daemon PID found. If service was started in foreground, stop it with Ctrl+C in that terminal."
                )
            console.print(
                f"[bold green]Force-stopped KeepGPU service daemon[/bold green] at http://{host}:{port}/"
            )
            return

        try:
            status = _rpc_call("status", {}, host, port)
            status = _validate_status_result(status, single_job=False)
        except ServiceUnreachableError as exc:
            raise RuntimeError(
                f"KeepGPU service is unavailable at {host}:{port}. Non-force service-stop must verify no tracked keep sessions before stopping the daemon. "
                "For an unresponsive auto-started daemon, run `keep-gpu service-stop --force`."
            ) from exc

        active_jobs = status.get("active_jobs", [])
        if active_jobs:
            raise RuntimeError(
                "Tracked keep sessions detected. Stop sessions first (`keep-gpu stop --all`) or re-run with --force."
            )
        stop_result = _rpc_call("stop_keep", {}, host, port, timeout=45.0)
        _validate_stop_keep_result(stop_result)

        stopped = _stop_service_process(host, port)
        if not stopped:
            raise RuntimeError(
                "No managed daemon PID found. If service was started in foreground, stop it with Ctrl+C in that terminal."
            )

        console.print(
            f"[bold green]Stopped KeepGPU service daemon[/bold green] at http://{host}:{port}/"
        )
    except (RuntimeError, typer.BadParameter) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
