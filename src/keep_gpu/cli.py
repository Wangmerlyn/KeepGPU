"""CLI entrypoint for KeepGPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
import typer
from rich.console import Console

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.mcp.server import KeepGPUServer, run_http
from keep_gpu.utilities.logger import setup_logger

DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8765

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Keep GPUs active with blocking or service-driven workflows.",
)
console = Console()
logger = setup_logger(__name__)


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
    if not gpu_ids:
        return None
    try:
        return [int(i.strip()) for i in gpu_ids.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid characters in --gpu-ids '{gpu_ids}'. Use comma-separated integers."
        ) from exc


def _service_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _http_json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    data = None
    headers = {"content-type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def _service_available(host: str, port: int) -> bool:
    try:
        payload = _http_json_request("GET", f"{_service_base_url(host, port)}/health")
        return bool(payload.get("ok"))
    except Exception:
        return False


def _start_service_process(host: str, port: int) -> None:
    runtime_dir = Path.home() / ".keepgpu"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / f"service-{host.replace('.', '_')}-{port}.log"
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "keep_gpu.mcp.server",
                "--mode",
                "http",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )


def _ensure_service_running(host: str, port: int, auto_start: bool = True) -> None:
    if _service_available(host, port):
        return

    if not auto_start:
        raise RuntimeError(
            f"KeepGPU service is unavailable at {host}:{port}. Start it with `keep-gpu serve`."
        )

    _start_service_process(host, port)
    for _ in range(30):
        if _service_available(host, port):
            return
        time.sleep(0.2)

    raise RuntimeError(
        f"Failed to auto-start KeepGPU service at {host}:{port}. Try `keep-gpu serve` manually."
    )


def _rpc_call(
    method: str,
    params: Optional[Dict[str, Any]],
    host: str,
    port: int,
) -> Dict[str, Any]:
    payload = {
        "id": int(time.time() * 1000),
        "method": method,
        "params": params or {},
    }
    response = _http_json_request(
        "POST", f"{_service_base_url(host, port)}/rpc", payload
    )
    if "error" in response:
        error = response["error"]
        raise RuntimeError(error.get("message", str(error)))
    return response.get("result", {})


def _run_blocking(
    interval: int,
    gpu_ids: Optional[str],
    vram: str,
    legacy_threshold: Optional[str],
    busy_threshold: int,
) -> None:
    vram, busy_threshold, legacy_mode = _apply_legacy_threshold(
        vram, legacy_threshold, busy_threshold
    )
    if legacy_mode == "vram":
        console.print(
            "[yellow]`--threshold` for VRAM is deprecated; use `--vram`.[/yellow]"
        )
    elif legacy_mode == "busy":
        console.print(
            "[yellow]`--threshold` for utilization is deprecated; use `--busy-threshold`.[/yellow]"
        )

    gpu_id_list = _parse_gpu_ids(gpu_ids)
    if gpu_id_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_id_list))
        logger.info("Using specified GPUs: %s", gpu_id_list)
        gpu_count = len(gpu_id_list)
    else:
        gpu_count = torch.cuda.device_count()
        logger.info("Using all available GPUs")

    logger.info("GPU count: %s", gpu_count)
    logger.info("VRAM to keep occupied: %s", vram)
    logger.info("Check interval: %s seconds", interval)
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
    interval: int = typer.Option(
        300,
        help="Interval in seconds between GPU usage checks (blocking mode only).",
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated GPU IDs for blocking mode (default: all).",
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
        -1,
        "--busy-threshold",
        "--util-threshold",
        help="Max utilization threshold before backing off (blocking mode).",
    ),
):
    """Run blocking keep-alive mode when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return
    try:
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
    run_http(KeepGPUServer(), host=host, port=port)


@app.command("start")
def start(
    gpu_ids: Optional[str] = typer.Option(None, help="Comma-separated GPU IDs."),
    vram: str = typer.Option("1GiB", "--vram", help="VRAM to keep per GPU."),
    interval: int = typer.Option(300, help="Interval in seconds between checks."),
    busy_threshold: int = typer.Option(
        -1,
        "--busy-threshold",
        "--util-threshold",
        help="Back off when utilization exceeds this percent.",
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
    """Start a non-blocking keep session and return a job id."""
    try:
        _ensure_service_running(host, port, auto_start=auto_start)
        result = _rpc_call(
            "start_keep",
            {
                "gpu_ids": _parse_gpu_ids(gpu_ids),
                "vram": vram,
                "interval": interval,
                "busy_threshold": busy_threshold,
                "job_id": job_id,
            },
            host,
            port,
        )
        console.print(
            f"[bold green]Started keep session[/bold green] job_id={result['job_id']}"
        )
    except (RuntimeError, typer.BadParameter, URLError) as exc:
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
        result = _rpc_call(
            "status",
            {"job_id": job_id} if job_id else {},
            host,
            port,
        )
        console.print_json(data=json.dumps(result))
    except (RuntimeError, URLError) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
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
        if not job_id and not all_sessions:
            raise RuntimeError("Provide --job-id or use --all.")
        result = _rpc_call(
            "stop_keep",
            {} if all_sessions else {"job_id": job_id},
            host,
            port,
        )
        console.print_json(data=json.dumps(result))
    except (RuntimeError, URLError) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@app.command("list-gpus")
def list_gpus(
    host: str = typer.Option(DEFAULT_SERVICE_HOST, "--host", help="Service host."),
    port: int = typer.Option(DEFAULT_SERVICE_PORT, "--port", help="Service port."),
):
    """List GPU telemetry from local service."""
    try:
        result = _rpc_call("list_gpus", {}, host, port)
        console.print_json(data=json.dumps(result))
    except (RuntimeError, URLError) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
