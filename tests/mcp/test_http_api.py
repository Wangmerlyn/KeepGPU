import json
import math
import threading
from typing import Any, cast
from urllib.error import HTTPError
from socketserver import TCPServer, ThreadingMixIn
from urllib.request import Request, urlopen

import pytest

from keep_gpu.mcp.server import KeepGPUServer, _JSONRPCHandler


class DummyController:
    def __init__(self, gpu_ids=None, interval=0, vram_to_keep=None, busy_threshold=0):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.vram_to_keep = vram_to_keep
        self.busy_threshold = busy_threshold
        self.kept = False
        self.released = False

    def keep(self):
        self.kept = True

    def release(self):
        self.released = True


def dummy_factory(**kwargs):
    return DummyController(**kwargs)


class DummyKeepGPUServer(KeepGPUServer):
    def list_gpus(self):
        return {"gpus": [{"id": 0, "name": "GPU 0"}]}


def make_server() -> KeepGPUServer:
    return DummyKeepGPUServer(controller_factory=cast(Any, dummy_factory))


def _start_http_server(server: KeepGPUServer):
    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, thread, base


def _start_bare_http_server():
    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, thread, base


def _start_threaded_http_server(server: KeepGPUServer):
    class _Server(ThreadingMixIn, TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, thread, base


def _request_json(method, url, payload=None):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, method=method)
    request.add_header("content-type", "application/json")
    try:
        with urlopen(request, timeout=2.0) as response:  # nosec B310
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def _request_raw(method, url, data=None):
    request = Request(url=url, data=data, method=method)
    request.add_header("content-type", "application/json")
    try:
        with urlopen(request, timeout=2.0) as response:  # nosec B310
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def test_http_health_and_static_index():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status, payload = _request_json("GET", f"{base}/health")
        assert status == 200
        assert payload["ok"] is True

        request = Request(url=f"{base}/", method="GET")
        with urlopen(request, timeout=2.0) as response:  # nosec B310
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert "KeepGPU Control Deck" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_session_lifecycle():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        _, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        job_id = start_payload["job_id"]

        _, status_payload = _request_json("GET", f"{base}/api/sessions")
        assert status_payload["active_jobs"]
        assert status_payload["active_jobs"][0]["job_id"] == job_id

        _, stop_payload = _request_json("DELETE", f"{base}/api/sessions/{job_id}")
        assert job_id in stop_payload["stopped"]

        _, all_stopped_payload = _request_json("DELETE", f"{base}/api/sessions")
        assert all_stopped_payload["stopped"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_session_start_defaults_to_eco_safe_busy_threshold():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "http-default",
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
            },
        )

        assert status_code == 200
        assert start_payload == {"job_id": "http-default"}
        _, status_payload = _request_json("GET", f"{base}/api/sessions/http-default")
        assert status_payload["params"]["busy_threshold"] == 25
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_session_start_preserves_explicit_unconditional_busy_threshold():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "http-unconditional",
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": -1,
            },
        )

        assert status_code == 200
        assert start_payload == {"job_id": "http-unconditional"}
        _, status_payload = _request_json(
            "GET", f"{base}/api/sessions/http-unconditional"
        )
        assert status_payload["params"]["busy_threshold"] == -1
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_status_reports_runtime_failed_session():
    class RuntimeFailedController(DummyController):
        def runtime_error(self):
            return RuntimeError("rank 0: allocation retries exhausted")

    server = DummyKeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: RuntimeFailedController(**kwargs))
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "http-runtime-failure",
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
            },
        )

        assert status_code == 200
        assert start_payload == {"job_id": "http-runtime-failure"}

        _, session_payload = _request_json(
            "GET", f"{base}/api/sessions/http-runtime-failure"
        )
        assert session_payload["active"] is True
        assert session_payload["state"] == "runtime_failed"
        assert session_payload["last_error"] == "rank 0: allocation retries exhausted"

        _, list_payload = _request_json("GET", f"{base}/api/sessions")
        assert list_payload["active_jobs"] == [
            {
                "job_id": "http-runtime-failure",
                "params": {
                    "gpu_ids": [0],
                    "vram": "256MB",
                    "interval": 20,
                    "busy_threshold": 25,
                },
                "state": "runtime_failed",
                "last_error": "rank 0: allocation retries exhausted",
            }
        ]
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_start_validates_gpu_ids_against_listed_visible_ids(monkeypatch):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    monkeypatch.setattr(
        server,
        "list_gpus",
        lambda: {"gpus": [{"id": 0, "name": "GPU 0"}, {"id": 2, "name": "GPU 2"}]},
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [1],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert "listed visible GPU IDs" in payload["error"]["message"]
    assert controllers == []


def test_http_start_rejects_explicit_gpu_ids_when_no_visible_ids(monkeypatch):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    monkeypatch.setattr(server, "list_gpus", lambda: {"gpus": []})
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert "listed visible GPU IDs (none)" in payload["error"]["message"]
    assert controllers == []


@pytest.mark.parametrize(
    ("request_payload", "message"),
    [
        (
            {
                "gpu_ids": [0],
                "vram": [],
                "interval": 20,
                "busy_threshold": 5,
            },
            "vram_to_keep must be str or int bytes",
        ),
        (
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 0,
                "busy_threshold": 5,
            },
            "interval must be positive",
        ),
        (
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 101,
            },
            "busy_threshold must be -1 or an integer between 0 and 100",
        ),
        (
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
                "job_id": "",
            },
            "job_id must be a URL-path-safe non-empty string",
        ),
    ],
)
def test_http_start_rejects_invalid_fields_before_listing_gpus(
    request_payload, message
):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    list_calls = []

    def fail_list_gpus():
        list_calls.append(True)
        raise AssertionError("list_gpus should not run for invalid input")

    server.list_gpus = fail_list_gpus  # type: ignore[method-assign]
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            request_payload,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert message in payload["error"]["message"]
    assert list_calls == []
    assert controllers == []


@pytest.mark.parametrize(
    ("request_payload", "message"),
    [
        (
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 10**1000,
                "busy_threshold": 5,
            },
            "interval must be no more than",
        ),
        (
            {
                "gpu_ids": [0],
                "vram": 10**1000,
                "interval": 20,
                "busy_threshold": 5,
            },
            "vram must be no more than",
        ),
        (
            {
                "gpu_ids": [0],
                "vram": ("9" * 500) + "GiB",
                "interval": 20,
                "busy_threshold": 5,
            },
            "vram must be no more than",
        ),
    ],
)
def test_http_start_rejects_oversized_numeric_inputs_before_listing_gpus(
    request_payload, message
):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    list_calls = []

    def fail_list_gpus():
        list_calls.append(True)
        raise AssertionError("list_gpus should not run for invalid input")

    server.list_gpus = fail_list_gpus  # type: ignore[method-assign]
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            request_payload,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert message in payload["error"]["message"]
    assert list_calls == []
    assert controllers == []


def test_http_start_rejects_duplicate_job_id_before_listing_gpus():
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    server.start_keep(job_id="existing-job", gpu_ids=[0])
    list_calls = []

    def fail_list_gpus():
        list_calls.append(True)
        raise AssertionError("list_gpus should not run for duplicate job_id")

    server.list_gpus = fail_list_gpus  # type: ignore[method-assign]
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
                "job_id": "existing-job",
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert "job_id existing-job already exists" in payload["error"]["message"]
    assert list_calls == []
    assert len(controllers) == 1


def test_http_status_reports_starting_session_during_controller_keep():
    keep_started = threading.Event()
    keep_release = threading.Event()
    result_holder = {}

    class BlockingStartController(DummyController):
        def keep(self):
            self.kept = True
            keep_started.set()
            keep_release.wait(timeout=1.0)

    server = DummyKeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    httpd, thread, base = _start_threaded_http_server(server)

    def start_session():
        result_holder["response"] = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "starting-job",
                "gpu_ids": [0],
                "vram": "512MB",
                "interval": 7,
                "busy_threshold": 25,
            },
        )

    start_thread = threading.Thread(target=start_session)
    start_thread.start()
    try:
        assert keep_started.wait(timeout=1.0)
        expected_params = {
            "gpu_ids": [0],
            "vram": "512MB",
            "interval": 7,
            "busy_threshold": 25,
        }
        _, list_payload = _request_json("GET", f"{base}/api/sessions")
        assert list_payload["active_jobs"] == [
            {
                "job_id": "starting-job",
                "params": expected_params,
                "state": "starting",
                "last_error": None,
            }
        ]
        _, status_payload = _request_json("GET", f"{base}/api/sessions/starting-job")
        assert status_payload == {
            "active": True,
            "job_id": "starting-job",
            "params": expected_params,
            "state": "starting",
            "last_error": None,
        }
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert not start_thread.is_alive()
    assert result_holder["response"] == (200, {"job_id": "starting-job"})


def test_http_stop_timeout_keeps_session_visible(monkeypatch):
    server = make_server()
    monkeypatch.setattr(server, "_release_with_timeout", lambda controller, **_: False)

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        _, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        job_id = start_payload["job_id"]

        _, stop_payload = _request_json("DELETE", f"{base}/api/sessions/{job_id}")
        assert stop_payload["stopped"] == []
        assert stop_payload["timed_out"] == [job_id]

        _, status_payload = _request_json("GET", f"{base}/api/sessions/{job_id}")
        assert status_payload["active"] is True
        assert status_payload["state"] == "stopping"
        assert "Timed out" in status_payload["last_error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_session_trailing_slash_rejected():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        _, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        job_id = start_payload["job_id"]

        status_code, error_payload = _request_json(
            "DELETE", f"{base}/api/sessions/{job_id}/"
        )
        assert status_code == 400
        assert "Missing job_id" in error_payload["error"]["message"]

        _, status_payload = _request_json("GET", f"{base}/api/sessions")
        assert status_payload["active_jobs"]
        assert status_payload["active_jobs"][0]["job_id"] == job_id
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_unknown_api_route_returns_json_404():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json("GET", f"{base}/api/unknown")
        assert status_code == 404
        assert payload["error"]["message"] == "Unknown endpoint"
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize("data", [None, b"{bad json"])
def test_http_post_unknown_api_route_returns_json_404_before_body_parse(data):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_raw("POST", f"{base}/api/unknown", data)
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 404
    assert payload["error"]["message"] == "Unknown endpoint"


def test_http_get_api_gpus_runtime_error_returns_json_500(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "list_gpus",
        lambda: (_ for _ in ()).throw(RuntimeError("telemetry exploded")),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}/api/gpus")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["message"] == "telemetry exploded"
    assert payload["error"]["type"] == "RuntimeError"


def test_http_delete_sessions_runtime_error_returns_json_500(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "stop_keep",
        lambda **_: (_ for _ in ()).throw(RuntimeError("release exploded")),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("DELETE", f"{base}/api/sessions")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["message"] == "release exploded"
    assert payload["error"]["type"] == "RuntimeError"


def test_http_post_sessions_runtime_type_error_returns_json_500(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "start_keep",
        lambda **_: (_ for _ in ()).throw(TypeError("internal type exploded")),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("POST", f"{base}/api/sessions", {})
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["message"] == "internal type exploded"
    assert payload["error"]["type"] == "TypeError"


def test_http_post_sessions_runtime_value_error_returns_json_500(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "start_keep",
        lambda **_: (_ for _ in ()).throw(ValueError("startup invariant broke")),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("POST", f"{base}/api/sessions", {})
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["message"] == "startup invariant broke"
    assert payload["error"]["type"] == "ValueError"


def test_http_get_setup_runtime_error_returns_json_500():
    httpd, thread, base = _start_bare_http_server()

    try:
        status_code, payload = _request_json("GET", f"{base}/api/gpus")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["type"] == "AttributeError"


def test_http_delete_setup_runtime_error_returns_json_500():
    httpd, thread, base = _start_bare_http_server()

    try:
        status_code, payload = _request_json("DELETE", f"{base}/api/sessions")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["type"] == "AttributeError"


def test_http_post_setup_runtime_error_returns_json_500():
    httpd, thread, base = _start_bare_http_server()

    try:
        status_code, payload = _request_json("POST", f"{base}/api/sessions", {})
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)

    assert status_code == 500
    assert payload["error"]["type"] == "AttributeError"


def test_http_post_rejects_unknown_fields():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
                "unexpected": "value",
            },
        )
        assert status_code == 400
        assert "Unknown request fields" in payload["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize("payload", [[], ["gpu_ids"], "gpu_ids", 1])
def test_http_post_rejects_non_object_json_without_creating_session(payload):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, response = _request_json("POST", f"{base}/api/sessions", payload)

        assert status_code == 400
        assert "JSON body must be an object" in response["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_non_positive_interval():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 0,
                "busy_threshold": 5,
            },
        )
        assert status_code == 400
        assert "interval must be positive" in payload["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_nan_interval_without_creating_session():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": math.nan,
                "busy_threshold": 5,
            },
        )

        assert status_code == 400
        assert "interval must be finite and positive" in payload["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_empty_gpu_ids():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        assert status_code == 400
        assert "gpu_ids must select at least one GPU" in payload["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_duplicate_gpu_ids():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0, 1, 0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        assert status_code == 400
        assert (
            "gpu_ids must not contain duplicate values" in payload["error"]["message"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_busy_threshold_above_percent_range():
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 101,
            },
        )
        assert status_code == 400
        assert (
            "busy_threshold must be -1 or an integer between 0 and 100"
            in payload["error"]["message"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_post_rejects_invalid_job_id_without_creating_session():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
                "job_id": "",
            },
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_get_rejects_invalid_decoded_job_id_path():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}/api/sessions/bad%3Fjob")

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_get_rejects_query_shaped_job_id_path():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "GET", f"{base}/api/sessions/{active_job_id}?bad=query"
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_get_rejects_parameterized_job_id_path():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "GET", f"{base}/api/sessions/{active_job_id};bad"
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_invalid_decoded_job_id_path_without_stopping_session():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("DELETE", f"{base}/api/sessions/bad%2Fjob")

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_query_shaped_job_id_path_without_stopping_session():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "DELETE", f"{base}/api/sessions/{active_job_id}?bad=query"
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_parameterized_job_id_path_without_stopping_session():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "DELETE", f"{base}/api/sessions/{active_job_id};bad"
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_extra_session_path_segment_without_stopping_session():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "DELETE", f"{base}/api/sessions/prefix/{active_job_id}"
        )

        assert status_code == 400
        assert "job_id" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_query_shaped_collection_path_without_stopping_sessions():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("DELETE", f"{base}/api/sessions?bad=query")

        assert status_code == 400
        assert "session path" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


def test_http_delete_rejects_parameterized_collection_path_without_stopping_sessions():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("DELETE", f"{base}/api/sessions;bad")

        assert status_code == 400
        assert "session path" in payload["error"]["message"]
        assert server.status(active_job_id)["active"] is True
        assert controller.released is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)
