import json
import math
import socket
import threading
from socketserver import TCPServer, ThreadingMixIn
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from keep_gpu.mcp import server as server_module
from keep_gpu.mcp.server import (
    KeepGPUServer,
    SessionStartupUnavailable,
    _JSONRPCHandler,
)
from keep_gpu.utilities import platform_manager as pm
from keep_gpu.utilities.platform_manager import DeviceEnumerationUnavailableError


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


def _gpu_record(gpu_id: int, *, name=None):
    return {
        "id": gpu_id,
        "visible_id": gpu_id,
        "platform": "CUDA",
        "name": name or f"GPU {gpu_id}",
        "memory_total": None,
        "memory_used": None,
        "utilization": None,
    }


class DummyKeepGPUServer(KeepGPUServer):
    def list_gpus(self):
        return {"gpus": [_gpu_record(0)]}


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


def _request_http_response(method, url, data=None):
    request = Request(url=url, data=data, method=method)
    request.add_header("content-type", "application/json")
    try:
        with urlopen(request, timeout=2.0) as response:  # nosec B310
            body = response.read()
            return response.status, response.headers, body
    except HTTPError as exc:
        body = exc.read()
        return exc.code, exc.headers, body


def _allow_methods(headers):
    return {
        method.strip()
        for method in headers.get("allow", "").split(",")
        if method.strip()
    }


def _send_raw_http_json_request(httpd, request: bytes):
    host, port = httpd.server_address
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.settimeout(2.0)
        sock.sendall(request)

        chunks = []
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout as exc:
                raise AssertionError(
                    "HTTP response timed out before the client closed the socket"
                ) from exc
            if not chunk:
                break
            chunks.append(chunk)

    response = b"".join(chunks)
    assert response
    header_bytes, body = response.split(b"\r\n\r\n", 1)
    status_line = header_bytes.splitlines()[0].decode("iso-8859-1")
    status_code = int(status_line.split()[1])
    return status_code, json.loads(body.decode("utf-8")) if body else {}


def _raw_post_with_content_length(httpd, path: str, content_length: str):
    host, port = httpd.server_address
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {content_length}\r\n"
        "Connection: close\r\n"
        "\r\n"
        "{}"
    ).encode("ascii")
    return _send_raw_http_json_request(httpd, request)


def _raw_post_with_content_length_headers(
    httpd, path: str, content_lengths: list[str], body: bytes = b"{}"
):
    host, port = httpd.server_address
    headers = "".join(
        f"Content-Length: {content_length}\r\n" for content_length in content_lengths
    )
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"{headers}"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii") + body
    return _send_raw_http_json_request(httpd, request)


def _raw_http_json_request(httpd, method: str, target: str, payload=None):
    host, port = httpd.server_address
    if payload is None:
        body = b""
    elif isinstance(payload, bytes):
        body = payload
    else:
        body = json.dumps(payload).encode("utf-8")
    request = (
        f"{method} {target} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii") + body
    return _send_raw_http_json_request(httpd, request)


@pytest.mark.parametrize(
    ("method", "path", "allowed_methods"),
    [
        ("PUT", "/api/sessions", {"GET", "POST", "DELETE"}),
        ("PATCH", "/api/sessions/demo", {"GET", "DELETE"}),
        ("OPTIONS", "/api/sessions", {"GET", "POST", "DELETE"}),
    ],
)
def test_http_api_known_routes_reject_unsupported_methods_with_json_405(
    method, path, allowed_methods
):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response(method, f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == allowed_methods
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Method not allowed"}
    }


def test_http_rpc_rejects_unsupported_options_with_json_405():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("OPTIONS", f"{base}/rpc")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == {"POST"}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Method not allowed"}
    }


@pytest.mark.parametrize(
    ("method", "path", "allowed_methods", "data"),
    [
        ("POST", "/health", {"GET"}, b"{bad json"),
        ("POST", "/api/gpus", {"GET"}, b"{bad json"),
        ("POST", "/api/sessions/demo", {"GET", "DELETE"}, b"{bad json"),
        ("DELETE", "/", {"GET", "POST"}, None),
        ("DELETE", "/health", {"GET"}, None),
        ("DELETE", "/api/gpus", {"GET"}, None),
        ("DELETE", "/rpc", {"POST"}, None),
    ],
)
def test_http_implemented_handlers_reject_known_wrong_methods_with_json_405(
    method, path, allowed_methods, data
):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response(method, f"{base}{path}", data)
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == allowed_methods
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Method not allowed"}
    }


def test_http_rpc_get_rejects_with_json_405_instead_of_static_fallback():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("GET", f"{base}/rpc")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == {"POST"}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Method not allowed"}
    }


def test_http_rpc_head_rejects_with_json_405_and_empty_body():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}/rpc")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == {"POST"}
    assert body == b""


@pytest.mark.parametrize("rpc_path", ["/rpc/", "/rp%63"])
def test_http_rpc_noncanonical_head_returns_json_404_without_body(rpc_path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}{rpc_path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert body == b""


@pytest.mark.parametrize(
    "rpc_path",
    [
        "/rpc/",
        "/rpc%2F",
        "/rpc%3Bdebug",
        "/rpc%3Fdebug=1",
        "/rp%63",
        "/%72pc",
        "/%2Frpc",
    ],
)
def test_http_rpc_noncanonical_get_returns_json_404_without_static_fallback(
    rpc_path,
):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("GET", f"{base}{rpc_path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


@pytest.mark.parametrize("rpc_path", ["/rpc/", "/rpc;debug", "/rpc?debug=1"])
def test_http_rpc_noncanonical_path_rejects_before_jsonrpc_parse(rpc_path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_raw("POST", f"{base}{rpc_path}", b"{bad json")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload["error"]["message"] == "Unknown endpoint"


@pytest.mark.parametrize("rpc_path", ["/rp%63", "/%72pc", "/%2Frpc"])
def test_http_rpc_encoded_exact_alias_rejects_before_jsonrpc_parse(rpc_path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_raw("POST", f"{base}{rpc_path}", b"{bad json")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload["error"]["message"] == "Unknown endpoint"


def test_http_rpc_raw_double_slash_rejects_before_jsonrpc_parse():
    server = make_server()
    httpd, thread, _ = _start_http_server(server)

    try:
        status, payload = _raw_http_json_request(
            httpd,
            "POST",
            "//rpc",
            b"{bad json",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}


@pytest.mark.parametrize("rpc_path", ["/rp%63", "/%72pc", "/%2Frpc"])
def test_http_rpc_encoded_exact_alias_unsupported_method_returns_json_404(rpc_path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("OPTIONS", f"{base}{rpc_path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/api/unknown"),
        ("OPTIONS", "/api/unknown"),
        ("OPTIONS", "/api"),
    ],
)
def test_http_unknown_api_routes_reject_unsupported_methods_with_json_404(method, path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response(method, f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


@pytest.mark.parametrize(
    "path",
    [
        "/api%2Fsessions",
        "/api%2Funknown",
        "/api%2Fsessions%2Fjob",
        "/%2Fapi/gpus",
        "/api%3Bdebug",
        "/api%3Fsessions",
        "/api%23sessions",
    ],
)
def test_http_encoded_api_routes_return_json_404_without_static_fallback(path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("GET", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


def test_http_raw_double_slash_api_sessions_post_returns_404_without_start():
    server = make_server()
    httpd, thread, _ = _start_http_server(server)

    try:
        status, payload = _raw_http_json_request(
            httpd,
            "POST",
            "//api/sessions",
            {
                "job_id": "double-slash-start",
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}
    assert server.status()["active_jobs"] == []


def test_http_raw_double_slash_api_session_delete_returns_404_without_stop():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        _, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "double-slash-stop",
                "gpu_ids": [0],
                "vram": "64MB",
                "interval": 20,
                "busy_threshold": 5,
            },
        )
        job_id = start_payload["job_id"]

        status, payload = _raw_http_json_request(
            httpd, "DELETE", f"//api/sessions/{job_id}"
        )
        _, status_payload = _request_json("GET", f"{base}/api/sessions")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}
    assert [job["job_id"] for job in status_payload["active_jobs"]] == [job_id]


@pytest.mark.parametrize("path", ["/api%2Fsessions", "/%2Fapi/gpus"])
def test_http_encoded_api_route_unsupported_method_returns_json_404(path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("OPTIONS", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


@pytest.mark.parametrize(
    "path",
    ["/api%3Bdebug", "/api%3Fsessions", "/api%23sessions"],
)
def test_http_encoded_api_separator_unsupported_method_returns_json_404(path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("OPTIONS", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api%2Fsessions"),
        ("POST", "/api%3Bdebug"),
        ("POST", "/api%3Fsessions"),
        ("POST", "/api%23sessions"),
        ("DELETE", "/api%2Fsessions"),
        ("DELETE", "/api%3Bdebug"),
        ("DELETE", "/api%3Fsessions"),
        ("DELETE", "/api%23sessions"),
    ],
)
def test_http_encoded_api_route_implemented_methods_return_json_404(method, path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response(method, f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


def test_http_multisegment_session_route_rejects_unsupported_method_with_json_404():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response(
            "OPTIONS", f"{base}/api/sessions/foo/bar"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert b"<html" not in body.lower()
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Unknown endpoint"}
    }


def test_http_get_rejects_extra_session_path_segment_with_json_404():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_json("GET", f"{base}/api/sessions/foo/bar")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}


def test_http_head_api_sessions_rejects_with_json_405_headers_and_empty_body():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}/api/sessions")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 405
    assert headers.get("content-type") == "application/json"
    assert _allow_methods(headers) == {"GET", "POST", "DELETE"}
    assert body == b""


def test_http_api_encoded_noncanonical_head_returns_json_404_without_body():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}/api%2Fsessions")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get("content-type") == "application/json"
    assert "allow" not in {name.lower() for name in headers.keys()}
    assert body == b""


def test_http_unsupported_method_helper_ignores_uninitialized_handler():
    handler = _JSONRPCHandler.__new__(_JSONRPCHandler)

    assert handler._send_api_rpc_unsupported_method_response() is False


@pytest.mark.parametrize("rpc_path", ["/", "/rpc"])
def test_http_jsonrpc_parse_error_returns_jsonrpc_envelope(rpc_path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, payload = _request_raw("POST", f"{base}{rpc_path}", b"{")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 200
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] is None
    assert payload["error"]["code"] == -32700


def test_http_post_sessions_rejects_negative_content_length_without_client_close():
    server = make_server()
    httpd, thread, _ = _start_threaded_http_server(server)

    try:
        status, payload = _raw_post_with_content_length(httpd, "/api/sessions", "-1")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 400
    assert (
        "Content-Length must be a non-negative integer" in payload["error"]["message"]
    )
    assert server.status()["active_jobs"] == []


@pytest.mark.parametrize("rpc_path", ["/", "/rpc"])
def test_http_jsonrpc_rejects_negative_content_length_with_parse_error(rpc_path):
    server = make_server()
    httpd, thread, _ = _start_threaded_http_server(server)

    try:
        status, payload = _raw_post_with_content_length(httpd, rpc_path, "-1")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 200
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] is None
    assert payload["error"]["code"] == -32700
    assert (
        "Content-Length must be a non-negative integer" in payload["error"]["message"]
    )


@pytest.mark.parametrize(
    ("path", "expected_status"),
    [
        ("/api/sessions", 400),
        ("/rpc", 200),
    ],
)
def test_http_post_rejects_non_integer_content_length(path, expected_status):
    server = make_server()
    httpd, thread, _ = _start_threaded_http_server(server)

    try:
        status, payload = _raw_post_with_content_length(httpd, path, "abc")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == expected_status
    assert "Content-Length must be a non-negative integer" in (
        payload["error"]["message"]
    )
    if path in ("/", "/rpc"):
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] is None
        assert payload["error"]["code"] == -32700


@pytest.mark.parametrize(
    ("path", "expected_status"),
    [
        ("/api/sessions", 400),
        ("/", 200),
        ("/rpc", 200),
    ],
)
def test_http_post_rejects_duplicate_content_length(path, expected_status):
    server = make_server()
    httpd, thread, _ = _start_threaded_http_server(server)

    try:
        status, payload = _raw_post_with_content_length_headers(
            httpd, path, ["2", "2"], body=b"{}"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == expected_status
    assert "Content-Length must appear exactly once" in (payload["error"]["message"])
    assert server.status()["active_jobs"] == []
    if path in ("/", "/rpc"):
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] is None
        assert payload["error"]["code"] == -32700


@pytest.mark.parametrize(
    ("path", "content_length", "expected_status"),
    [
        ("/api/sessions", "+2", 400),
        ("/api/sessions", "0_2", 400),
        ("/", "+2", 200),
        ("/", "0_2", 200),
        ("/rpc", "+2", 200),
        ("/rpc", "0_2", 200),
    ],
)
def test_http_post_rejects_loose_content_length_syntax(
    path, content_length, expected_status
):
    server = make_server()
    httpd, thread, _ = _start_threaded_http_server(server)

    try:
        status, payload = _raw_post_with_content_length_headers(
            httpd, path, [content_length], body=b"{}"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == expected_status
    assert "Content-Length must be a non-negative integer" in (
        payload["error"]["message"]
    )
    assert server.status()["active_jobs"] == []
    if path in ("/", "/rpc"):
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] is None
        assert payload["error"]["code"] == -32700


def test_http_json_body_rejects_unicode_decimal_content_length_before_read():
    class FakeHeaders:
        def get_all(self, name):
            assert name == "content-length"
            return ["٢"]

    class ReadForbidden:
        def read(self, _length):
            raise AssertionError("body must not be read")

    handler = cast(Any, object.__new__(_JSONRPCHandler))
    handler.headers = FakeHeaders()
    handler.rfile = ReadForbidden()

    with pytest.raises(ValueError, match="Content-Length must be a non-negative"):
        _JSONRPCHandler._read_json_body(handler)


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


@pytest.mark.parametrize("path", ["/assets/missing.js", "/missing.keepgpu-test.js"])
def test_http_missing_static_asset_returns_404_not_dashboard_index(path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("GET", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get_content_type() == "application/json"
    assert json.loads(body.decode("utf-8")) == {
        "error": {"message": "Static asset not found"}
    }


@pytest.mark.parametrize("path", ["/assets/missing.js", "/missing.keepgpu-test.js"])
def test_http_head_missing_static_asset_returns_json_404_without_body(path):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get_content_type() == "application/json"
    assert body == b""


@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/", "text/html"), ("/assets/index.css", "text/css")],
)
def test_http_head_static_success_returns_headers_without_body(path, content_type):
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 200
    assert headers.get_content_type() == content_type
    assert int(headers["content-length"]) > 0
    assert body == b""


def test_http_head_static_runtime_error_returns_json_500_without_body(monkeypatch):
    server = make_server()

    def fail_guess_type(_path):
        raise RuntimeError("static exploded")

    monkeypatch.setattr(server_module.mimetypes, "guess_type", fail_guess_type)
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("HEAD", f"{base}/")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 500
    assert headers.get_content_type() == "application/json"
    assert body == b""


def test_http_missing_dashboard_shell_reports_ui_not_built(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "STATIC_DIR", tmp_path)

    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status, headers, body = _request_http_response("GET", f"{base}/")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status == 404
    assert headers.get_content_type() == "application/json"
    assert json.loads(body.decode("utf-8")) == {"error": {"message": "UI not built"}}


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


def test_http_session_start_preserves_fractional_interval():
    server = make_server()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, start_payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {
                "job_id": "http-fractional",
                "gpu_ids": [0],
                "vram": "256MB",
                "interval": 0.5,
            },
        )

        assert status_code == 200
        assert start_payload == {"job_id": "http-fractional"}
        _, status_payload = _request_json("GET", f"{base}/api/sessions/http-fractional")
        assert status_payload["params"]["interval"] == 0.5
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
        lambda: {"gpus": [_gpu_record(0), _gpu_record(2)]},
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


@pytest.mark.parametrize(
    ("list_gpus_result", "message_fragment"),
    [
        ({"gpus": [{"id": 0, "name": "GPU 0"}]}, "missing 'visible_id'"),
        ({}, "expected an object with a 'gpus' record list"),
    ],
)
def test_http_start_rejects_malformed_gpu_listing_before_startup(
    monkeypatch, list_gpus_result, message_fragment
):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )
    monkeypatch.setattr(server, "list_gpus", lambda: list_gpus_result)
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

    assert status == 500
    assert "Malformed list_gpus response" in payload["error"]["message"]
    assert message_fragment in payload["error"]["message"]
    assert payload["error"]["type"] == "RuntimeError"
    assert controllers == []


@pytest.mark.parametrize(
    ("list_gpus_result", "message"),
    [
        ({"gpus": []}, "No usable visible GPUs are available"),
        (
            DeviceEnumerationUnavailableError("Unable to enumerate visible GPUs"),
            "Unable to enumerate visible GPUs",
        ),
    ],
)
def test_http_start_reports_startup_unavailable_when_gpu_listing_unusable(
    monkeypatch, list_gpus_result, message
):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )

    def list_gpus():
        if isinstance(list_gpus_result, Exception):
            raise list_gpus_result
        return list_gpus_result

    monkeypatch.setattr(server, "list_gpus", list_gpus)
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

    assert status == 503
    assert payload["error"]["message"] == message
    assert payload["error"]["type"] == "SessionStartupUnavailable"
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
        assert status_code == 404
        assert error_payload == {"error": {"message": "Unknown endpoint"}}

        _, status_payload = _request_json("GET", f"{base}/api/sessions")
        assert status_payload["active_jobs"]
        assert status_payload["active_jobs"][0]["job_id"] == job_id
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize("path", ["/api/unknown", "/api"])
def test_http_unknown_api_route_returns_json_404(path):
    server = make_server()

    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server(("127.0.0.1", 0), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        status_code, payload = _request_json("GET", f"{base}{path}")
        assert status_code == 404
        assert payload["error"]["message"] == "Unknown endpoint"
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "path", ["/api/gpus?bad=query", "/api/gpus;bad", "/%2Fapi/gpus"]
)
def test_http_get_api_gpus_noncanonical_route_returns_json_404_without_listing(path):
    server = make_server()
    list_calls = []

    def fail_list_gpus():
        list_calls.append(True)
        raise AssertionError("list_gpus should not run for noncanonical route")

    server.list_gpus = fail_list_gpus  # type: ignore[method-assign]
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}
    assert list_calls == []


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


def test_http_get_api_gpus_enumeration_unavailable_returns_json_503(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "list_gpus",
        lambda: (_ for _ in ()).throw(
            DeviceEnumerationUnavailableError("Unable to enumerate visible GPUs")
        ),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}/api/gpus")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 503
    assert payload["error"]["message"] == "Unable to enumerate visible GPUs"
    assert payload["error"]["type"] == "DeviceEnumerationUnavailableError"


@pytest.mark.parametrize(
    ("records", "message_fragment"),
    [
        (
            [
                {
                    "id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                }
            ],
            "visible_id",
        ),
        (
            [
                {
                    "id": -1,
                    "visible_id": -1,
                    "platform": "CUDA",
                    "name": "GPU hidden",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                }
            ],
            "non-negative",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                },
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU alias",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                },
            ],
            "duplicate",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": -1,
                }
            ],
            "between 0 and 100",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": -1,
                    "memory_used": 0,
                    "utilization": None,
                }
            ],
            "'memory_total' must be a non-negative integer or null",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": -1,
                    "utilization": None,
                }
            ],
            "'memory_used' must be a non-negative integer or null",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": 1024,
                    "memory_used": 2048,
                    "utilization": None,
                }
            ],
            "'memory_used' must not exceed 'memory_total'",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": 101,
                }
            ],
            "between 0 and 100",
        ),
    ],
)
def test_http_get_api_gpus_malformed_record_returns_json_500(
    monkeypatch, records, message_fragment
):
    monkeypatch.setattr(
        server_module,
        "get_gpu_info",
        lambda: records,
    )
    server = KeepGPUServer(controller_factory=cast(Any, dummy_factory))
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}/api/gpus")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 500
    assert "Malformed list_gpus response" in payload["error"]["message"]
    assert message_fragment in payload["error"]["message"]
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


def test_http_post_sessions_startup_unavailable_returns_json_503(monkeypatch):
    server = make_server()
    monkeypatch.setattr(
        server,
        "start_keep",
        lambda **_: (_ for _ in ()).throw(
            SessionStartupUnavailable("No usable GPUs are available")
        ),
    )
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("POST", f"{base}/api/sessions", {})
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 503
    assert payload["error"]["message"] == "No usable GPUs are available"
    assert payload["error"]["type"] == "SessionStartupUnavailable"


def test_http_post_sessions_unavailable_mps_returns_json_503(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    server = KeepGPUServer()
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json(
            "POST",
            f"{base}/api/sessions",
            {"job_id": "mps-unavailable"},
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 503
    assert payload["error"]["message"] == "PyTorch MPS backend is not available"
    assert payload["error"]["type"] == "SessionStartupUnavailable"


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

        assert status_code == 404
        assert payload == {"error": {"message": "Unknown endpoint"}}
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
