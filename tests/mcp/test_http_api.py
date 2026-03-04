import json
import threading
from socketserver import TCPServer
from urllib.request import Request, urlopen

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


def _request_json(method, url, payload=None):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, method=method)
    request.add_header("content-type", "application/json")
    with urlopen(request, timeout=2.0) as response:  # nosec B310
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def test_http_health_and_static_index():
    server = KeepGPUServer(controller_factory=dummy_factory)

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
    server = KeepGPUServer(controller_factory=dummy_factory)

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
