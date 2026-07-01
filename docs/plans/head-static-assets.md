# HEAD Static Assets Implementation Plan

**Goal:** Make HEAD requests for missing dashboard asset paths return the same structured JSON 404 headers as GET, with no response body.

**Background:** `_JSONRPCHandler.do_GET()` routes dashboard/static paths through `_serve_static()`, which already identifies missing `/assets/*` and extension-bearing paths as structured JSON 404 errors. Because `_JSONRPCHandler` does not implement `do_HEAD()`, those requests currently fall through to `BaseHTTPRequestHandler` HTML 501 responses.

**Approach:** Add focused regression tests for missing assets, static success, noncanonical API/RPC routes, and runtime-error envelopes under `HEAD`, verify the original missing-asset case fails, then route HEAD static requests through existing static semantics using `write_body=False`. Keep GET behavior unchanged and update `AGENTS.md` only to clarify that dashboard/static HEAD responses are header-only while API/RPC HEAD keeps structured 405/404 handling.

**Todo:**
- [x] Add the minimal failing parametrized HEAD missing-asset test near the existing GET missing-asset test.
- [x] Run the targeted new test and record the RED failure.
- [x] Implement the smallest `_JSONRPCHandler` change needed for HEAD missing assets.
- [x] Update `AGENTS.md` narrowly for the HEAD/GET missing dashboard asset contract.
- [x] Address local review by preserving `HEAD /` static success, suppressing
  HEAD runtime-error bodies, and covering HEAD noncanonical API/RPC routes.
- [x] Address hosted review by avoiding full static file reads for HEAD
  content length and removing unreachable `/rpc` handling from `do_HEAD()`.
- [x] Run the requested targeted, full MCP HTTP API, docs, pre-commit, and diff checks.
- [x] Commit with `fix(mcp): return structured head asset errors`.
