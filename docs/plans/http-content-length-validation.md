# HTTP Content-Length Validation Plan

## Background

The HTTP JSON body reader uses `Content-Length` to decide how many bytes to read
before REST or JSON-RPC request handling. It already rejects negative and
non-integer values, but Python's `int()` accepts loose forms such as `+2` and
`0_2`, and `headers.get()` hides duplicate `Content-Length` fields.

## Goal

Reject ambiguous or loose `Content-Length` values before any body read or
session side effect. REST endpoints keep structured JSON `400` errors, while
JSON-RPC endpoints keep parse-error envelopes.

## Design

- Use `headers.get_all("content-length")` to detect duplicate headers.
- Accept only one present value, or preserve the existing missing-header
  behavior as length zero.
- Accept only parser-normalized plain ASCII decimal digits before converting to
  `int`; leading zeroes remain valid digits, while signs, underscores, and
  Unicode decimal digits do not.
- Keep the change local to the shared HTTP JSON body reader and MCP HTTP tests.

## Todo

- [x] Add failing REST and JSON-RPC tests for duplicate `Content-Length`.
- [x] Add failing REST and JSON-RPC tests for `int()`-accepted loose syntax.
- [x] Tighten the shared HTTP body reader before `rfile.read()`.
- [x] Update `AGENTS.md` with the stricter HTTP body-reader contract.
- [x] Run focused MCP HTTP tests, full tests, docs build, and pre-commit
      locally.
- [ ] Run local review, hosted PR checks, and merge only after all review
      comments are resolved.
