# Service PID Ownership Implementation Plan

## Background

Audit agents found that `keep-gpu stop --all` can convert any `RuntimeError`
from the service stop RPC into a daemon kill attempt when a PID file exists.
The fallback also reports success even if `_stop_service_process()` refuses to
stop the process. Separately, the current PID file stores only a bare integer,
so a later PID reuse or command-line substring spoof can look like a managed
KeepGPU daemon.

Server-side duplicate `job_id` startup races and stop-all release latency are
tracked as separate issues and are intentionally out of scope for this branch.

## Goal

Make service shutdown humane and ownership-safe: fallback force-stop is allowed
only for unreachable service failures, and no CLI path may signal a daemon
unless the auto-start ownership record verifies the process.

## Solution

- Store a structured daemon ownership record instead of only a bare PID.
- Verify PID, host, port, command identity, user id, and process start identity
  when available before signaling a process.
- Keep `--force` as "skip active-session RPC checks", not "skip ownership".
- Restrict `stop --all` fallback to service-unreachable/timeout failures.
- Report an error instead of a success payload when fallback cannot verify or
  stop the managed daemon.

## Todo

- [x] Add failing CLI tests for non-unreachable RPC errors, failed fallback
      stops, unmanaged PID fallback, substring-spoof command lines, and
      structured ownership records.
- [x] Implement structured PID ownership helpers in `src/keep_gpu/cli.py`.
- [x] Narrow fallback logic and require `_stop_service_process()` success before
      reporting force-stop.
- [x] Update `AGENTS.md`, CLI guide, and CLI reference docs with the ownership
      rule.
- [x] Run targeted CLI tests, full tests, docs build, and pre-commit.
- [ ] Open a GitHub PR, run local subagent review, resolve all comments, then
      squash merge.
