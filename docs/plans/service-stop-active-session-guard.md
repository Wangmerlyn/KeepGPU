# Service Stop Active Session Guard Plan

## Background

Non-force `keep-gpu service-stop` checks `status`, calls `stop_keep`, then
signals the ownership-verified daemon process. A session can appear after the
first status snapshot. If `stop_keep` reports it stopped that session, or if a
new session appears after `stop_keep` returns a clean empty result, the CLI must
not stop the daemon as if it had proven the service idle.

## Goal

Keep non-force daemon shutdown conservative: signal the daemon only after the
service is reachable, `stop_keep` reports no session work, and a final status
check still shows no active sessions.

## Design

- Keep `service-stop --force` unchanged: it skips session RPC checks but still
  requires ownership-verified daemon identity.
- Treat any non-empty `stopped`, `timed_out`, or `failed` `stop_keep` field as
  not clean for non-force daemon shutdown.
- Recheck all-session `status` after a clean `stop_keep` result and before
  `_stop_service_process()`.
- Keep the change local to CLI service-stop validation and tests.

## Todo

- [x] Add a failing regression for a late session stopped by `stop_keep`.
- [x] Add a failing regression for a late session visible only on the final
      status recheck.
- [x] Tighten `service-stop` guards before daemon signaling.
- [x] Update `AGENTS.md` and CLI docs with the stricter non-force contract.
- [x] Run targeted CLI tests, full tests, docs build, and pre-commit locally.
- [ ] Run local review, hosted PR checks, and merge only after all review
      comments are resolved.
