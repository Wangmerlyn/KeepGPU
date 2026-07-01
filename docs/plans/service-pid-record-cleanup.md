# Service PID Record Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent auto-started KeepGPU service daemons from being stranded when PID ownership recording fails or cannot capture trustworthy process identity.

**Architecture:** Keep daemon ownership safety centralized in `src/keep_gpu/cli.py`. PID records should be written only when `pid`, `port`, `uid`, and `start_time` are exact safe JSON values; `_start_service_process()` owns cleanup of the process it just spawned if record creation/write fails.

**Tech Stack:** Python, Typer CLI helpers, pytest, MkDocs.

---

## Task 1: Clean Up Just-Spawned Daemons on PID-Record Failure

**Files:**
- Modify: `src/keep_gpu/cli.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `AGENTS.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/reference/cli.md`
- Create: `docs/plans/service-pid-record-cleanup.md`

- [x] **Step 1: Write failing cleanup regression for PID-record write failure**

Add a fake process test near the existing service PID ownership tests:

```python
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
```

- [x] **Step 2: Write failing regression for unknown recorded identity**

Add a test proving auto-start does not leave a daemon running when UID or start identity is unknown:

```python
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

    with pytest.raises(RuntimeError, match="Cannot verify KeepGPU service daemon ownership"):
        cli._start_service_process("127.0.0.1", 8765)

    assert process.terminated == 1
    assert process.wait_timeouts == [1.0]
    assert not cli._service_pid_path("127.0.0.1", 8765).exists()
```

- [x] **Step 3: Verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'start_service_process and (pid_write_fails or identity_is_unknown)'
```

Expected: the write-failure test fails because the fake process is not terminated, and the unknown-identity test fails because `_write_service_pid()` currently records unverifiable identity and `_start_service_process()` returns success.

- [x] **Step 4: Implement minimal cleanup and identity validation**

Add a small cleanup helper in `src/keep_gpu/cli.py`:

```python
def _terminate_spawned_service_process(process: subprocess.Popen) -> None:
    try:
        process.terminate()
    except Exception:
        logger.debug("Failed to send SIGTERM to spawned service process", exc_info=True)
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            logger.debug("Failed to kill spawned service process", exc_info=True)
    except Exception:
        logger.debug("Failed to wait for spawned service process", exc_info=True)
```

Validate identity before building the PID record:

```python
uid = _process_uid(pid)
start_time = _process_start_identity(pid)
if (
    not isinstance(uid, int)
    or isinstance(uid, bool)
    or not isinstance(start_time, str)
    or not start_time
):
    raise RuntimeError(
        f"Cannot verify KeepGPU service daemon ownership for pid={pid}"
    )
```

Wrap `_write_service_pid()` in `_start_service_process()`:

```python
pid_snapshot = _snapshot_service_pid_file(host, port)
process = subprocess.Popen(_service_command(host, port), **popen_kwargs)
try:
    _write_service_pid(host, port, process.pid)
except Exception:
    _terminate_spawned_service_process(process)
    _restore_service_pid_file(host, port, pid_snapshot)
    raise
return process.pid
```

If `_snapshot_service_pid_file()` sees a permission or I/O error while reading
an existing PID file, propagate that error before spawning the service. Unknown
prior PID-file state must not be treated as absence, because cleanup would risk
deleting another daemon's ownership record.

- [x] **Step 5: Adjust existing ownership tests and docs**

Update direct `_write_service_pid()` tests to monkeypatch known identity before writing records. For tests that intentionally need malformed records, write JSON payloads directly instead of using `_write_service_pid()`.

Document that auto-start aborts and cleans up if it cannot create a trustworthy ownership record.

- [x] **Step 6: Verify GREEN and PR**

Run:

```bash
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'service_process or service_stop or pid_record or ownership'
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q
PYTHONPATH=src pytest tests -q
PYTHONPATH=src mkdocs build --strict
pre-commit run --all-files --show-diff-on-failure
git diff --check
```

Then push the branch, open a PR, run local subagent review, address every comment, wait for hosted checks, and squash-merge only after all review threads are resolved.

Verification:
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'start_service_process and (pid_write_fails or identity_is_unknown)'`
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'preserves_existing_pid_record_when_pid_write_fails'`
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'aborts_before_spawn_when_pid_snapshot_fails'`
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'waits_when_terminate_signal_fails'`
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'service_process or service_stop or pid_record or ownership'`
- `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q`
- `PYTHONPATH=src pytest tests -q`
- `PYTHONPATH=src mkdocs build --strict`
- `pre-commit run --all-files --show-diff-on-failure`
- `git diff --check`
