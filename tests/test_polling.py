from tests import polling


def test_wait_until_uses_monotonic_deadline(monkeypatch):
    calls = {"predicate": 0}

    def predicate():
        calls["predicate"] += 1
        return calls["predicate"] == 2

    monotonic_calls = [1.0, 1.1, 1.2]
    monkeypatch.setattr(
        polling.time,
        "monotonic",
        lambda: (
            monotonic_calls.pop(0) if len(monotonic_calls) > 1 else monotonic_calls[0]
        ),
    )
    monkeypatch.setattr(polling.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        polling.time,
        "time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock used")),
    )

    assert polling.wait_until(predicate, timeout_s=1.0, interval_s=0.01) is True
