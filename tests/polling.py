import time
from typing import Callable


def wait_until(
    predicate: Callable[[], bool],
    timeout_s: float = 3.0,
    interval_s: float = 0.05,
) -> bool:
    """Poll predicate until it returns True or timeout is reached."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False
