import abc
import threading


class AtomicInteger(abc.ABC):
    _counter: int
    _lock: threading.Lock

    def __init__(self, initial_value: int = 0):
        self._counter = initial_value
        self._lock = threading.Lock()

    def fetch_and_add(self) -> int:
        """Get the current value and then increment the internal value"""
        with self._lock:  # Acquire the lock before modifying counter
            value = self._counter
            self._counter += 1
        return value

    def get(self) -> int:
        """Atomically read the current value."""
        with self._lock:
            return self._counter
