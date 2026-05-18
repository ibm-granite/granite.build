from __future__ import annotations


class MockJobStatus:
    def __init__(self, name: str, is_terminal: bool):
        self._name = name
        self._is_terminal = is_terminal

    def is_terminal(self) -> bool:
        return self._is_terminal

    def __str__(self) -> str:
        return f"JobStatus.{self._name}"

    def __repr__(self) -> str:
        return f"MockJobStatus({self._name!r}, is_terminal={self._is_terminal})"

    def __eq__(self, other) -> bool:
        if isinstance(other, MockJobStatus):
            return self._name == other._name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name)
