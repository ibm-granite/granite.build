"""Thin prologue/epilogue abstraction for inline (same-instance) asset IO.
See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4."""

from abc import ABC
from collections.abc import Sequence

from gbserver.environment.io.descriptors import InputIO


class EnvironmentIO(ABC):
    # Sequence (covariant) rather than list (invariant) so callers may pass a
    # concrete list[HfInputIO] without a type error.
    def render_prologue(self, inputs: Sequence[InputIO]) -> str:
        """Shell prepended to the step's setup (e.g. hf download). Default: none."""
        return ""
