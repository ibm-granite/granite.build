"""Thin prologue/epilogue abstraction for inline (same-instance) asset IO.
See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4."""

from abc import ABC

from gbserver.environment.io.descriptors import InputIO, OutputIO


class EnvironmentIO(ABC):
    def render_prologue(self, inputs: list[InputIO]) -> str:
        """Shell prepended to the step's setup (e.g. hf download). Default: none."""
        return ""

    def render_epilogue(self, outputs: list[OutputIO], capture_var: str) -> str:
        """Shell appended to the step's run (e.g. hf upload). `capture_var` names
        the shell var holding the tee'd stdout capture file the epilogue greps
        for GB_ARTIFACT_PATH. Default: none."""
        return ""
