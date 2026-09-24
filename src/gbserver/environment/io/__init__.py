"""Inline (same-instance) asset IO: prologue/epilogue abstraction. See
docs/plans/2026-09-18-inline-hfpush-envio-design.md §4."""

from typing import Optional

from gbserver.environment.io.base import EnvironmentIO
from gbserver.environment.io.descriptors import (
    HfInputIO,
    HfOutputIO,
    InlineDeferredPush,
    InputIO,
    OutputIO,
)
from gbserver.environment.io.skypilot import SkypilotIO

__all__ = [
    "EnvironmentIO",
    "SkypilotIO",
    "HfInputIO",
    "HfOutputIO",
    "InlineDeferredPush",
    "InputIO",
    "OutputIO",
    "build_env_io",
]


def build_env_io(env) -> Optional[EnvironmentIO]:
    """Return the inline-IO renderer for an environment, or None.

    Mirrors ``gbserver.environment.shared_fs.build_provider``. The
    discriminator is the environment *class* identifier ``type``: an
    ``Environment`` instance sets ``self.type = self.__class__.__name__``
    (environment.py) and ``EnvironmentConfig.type`` carries the same value,
    so a genuine SkyPilot environment reports ``"Skypilot"`` (capitalized),
    matching the ``type`` written in environment.yaml. (The brief's sketch
    used a lowercase ``"skypilot"`` literal, which never matches reality.)
    """
    if getattr(env, "type", None) == "Skypilot":
        return SkypilotIO()
    return None
