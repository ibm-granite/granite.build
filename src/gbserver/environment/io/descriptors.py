"""Env-agnostic, store-declared IO descriptors for the inline (same-instance)
prologue path. See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4.

Only the inline INPUT (download-before-step) path survives: SkyPilot/AWS
artifact push is a dispatched step over a shared filesystem whose destination
resolves at push time (#390), so the output/upload descriptors were removed.
"""

from pydantic import BaseModel


class InputIO(BaseModel):
    """Base for a declared inline input (download-before-step)."""


class HfInputIO(InputIO):
    repo: str
    revision: str
    type: str = "model"
    dest: str
    token: str = ""
