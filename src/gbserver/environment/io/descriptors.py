"""Env-agnostic, store-declared IO descriptors for the inline (same-instance)
prologue/epilogue path. See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4."""

from typing import Optional

from pydantic import BaseModel


class InputIO(BaseModel):
    """Base for a declared inline input (download-before-step)."""


class OutputIO(BaseModel):
    """Base for a declared inline output (upload-after-step)."""


class HfInputIO(InputIO):
    repo: str
    revision: str
    type: str = "model"
    dest: str
    token: str = ""


class HfOutputIO(OutputIO):
    # Destination only; the source path is resolved at runtime from the
    # step's GB_ARTIFACT_PATH marker (marker-capture), so there is no `src`.
    repo: str
    revision: str = "main"
    private: bool = True
    resource_group_id: Optional[str] = None
    path_in_repo: str = ""
    uri: str
    binding_id: str
    token: str = ""
    hf_type: str = "model"


class InlineDeferredPush:
    """Third `pushasset_<store>` outcome: the upload was folded into the
    producing step's epilogue, so the post-step push must be a no-op that
    still emits CREATED and registers the binding (see spec §6)."""
