# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Scheme dispatch for artifact listing."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from autotunex.core.exceptions import JobArtifactsNotFoundError
from autotunex.models.asset import AssetSummary
from autotunex.services.storage.artifacts import OpenedArtifact
from autotunex.services.storage.registry import resolve_artifact_lister


class _Lister:
    async def list_files(self, *, location: str) -> list[AssetSummary]:
        return []

    async def open_file(self, *, location: str, path: str) -> OpenedArtifact:
        async def _empty() -> AsyncIterator[bytes]:
            yield b""

        return OpenedArtifact(filename=path, media_type="", size=0, stream=_empty())


def test_hf_scheme_resolves_to_repo_id() -> None:
    fs, hf = _Lister(), _Lister()

    lister, location = resolve_artifact_lister(
        "hf://huggingface.co/models/ibm-research/autotunex_x", filesystem=fs, huggingface=hf
    )

    assert lister is hf
    assert location == "ibm-research/autotunex_x"


def test_file_scheme_resolves_to_path() -> None:
    fs, hf = _Lister(), _Lister()

    lister, location = resolve_artifact_lister(
        "file:///data/outputs/autotune_abc/", filesystem=fs, huggingface=hf
    )

    assert lister is fs
    assert location == "/data/outputs/autotune_abc/"


def test_unknown_scheme_raises_not_found() -> None:
    with pytest.raises(JobArtifactsNotFoundError):
        resolve_artifact_lister("s3://bucket/key", filesystem=_Lister(), huggingface=_Lister())
