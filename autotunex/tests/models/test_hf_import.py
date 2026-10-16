# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Request-shape invariants for the HuggingFace import schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autotunex.models.hf_import import HfImportRequest, HfPreviewRequest


def _request(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": "ds",
        "repo_id": "tatsu-lab/alpaca",
        "revision": "a" * 40,
        "config": "default",
        "train_split": "train",
        "column_mapping": {"input": "instruction"},
    }
    body.update(overrides)
    return body


def test_an_import_rejects_a_validation_split_and_a_percentage_together() -> None:
    """Both set means one is silently dropped, so refuse the pair outright.

    ``dataset_runner._prepare_files`` tests its validation *file* first and
    returns, so the percentage would never take effect and nothing would say so.
    ``DatasetService.upload`` rejects the same combination.
    """
    with pytest.raises(ValidationError, match="not both"):
        HfImportRequest.model_validate(
            _request(validation_split="validation", validation_percentage=10)
        )


@pytest.mark.parametrize("field", ["validation_split", "validation_percentage"])
def test_an_import_accepts_either_validation_source_on_its_own(field: str) -> None:
    """Only the *combination* is refused; each alone is a legitimate request."""
    value: object = "validation" if field == "validation_split" else 10

    request = HfImportRequest.model_validate(_request(**{field: value}))

    assert getattr(request, field) == value


@pytest.mark.parametrize(
    "repo_id",
    [
        "../../etc/passwd",
        "o/../../../some/path",
        "../x",
        "o/..",
        "owner/name/extra",
        "owner//name",
        ".hidden/x",
        "o/r?x=1",
        "o r/x",
    ],
)
def test_a_repo_id_that_could_escape_the_hub_api_path_is_rejected(repo_id: str) -> None:
    """``repo_id`` lands in the Hub URL path, so it must not be able to leave it.

    ``hf_hub`` interpolates it into ``/api/datasets/{repo_id}`` and httpx
    normalizes ``..`` per RFC 3986 — harmless against ``huggingface.co``, but
    ``hf_hub_base_url`` is documented as overridable to a mirror, which makes an
    unconstrained value a request-forgery primitive against an internal host.
    """
    with pytest.raises(ValidationError):
        HfImportRequest.model_validate(_request(repo_id=repo_id))


@pytest.mark.parametrize(
    "repo_id",
    ["tatsu-lab/alpaca", "databricks/databricks-dolly-15k", "HuggingFaceFW/fineweb", "squad"],
)
def test_a_real_hub_repo_id_still_validates(repo_id: str) -> None:
    """The guard must not refuse ids people actually import, bare canonical ones included."""
    assert HfImportRequest.model_validate(_request(repo_id=repo_id)).repo_id == repo_id


def test_the_preview_request_guards_repo_id_the_same_way() -> None:
    """Preview reaches the same Hub URL builders, so it needs the same guard."""
    with pytest.raises(ValidationError):
        HfPreviewRequest.model_validate(
            {
                "repo_id": "o/../../../some/path",
                "config": "default",
                "train_split": "train",
                "column_mapping": {"input": "instruction"},
            }
        )


@pytest.mark.parametrize("model", [HfImportRequest, HfPreviewRequest])
@pytest.mark.parametrize("revision", [None, "main", "refs/convert/parquet", "abc123", "../x"])
def test_preview_and_import_require_an_immutable_revision(
    model: type[HfImportRequest] | type[HfPreviewRequest], revision: str | None
) -> None:
    body = _request(revision=revision)
    if model is HfPreviewRequest:
        body.pop("name")
    if revision is None:
        body.pop("revision")

    with pytest.raises(ValidationError, match="revision"):
        model.model_validate(body)
