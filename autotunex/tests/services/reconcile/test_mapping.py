"""to_run_status maps every gbserver status, and warns on the unknown."""

from __future__ import annotations

import pytest

from autotunex.models.status import RunStatus
from autotunex.services.reconcile.mapping import to_run_status


@pytest.mark.parametrize(
    ("cluster_status", "expected"),
    [
        ("submitted", None),
        ("pending", None),
        ("retry_pending", None),
        ("running", RunStatus.RUNNING),
        ("success", RunStatus.COMPLETED),
        ("failed", RunStatus.ERROR),
        ("invalid", RunStatus.ERROR),
        ("cancelled", RunStatus.TERMINATED),
        ("cancel_requested", None),
    ],
)
def test_maps_each_known_cluster_status(cluster_status: str, expected: RunStatus | None) -> None:
    assert to_run_status(cluster_status) == expected


def test_unknown_status_maps_to_none() -> None:
    assert to_run_status("teleporting") is None
