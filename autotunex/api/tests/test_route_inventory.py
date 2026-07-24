# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import os

from fastapi.routing import APIRoute

import server  # assembled app — only importable under conftest stubs (autotune is stubbed)

_BASELINE = os.path.join(os.path.dirname(__file__), "_route_baseline.json")


def _route_set(app):
    """Stable, comparable snapshot of every APIRoute on the app."""
    rows = []
    for r in app.routes:
        if isinstance(r, APIRoute):
            rows.append(
                [
                    sorted(r.methods or []),
                    r.path,
                    sorted(getattr(r, "tags", []) or []),
                    bool(getattr(r, "include_in_schema", True)),
                ]
            )
    return sorted(rows, key=lambda x: (x[1], x[0], x[2]))


def test_route_inventory_matches_baseline():
    current = _route_set(server.app)
    if not os.path.exists(_BASELINE):
        # First run only: record the baseline (BEFORE any extraction).
        with open(_BASELINE, "w") as f:
            json.dump(current, f, indent=2)
    with open(_BASELINE) as f:
        baseline = json.load(f)
    assert current == baseline, (
        "Route set drifted from the recorded baseline. A route's "
        "path / verb / tags / include_in_schema changed during extraction. "
        "If genuinely intended, delete tests/_route_baseline.json to re-record; "
        "otherwise fix the extraction to preserve the route verbatim."
    )
