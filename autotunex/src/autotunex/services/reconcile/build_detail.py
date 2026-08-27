# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Transform gbserver's status + events responses into the UX ``build_status`` shape.

The frontend Status tab renders ``build_status.build_history[].description`` and
the ``BuildStatus`` type declares ``{details, targets, build_history}``. gbserver
exposes that data across *two* endpoints — ``/status`` (build + target runs) and
``/events`` (the event log) — so the reconcile loop fetches both and this pure
function assembles them into the one dict the UX expects. It reproduces the 2025
API's ``gb_service._fetch_gb_status`` mapping field-for-field, including the
empty-``msg`` drop and backtick strip on history lines.

Every field is read defensively with ``.get`` so a partial or unexpected body
degrades to empty values rather than raising inside a reconcile sweep.
"""

from __future__ import annotations

from typing import Any


def _artifacts(raw: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map gbserver artifact records to the UX ``{artifact_id, uri}`` shape."""
    return [{"artifact_id": a.get("uuid", ""), "uri": a.get("uri", "")} for a in raw]


def _targets(target_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten gbserver target runs into the UX ``Target`` shape.

    Mirrors ``gbcli.services.service_build.process_target_runs_to_json`` without
    depending on gbcli: input/output artifacts and steps are projected to the
    small field sets the frontend renders.
    """
    targets: list[dict[str, Any]] = []
    for run in target_runs:
        target = run.get("target") or {}
        steps = [
            {
                "step_id": step.get("uuid", ""),
                "uri": step.get("definition_uri", ""),
                "status": step.get("status", ""),
                "started_at": step.get("started_at", ""),
            }
            for step in run.get("steps", [])
        ]
        targets.append(
            {
                "target_name": target.get("name"),
                "build_id": target.get("build_id", ""),
                "target_id": target.get("uuid"),
                "status": target.get("status"),
                "skipped_for_prerun_target_id": target.get("skipped_for_prerun_target_id", ""),
                "input_artifacts": _artifacts(run.get("input_artifacts", [])),
                "output_artifacts": _artifacts(run.get("output_artifacts", [])),
                "steps": steps,
            }
        )
    return targets


def _build_history(events_body: dict[str, Any]) -> list[dict[str, str]]:
    """Project gbserver build events to ``[{time, description}]``.

    Events with an empty ``msg`` are dropped and backticks are stripped from the
    description, exactly as the 2025 API did before rendering the Status tab.
    """
    history: list[dict[str, str]] = []
    for event in events_body.get("events", []):
        build_event = event.get("build_event") or {}
        msg = (build_event.get("payload") or {}).get("msg", "")
        if not msg:
            continue
        history.append(
            {"time": build_event.get("timestamp", ""), "description": msg.replace("`", "")}
        )
    return history


def build_status_detail(status_body: dict[str, Any], events_body: dict[str, Any]) -> dict[str, Any]:
    """Assemble the UX ``build_status`` dict from gbserver's status and events bodies.

    Args:
        status_body: the ``GET /builds/{id}/status`` response body.
        events_body: the ``GET /builds/{id}/events`` response body; ``{}`` when the
            events call failed, yielding an empty ``build_history``.

    Returns:
        ``{"details": {...}, "targets": [...], "build_history": [...]}``.
    """
    status = status_body.get("status") or {}
    build = status.get("build") or {}
    details = {
        "build_id": build.get("uuid", ""),
        "name": build.get("name", ""),
        "started_at": build.get("created_time", ""),
        "updated_at": build.get("updated_time", ""),
        "status": build.get("status", ""),
        "source_pr": build.get("source_uri", ""),
        "description": build.get("description", ""),
    }
    return {
        "details": details,
        "targets": _targets(status.get("target_runs", [])),
        "build_history": _build_history(events_body),
    }


def output_artifact_ref(detail: dict[str, Any]) -> dict[str, str] | None:
    """Return the build's primary output artifact ``{artifact_id, uri}``, or ``None``.

    Reads the first target's first output artifact from a :func:`build_status_detail`
    result — the produced model — and is used to populate ``gb_tasks.artifact_id`` /
    ``artifact_uri`` when a job ends. Matches the 2025 API's
    ``_extract_artifact_from_build``. Runs on any terminal state, not just success:
    a build that fails at push can still have registered the artifact. Returns
    ``None`` unless a non-empty ``artifact_id`` is present, so a status-only upsert
    never clobbers a stored id with a blank.
    """
    targets = detail.get("targets") or []
    if not targets:
        return None
    outputs = targets[0].get("output_artifacts") or []
    if not outputs:
        return None
    first = outputs[0] or {}
    artifact_id = first.get("artifact_id") or ""
    if not artifact_id:
        return None
    return {"artifact_id": artifact_id, "uri": first.get("uri") or ""}
