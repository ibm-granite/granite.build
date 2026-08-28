# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""build_status_detail: gbserver status + events -> {details, targets, build_history}.

Mirrors the transform the 2025 API performed in ``gb_service._fetch_gb_status``:
the frontend Status tab renders ``build_status.build_history[].description`` and
the ``BuildStatus`` type declares ``{details, targets, build_history}``, so the
new backend must produce that exact shape.
"""

from __future__ import annotations

from typing import Any

from autotunex.services.reconcile.build_detail import build_status_detail, output_artifact_ref


def _status_body() -> dict[str, Any]:
    return {
        "status": {
            "build": {
                "uuid": "b-123",
                "name": "my-build",
                "status": "success",
                "created_time": "2026-08-07T00:00:00Z",
                "updated_time": "2026-08-07T00:05:00Z",
                "source_uri": "https://github.example/pr/1",
                "description": "a build",
            },
            "target_runs": [
                {
                    "target": {
                        "name": "train",
                        "build_id": "b-123",
                        "uuid": "t-1",
                        "status": "success",
                    },
                    "input_artifacts": [{"uuid": "a-in", "uri": "s3://in"}],
                    "output_artifacts": [{"uuid": "a-out", "uri": "s3://out"}],
                    "steps": [
                        {
                            "uuid": "s-1",
                            "definition_uri": "step://1",
                            "status": "success",
                            "started_at": "2026-08-07T00:01:00Z",
                        }
                    ],
                }
            ],
        }
    }


def _events_body() -> dict[str, Any]:
    return {
        "events": [
            {
                "build_event": {
                    "timestamp": "2026-08-07T00:01:00Z",
                    "payload": {"msg": "step `one` done"},
                }
            },
            # Empty msg — dropped entirely, matching the 2025 filter.
            {"build_event": {"timestamp": "2026-08-07T00:02:00Z", "payload": {"msg": ""}}},
            {
                "build_event": {
                    "timestamp": "2026-08-07T00:03:00Z",
                    "payload": {"msg": "finished"},
                }
            },
        ]
    }


def test_build_history_maps_events_dropping_empty_and_stripping_backticks() -> None:
    detail = build_status_detail(_status_body(), _events_body())

    assert detail["build_history"] == [
        {"time": "2026-08-07T00:01:00Z", "description": "step one done"},
        {"time": "2026-08-07T00:03:00Z", "description": "finished"},
    ]


def test_details_are_built_from_the_status_build_block() -> None:
    detail = build_status_detail(_status_body(), _events_body())

    assert detail["details"] == {
        "build_id": "b-123",
        "name": "my-build",
        "started_at": "2026-08-07T00:00:00Z",
        "updated_at": "2026-08-07T00:05:00Z",
        "status": "success",
        "source_pr": "https://github.example/pr/1",
        "description": "a build",
    }


def test_targets_flatten_runs_steps_and_artifacts() -> None:
    detail = build_status_detail(_status_body(), _events_body())

    assert detail["targets"] == [
        {
            "target_name": "train",
            "build_id": "b-123",
            "target_id": "t-1",
            "status": "success",
            "skipped_for_prerun_target_id": "",
            "input_artifacts": [{"artifact_id": "a-in", "uri": "s3://in"}],
            "output_artifacts": [{"artifact_id": "a-out", "uri": "s3://out"}],
            "steps": [
                {
                    "step_id": "s-1",
                    "uri": "step://1",
                    "status": "success",
                    "started_at": "2026-08-07T00:01:00Z",
                }
            ],
        }
    ]


def test_missing_events_yield_empty_build_history() -> None:
    detail = build_status_detail(_status_body(), {})

    assert detail["build_history"] == []


def test_empty_status_body_does_not_crash() -> None:
    detail = build_status_detail({}, {})

    assert detail["build_history"] == []
    assert detail["targets"] == []
    assert detail["details"]["build_id"] == ""


# A trimmed copy of a real gbserver /status body: a *failed* build whose hfpush
# step failed, yet a model output artifact was still registered. artifact_id /
# artifact_uri must be populated from it even though the build failed.
_REAL_FAILED_BUILD = {
    "status": {
        "build": {
            "uuid": "dcd60f52-4088-4f25-afdb-38ac7c4d9b9d",
            "name": "autotunex-granite-4.1-3b-financing-local-test",
            "status": "failed",
            "source_uri": "https://github.example/granite-dot-build/gbspace-public/pull/27649",
            "created_time": "2026-08-12T08:08:49.313923Z",
            "updated_time": "2026-08-12T08:30:42.368513Z",
        },
        "target_runs": [
            {
                "target": {"name": "custom", "uuid": "4e1543c7", "status": "failed"},
                "input_artifacts": [{"uuid": "7c5daf51", "uri": "hf://datasets/financing"}],
                "output_artifacts": [
                    {
                        "uuid": "d4affa76-52a8-4f57-bd5b-db49470fed5f",
                        "uri": "hf://huggingface.co/models/ibm-research/autotunex_a69082b7",
                        "status": "failed",
                    }
                ],
                "steps": [
                    {
                        "uuid": "59c126eb",
                        "definition_uri": "space://steps/hfpush",
                        "status": "failed",
                    }
                ],
            }
        ],
    },
    "retry_chain": None,
}


def test_output_artifact_ref_reads_first_target_first_output_artifact() -> None:
    detail = build_status_detail(_REAL_FAILED_BUILD, {})

    assert output_artifact_ref(detail) == {
        "artifact_id": "d4affa76-52a8-4f57-bd5b-db49470fed5f",
        "uri": "hf://huggingface.co/models/ibm-research/autotunex_a69082b7",
    }


def test_output_artifact_ref_is_none_when_there_are_no_targets() -> None:
    assert output_artifact_ref({"targets": [], "details": {}, "build_history": []}) is None


def test_output_artifact_ref_is_none_when_target_has_no_output_artifacts() -> None:
    detail: dict[str, Any] = {
        "targets": [{"output_artifacts": []}],
        "details": {},
        "build_history": [],
    }

    assert output_artifact_ref(detail) is None
