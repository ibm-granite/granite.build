"""The status vocabulary bridges two spellings with one declaration.

The database column holds ``PENDING`` (SQLAlchemy ``Enum`` persists a member's
name); the API emits ``pending`` (Pydantic serializes its value). These tests
pin both halves, because a change to either breaks the other silently.
"""

from __future__ import annotations

from autotunex.models.status import TERMINAL_RUN_STATUSES, DatasetStatus, GbTaskType, RunStatus


def test_run_status_has_the_six_states_the_schema_declares() -> None:
    names = {member.name for member in RunStatus}

    assert names == {"PENDING", "RUNNING", "PAUSED", "TERMINATED", "ERROR", "COMPLETED"}


def test_run_status_values_are_lowercase_for_the_api() -> None:
    assert RunStatus.PENDING.value == "pending"
    assert RunStatus.COMPLETED.value == "completed"


def test_run_status_names_are_uppercase_for_the_database() -> None:
    assert RunStatus.PENDING.name == "PENDING"


def test_terminal_statuses_are_the_three_with_no_outgoing_transitions() -> None:
    assert (
        frozenset({RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED})
        == TERMINAL_RUN_STATUSES
    )


def test_gb_task_type_values_match_the_schema_enum_verbatim() -> None:
    assert {member.value for member in GbTaskType} == {"RITS", "TUNING", "DOWNLOAD"}


def test_dataset_status_values_are_the_lowercase_lifecycle_strings() -> None:
    assert [s.value for s in DatasetStatus] == ["empty", "uploading", "ready", "error"]


def test_dataset_status_is_a_str_enum_so_it_compares_to_plain_strings() -> None:
    plain: str = "uploading"

    assert plain == DatasetStatus.UPLOADING
