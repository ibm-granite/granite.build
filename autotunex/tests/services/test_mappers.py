"""Snapshot precedence and task mapping.

``autotunex_jobs`` resolves a job's configuration name from
``config_snapshot['name']`` when present and falls back to the live
``configurations.name`` otherwise, so a job keeps reporting the configuration it
actually ran with even after that configuration is edited. These tests pin that
rule.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    TrialTable,
    UserTable,
)
from autotunex.models.configuration import ConfigurationJobRef
from autotunex.models.dataset import DatasetJobRef, DatasetPreview
from autotunex.models.status import GbTaskType, RunStatus
from autotunex.services.mappers import (
    configuration_to_read,
    dataset_to_read,
    latest_task_update,
    resolve_config_name,
    resolve_planned_trials,
    resolve_rl_tuner_type,
    trial_to_read,
    user_to_read,
)


def _task(updated_at: str | None, *, task_type: GbTaskType = GbTaskType.TUNING) -> GbTaskTable:
    return GbTaskTable(
        id=uuid4(),
        job_id=uuid4(),
        status=RunStatus.COMPLETED,
        type=task_type,
        updated_at=updated_at,
    )


def test_latest_task_update_returns_the_newest_timestamp() -> None:
    """A job's end is the last build task to report an update."""
    tasks = [
        _task("2026-08-11T09:00:00Z", task_type=GbTaskType.RITS),
        _task("2026-08-11T11:05:26Z", task_type=GbTaskType.TUNING),
    ]

    assert latest_task_update(tasks) == "2026-08-11T11:05:26Z"


def test_latest_task_update_ignores_tasks_without_a_timestamp() -> None:
    tasks = [_task(None), _task("2026-08-11T11:05:26Z", task_type=GbTaskType.DOWNLOAD)]

    assert latest_task_update(tasks) == "2026-08-11T11:05:26Z"


def test_latest_task_update_is_none_when_no_task_has_a_timestamp() -> None:
    assert latest_task_update([]) is None
    assert latest_task_update([_task(None)]) is None


def test_trial_to_read_tolerates_a_missing_timestamp() -> None:
    """A pipeline row carrying MySQL's zero date reads back as ``created_at=None``.

    ``UtcDateTime`` maps the unparseable zero date to ``None`` (see its own tests),
    so ``TrialRead`` must accept the absence rather than raise and 500 the whole
    job-detail response over one row the tuning pipeline wrote without a timestamp.
    """
    trial = TrialTable(
        id="t1",
        job_id=uuid4(),
        status=RunStatus.COMPLETED,
        created_at=None,
        updated_at=None,
    )

    read = trial_to_read(trial)

    assert read.created_at is None
    assert read.updated_at is None


def test_snapshot_name_wins_over_the_live_configuration(job: JobTable) -> None:
    job.config_snapshot = {"name": "as-it-ran"}

    assert resolve_config_name(job) == "as-it-ran"


def test_configuration_name_is_used_when_the_snapshot_has_no_name(job: JobTable) -> None:
    job.config_snapshot = {"rl_tuner_type": "ppo"}

    assert resolve_config_name(job) == "lora-sweep"


def test_configuration_name_is_used_when_the_snapshot_is_null(job: JobTable) -> None:
    job.config_snapshot = None

    assert resolve_config_name(job) == "lora-sweep"


def test_snapshot_rl_tuner_type_wins(job: JobTable) -> None:
    job.config_snapshot = {"rl_tuner_type": "ppo"}

    assert resolve_rl_tuner_type(job) == "ppo"


def test_rl_tuner_type_falls_back_to_the_configuration(job: JobTable) -> None:
    job.config_snapshot = None

    assert resolve_rl_tuner_type(job) is None


def test_planned_trials_reads_the_descriptor_default(job: JobTable) -> None:
    """The catalog shape: num_samples is a descriptor dict, the value is its default."""
    job.config_snapshot = {
        "config_data": {
            "tune_config": {
                "num_samples": {
                    "description": "Number of samples from the search space",
                    "default": 32,
                    "min_val": 1,
                    "max_val": 10000,
                    "type": "int",
                    "required": True,
                }
            }
        }
    }

    assert resolve_planned_trials(job) == 32


def test_planned_trials_reads_a_bare_scalar(job: JobTable) -> None:
    """The other shape in the wild — see src/ux/.../forms/default_config.ts."""
    job.config_snapshot = {"config_data": {"tune_config": {"num_samples": 4}}}

    assert resolve_planned_trials(job) == 4


def test_planned_trials_accepts_an_integral_float(job: JobTable) -> None:
    """JSON round-trips can widen an int; reporting 0 would be the misleading zero."""
    job.config_snapshot = {"config_data": {"tune_config": {"num_samples": 32.0}}}

    assert resolve_planned_trials(job) == 32


def test_planned_trials_is_zero_without_a_snapshot(job: JobTable) -> None:
    """Pipeline-written rows have no snapshot, so there is no budget to report."""
    job.config_snapshot = None

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_when_the_snapshot_has_no_config_data(job: JobTable) -> None:
    job.config_snapshot = {"name": "as-it-ran"}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_when_config_data_has_no_tune_config(job: JobTable) -> None:
    job.config_snapshot = {"config_data": {"training_config": {"num_epochs": 2}}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_when_tune_config_has_no_num_samples(job: JobTable) -> None:
    job.config_snapshot = {"config_data": {"tune_config": {"search_alg": "lds"}}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_when_the_descriptor_has_no_default(job: JobTable) -> None:
    job.config_snapshot = {
        "config_data": {"tune_config": {"num_samples": {"min_val": 1, "max_val": 10000}}}
    }

    assert resolve_planned_trials(job) == 0


def test_planned_trials_rejects_a_boolean(job: JobTable) -> None:
    """Bool is an int subclass in Python, so True would otherwise read as a budget of 1."""
    job.config_snapshot = {"config_data": {"tune_config": {"num_samples": True}}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_clamps_a_negative_to_zero(job: JobTable) -> None:
    """JobDetail.num_trials is Field(ge=0); an unclamped negative would 500 the read."""
    job.config_snapshot = {"config_data": {"tune_config": {"num_samples": -5}}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_for_a_non_numeric_value(job: JobTable) -> None:
    job.config_snapshot = {"config_data": {"tune_config": {"num_samples": "many"}}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_is_zero_when_tune_config_is_not_a_dict(job: JobTable) -> None:
    job.config_snapshot = {"config_data": {"tune_config": ["num_samples"]}}

    assert resolve_planned_trials(job) == 0


def test_planned_trials_never_falls_back_to_the_live_configuration(
    job: JobTable, configuration: ConfigurationTable
) -> None:
    """Unlike resolve_config_name, this reports only what the job actually ran with.

    The live row can have drifted since submit — that is what is_stale reports — so
    borrowing its budget would describe a search this job never ran.
    """
    configuration.config_data = {"tune_config": {"num_samples": {"default": 99}}}
    job.config_snapshot = {"config_data": {"tune_config": {"search_alg": "lds"}}}

    assert resolve_planned_trials(job) == 0


async def test_dataset_to_read_carries_generated_filenames_and_jobs(
    dataset: DatasetTable, session: AsyncSession
) -> None:
    # `dataset` is created before the generated columns are populated; refresh them.
    await session.refresh(dataset, ["status", "train_file", "validation_file"])
    job_ref = DatasetJobRef(id=uuid4(), experiment_name="exp", status=RunStatus.PENDING)

    read = dataset_to_read(dataset, associated_jobs=[job_ref])

    assert read.train_file == "alpaca_train"
    assert read.validation_file == "alpaca_validation"
    assert read.user_id == str(dataset.user_id)
    assert read.associated_jobs[0].experiment_name == "exp"
    assert read.preview is None


async def test_dataset_to_read_includes_preview_when_given(
    dataset: DatasetTable, session: AsyncSession
) -> None:
    await session.refresh(dataset, ["status", "train_file", "validation_file"])
    preview = DatasetPreview(train=[{"text": "hi"}], validation=[])

    read = dataset_to_read(dataset, associated_jobs=[], preview=preview)

    assert read.preview is not None
    assert read.preview.train == [{"text": "hi"}]


def test_configuration_to_read_carries_associated_jobs() -> None:
    # created_at/updated_at are populated by the ORM's column default on flush,
    # not on bare construction, so they are supplied explicitly here (as
    # FakeConfigurationRepository.seed() does) rather than left for a
    # never-flushed row to leave as None.
    now = datetime.now(UTC)
    configuration = ConfigurationTable(
        id=uuid4(),
        user_id=str(uuid4()),
        name="cfg",
        tuner_type="optuna",
        rl_tuner_type=None,
        config_data={"k": 1},
        created_at=now,
        updated_at=now,
    )
    job_ref = ConfigurationJobRef(id=uuid4(), experiment_name="exp", status=RunStatus.PENDING)

    read = configuration_to_read(configuration, [job_ref])

    assert read.name == "cfg"
    assert [j.experiment_name for j in read.associated_jobs] == ["exp"]


def test_configuration_to_read_defaults_to_no_jobs() -> None:
    now = datetime.now(UTC)
    configuration = ConfigurationTable(
        id=uuid4(),
        user_id=str(uuid4()),
        name="cfg",
        config_data={"k": 1},
        created_at=now,
        updated_at=now,
    )

    read = configuration_to_read(configuration, [])

    assert read.associated_jobs == []


def test_log_entry_to_read_copies_the_line_fields() -> None:
    from datetime import datetime

    from autotunex.db.tables import LogEntryTable
    from autotunex.services.mappers import log_entry_to_read

    entry = LogEntryTable(
        id=5,
        job_id=uuid4(),
        trial_id=None,
        level="WARN",
        filename="t.py",
        message="hi",
        iteration=2,
        epoch=1.5,
        timestamp=datetime(2026, 8, 10, 9, 0, 0),
    )

    read = log_entry_to_read(entry)

    assert read.id == 5
    assert read.level == "WARN"
    assert read.epoch == 1.5


def test_user_to_read_copies_every_field() -> None:
    user = UserTable(
        id=uuid4(),
        email="a@example.com",
        role="admin",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    read = user_to_read(user)

    assert read.id == user.id
    assert read.email == "a@example.com"
    assert read.role == "admin"
    assert read.created_at == user.created_at
    assert read.updated_at == user.updated_at
