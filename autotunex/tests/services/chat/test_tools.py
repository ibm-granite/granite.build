"""Tests for :mod:`autotunex.services.chat.tools`."""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    ResultTable,
    TrialTable,
)
from autotunex.models.auth import Principal
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.autotune import AutotuneCoreAdapter
from autotunex.services.chat.context import ToolContext
from autotunex.services.chat.tools import (
    TOOL_LABELS,
    TOOL_REFRESH_TARGETS,
    TOOLS,
    openai_tool_specs,
    run_tool,
)


def _ctx(session_factory: async_sessionmaker[AsyncSession], principal: Principal) -> ToolContext:
    return ToolContext(
        principal=principal, settings=Settings(job_backend="none"), session_factory=session_factory
    )


def test_registry_has_no_user_email_argument() -> None:
    """No tool takes an email — identity always comes from the context's principal."""
    for spec in TOOLS.values():
        assert "user_email" not in spec.params.model_fields


def test_openai_tool_specs_cover_every_tool() -> None:
    """Every registered tool is exposed as an OpenAI function spec."""
    names = {t["function"]["name"] for t in openai_tool_specs()}

    assert names == set(TOOLS)


def test_openai_tool_specs_have_the_expected_shape() -> None:
    """Each spec carries a type, name, description, and a JSON schema."""
    for spec in openai_tool_specs():
        assert spec["type"] == "function"
        function = spec["function"]
        assert isinstance(function["name"], str) and function["name"]
        assert isinstance(function["description"], str) and function["description"]
        assert (
            "properties" in function["parameters"] or function["parameters"].get("type") == "object"
        )


def test_tool_labels_cover_every_tool() -> None:
    """Every tool has a friendly, present-continuous status label."""
    assert set(TOOL_LABELS) == set(TOOLS)


def test_tool_refresh_targets_are_the_two_write_tools() -> None:
    """Only the two write tools trigger a UI refresh, and on the documented views."""
    assert TOOL_REFRESH_TARGETS == {"start_tuning_job": "tunings", "create_config": "configs"}


async def test_list_jobs_reports_empty_for_a_fresh_owner(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A freshly-provisioned owner with no jobs gets a friendly empty message."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_jobs", {}, ctx)

    assert "No fine-tuning jobs" in out


async def test_list_configs_reports_empty_for_a_fresh_owner(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A freshly-provisioned owner with no configurations gets a friendly empty message."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_configs", {}, ctx)

    assert "No configurations" in out


async def test_list_datasets_reports_empty_for_a_fresh_owner(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A freshly-provisioned owner with no datasets gets a friendly empty message."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_datasets", {}, ctx)

    assert "No datasets" in out


async def test_get_job_returns_error_string_when_missing(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A nonexistent job id becomes a plain error string, never a raised exception."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job", {"job_id": str(uuid.uuid4())}, ctx)

    assert "not found" in out.lower()
    assert out.startswith("Error:")


async def test_get_config_returns_error_string_when_missing(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A nonexistent configuration id becomes a plain error string."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_config", {"config_id": str(uuid.uuid4())}, ctx)

    assert "not found" in out.lower()


async def test_get_dataset_returns_error_string_when_missing(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A nonexistent dataset id becomes a plain error string."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_dataset", {"dataset_id": str(uuid.uuid4())}, ctx)

    assert "not found" in out.lower()


async def test_run_tool_reports_bad_arguments(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Missing required arguments never raise — they become an error string."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job", {}, ctx)  # missing required job_id

    assert "job_id" in out or "invalid" in out.lower()


async def test_run_tool_reports_a_malformed_uuid(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A syntactically-invalid UUID is a ``ValueError`` inside the handler, still caught."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job", {"job_id": "not-a-uuid"}, ctx)

    assert out.startswith("Error:")


async def test_run_tool_reports_an_unknown_tool_name(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """An unregistered tool name is a clean error string, not a KeyError."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("delete_the_database", {}, ctx)

    assert "unknown tool" in out.lower()


async def test_get_user_metadata_reports_zero_counts_for_a_fresh_owner(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A freshly-provisioned owner's metadata is present and starts at zero."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_user_metadata", {}, ctx)

    payload = json.loads(out)
    assert payload == {
        "number_of_jobs": 0,
        "number_of_configurations": 0,
        "number_of_datasets": 0,
    }


async def test_get_user_info_reports_the_calling_principal(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``get_user_info`` reflects the context's own principal, never another's."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_user_info", {}, ctx)

    payload = json.loads(out)
    assert payload == {
        "email": provisioned_principal.email,
        "user_id": str(provisioned_principal.user_id),
        "is_admin": provisioned_principal.is_admin,
    }


async def test_get_supported_dataset_types_never_raises(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """The dataset-type catalog tool degrades gracefully without the optional core."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_supported_dataset_types", {}, ctx)

    assert isinstance(out, str) and out


# --- seeding helpers --------------------------------------------------------
#
# The handler bodies below are only reachable with rows to serialize, so these
# build the smallest graph each tool actually reads. Rows go in through
# ``session_factory`` (the same in-memory engine ``ToolContext`` opens), owned by
# ``provisioned_principal``, so every assertion runs through the real
# ownership-scoped services rather than a stand-in.


async def _seed_config(
    factory: async_sessionmaker[AsyncSession],
    owner: Principal,
    *,
    name: str = "lora-sweep",
    tuner_type: str | None = "bayesian",
    rl_tuner_type: str | None = None,
) -> ConfigurationTable:
    configuration = ConfigurationTable(
        id=uuid.uuid4(),
        user_id=str(owner.user_id),
        name=name,
        tuner_type=tuner_type,
        rl_tuner_type=rl_tuner_type,
        config_data={"tune_config": {"num_samples": 7}},
    )
    async with factory() as db:
        db.add(configuration)
        await db.commit()
    return configuration


async def _seed_dataset(
    factory: async_sessionmaker[AsyncSession],
    owner: Principal,
    *,
    name: str = "alpaca",
    train_records: int | None = 120,
    validation_records: int | None = 30,
) -> DatasetTable:
    dataset = DatasetTable(
        id=uuid.uuid4(),
        user_id=str(owner.user_id),
        name=name,
        description="Instruction data.",
        data_format="jsonl",
        status=DatasetStatus.READY,
        train_records=train_records,
        train_file_size=4096,
        validation_records=validation_records,
    )
    async with factory() as db:
        db.add(dataset)
        await db.commit()
        # `train_file`/`validation_file` are stored generated columns derived from
        # `name`, so they only have a value once the row is read back.
        await db.refresh(dataset, ["train_file", "validation_file"])
    return dataset


async def _seed_job(
    factory: async_sessionmaker[AsyncSession],
    owner: Principal,
    configuration: ConfigurationTable,
    dataset: DatasetTable,
    *,
    experiment_name: str = "granite-lora-sep",
    with_task: bool = False,
) -> JobTable:
    job = JobTable(
        id=uuid.uuid4(),
        user_id=str(owner.user_id),
        status=RunStatus.RUNNING,
        seed=42,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="ibm-granite/granite-3.0-2b-instruct",
        model_source="huggingface",
        experiment_name=experiment_name,
        tuning_type="lora",
        # `num_trials` is the budget read off this snapshot, not a row count.
        config_snapshot={"config_data": {"tune_config": {"num_samples": 7}}},
    )
    async with factory() as db:
        db.add(job)
        if with_task:
            db.add(
                GbTaskTable(
                    id=uuid.uuid4(),
                    job_id=job.id,
                    status=RunStatus.RUNNING,
                    type=GbTaskType.TUNING,
                )
            )
        await db.commit()
    return job


async def _seed_trial(
    factory: async_sessionmaker[AsyncSession],
    job: JobTable,
    trial_id: str,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    metric: str | None = None,
) -> None:
    async with factory() as db:
        db.add(
            TrialTable(id=trial_id, job_id=job.id, status=status, config={"learning_rate": 3e-4})
        )
        if metric is not None:
            db.add(
                ResultTable(
                    id=uuid.uuid4(),
                    job_id=job.id,
                    trial_id=trial_id,
                    metric=metric,
                    metrics={"eval_loss": 0.42},
                )
            )
        await db.commit()


# --- read tools over a seeded graph -----------------------------------------


async def test_list_jobs_renders_each_job_as_a_markdown_line(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """The listing names the experiment, id, status and model for every job."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    job = await _seed_job(session_factory, provisioned_principal, configuration, dataset)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_jobs", {}, ctx)

    assert "**1 job(s):**" in out
    assert "granite-lora-sep" in out
    assert str(job.id) in out
    assert "status: `running`" in out
    assert "ibm-granite/granite-3.0-2b-instruct" in out


async def test_get_job_returns_its_key_fields_as_json(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``get_job`` reports identity, the trial budget, and how many tasks exist."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    job = await _seed_job(
        session_factory, provisioned_principal, configuration, dataset, with_task=True
    )
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job", {"job_id": str(job.id)}, ctx)

    assert json.loads(out) == {
        "id": str(job.id),
        "experiment_name": "granite-lora-sep",
        "status": "running",
        "model": "ibm-granite/granite-3.0-2b-instruct",
        "model_source": "huggingface",
        "num_trials": 7,
        "task_count": 1,
        "tuning_type": "lora",
    }


async def test_get_job_trials_returns_id_and_status_only(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """The trials tool stays in summary shape — no per-trial config or metrics."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    job = await _seed_job(session_factory, provisioned_principal, configuration, dataset)
    await _seed_trial(session_factory, job, "t1")
    await _seed_trial(session_factory, job, "t2", status=RunStatus.RUNNING)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job_trials", {"job_id": str(job.id)}, ctx)

    assert json.loads(out) == [
        {"id": "t1", "status": "completed"},
        {"id": "t2", "status": "running"},
    ]


async def test_get_job_results_includes_each_trials_reported_metrics(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Metrics come from the trial's merged one-to-one results row."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    job = await _seed_job(session_factory, provisioned_principal, configuration, dataset)
    await _seed_trial(session_factory, job, "t1", metric="eval_loss")
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_job_results", {"job_id": str(job.id)}, ctx)

    assert json.loads(out) == [
        {
            "trial_id": "t1",
            "status": "completed",
            "metric": "eval_loss",
            "metrics": {"eval_loss": 0.42},
        }
    ]


async def test_get_trial_logs_returns_the_stored_log_lines(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Each line keeps its level, message, iteration and epoch."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    job = await _seed_job(session_factory, provisioned_principal, configuration, dataset)
    async with session_factory() as db:
        db.add(
            LogEntryTable(
                job_id=job.id,
                trial_id="t1",
                level="INFO",
                message="step 10 done",
                iteration=10,
                epoch=0.5,
            )
        )
        await db.commit()
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_trial_logs", {"job_id": str(job.id), "trial_id": "t1"}, ctx)

    assert json.loads(out) == [
        {"level": "INFO", "message": "step 10 done", "iteration": 10, "epoch": 0.5}
    ]


async def test_list_configs_renders_the_rl_tuner_only_when_one_is_set(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """An RL configuration shows both tuners; a plain HPO one shows only its own."""
    await _seed_config(session_factory, provisioned_principal, name="rl", rl_tuner_type="grpo")
    await _seed_config(session_factory, provisioned_principal, name="hpo-only")
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_configs", {}, ctx)

    assert "**2 configuration(s):**" in out
    # Matched per line rather than over the whole listing: the two rows'
    # order is the service's to choose, and neither assertion depends on it.
    rl_line = next(line for line in out.splitlines() if "**rl**" in line)
    plain_line = next(line for line in out.splitlines() if "**hpo-only**" in line)
    assert "tuner: `bayesian` / `grpo`" in rl_line
    assert "tuner: `bayesian`" in plain_line
    assert "grpo" not in plain_line


async def test_list_configs_reports_an_absent_tuner_type_as_not_applicable(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A configuration with no HPO tuner recorded renders ``n/a``, not ``None``."""
    await _seed_config(session_factory, provisioned_principal, name="untyped", tuner_type=None)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_configs", {}, ctx)

    assert "tuner: `n/a`" in out


async def test_get_config_returns_its_fields_including_config_data(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``get_config`` carries the search space itself, not just the name."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_config", {"config_id": str(configuration.id)}, ctx)

    assert json.loads(out) == {
        "id": str(configuration.id),
        "name": "lora-sweep",
        "tuner_type": "bayesian",
        "rl_tuner_type": None,
        "config_data": {"tune_config": {"num_samples": 7}},
    }


async def test_list_datasets_shows_a_question_mark_for_unknown_record_counts(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A dataset whose counts are not yet known renders as ``?`` rather than ``None``."""
    await _seed_dataset(
        session_factory,
        provisioned_principal,
        name="counted",
        train_records=120,
        validation_records=30,
    )
    await _seed_dataset(
        session_factory,
        provisioned_principal,
        name="uncounted",
        train_records=None,
        validation_records=None,
    )
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("list_datasets", {}, ctx)

    assert "**2 dataset(s):**" in out
    assert "train: 120, val: 30" in out
    assert "train: ?, val: ?" in out


async def test_get_dataset_returns_its_file_and_record_fields(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """``get_dataset`` reports status and the per-split file/record details."""
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_dataset", {"dataset_id": str(dataset.id)}, ctx)

    payload = json.loads(out)
    assert payload["id"] == str(dataset.id)
    assert payload["name"] == "alpaca"
    assert payload["status"] == "ready"
    assert payload["train_file"] == "alpaca_train"
    assert payload["train_records"] == 120
    assert payload["validation_records"] == 30


# --- write tools ------------------------------------------------------------


async def test_create_config_persists_a_configuration_the_caller_owns(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """The created row is readable back through the caller's own scope."""
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool(
        "create_config",
        {"name": "from-chat", "config_data": {"tune_config": {"num_samples": 3}}},
        ctx,
    )

    created = json.loads(out)
    assert created["name"] == "from-chat"
    readback = json.loads(await run_tool("get_config", {"config_id": created["id"]}, ctx))
    assert readback["config_data"] == {"tune_config": {"num_samples": 3}}


async def test_start_tuning_job_submits_a_job_owned_by_the_caller(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A submitted job reports ``started`` and shows up in the caller's own listing."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = await _seed_dataset(session_factory, provisioned_principal)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool(
        "start_tuning_job",
        {
            "config_id": str(configuration.id),
            "dataset_id": str(dataset.id),
            "model": "ibm-granite/granite-3.0-2b-instruct",
            "experiment_name": "from-chat-run",
        },
        ctx,
    )

    payload = json.loads(out)
    assert payload["status"] == "started"
    assert payload["experiment_name"] == "from-chat-run"
    assert "from-chat-run" in await run_tool("list_jobs", {}, ctx)


async def test_start_tuning_job_reports_a_domain_failure_as_an_error_string(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """A dataset that is not ``ready`` is refused by the service, not by the tool."""
    configuration = await _seed_config(session_factory, provisioned_principal)
    dataset = DatasetTable(
        id=uuid.uuid4(),
        user_id=str(provisioned_principal.user_id),
        name="empty-ds",
        description="Not uploaded yet.",
        status=DatasetStatus.EMPTY,
    )
    async with session_factory() as db:
        db.add(dataset)
        await db.commit()
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool(
        "start_tuning_job",
        {
            "config_id": str(configuration.id),
            "dataset_id": str(dataset.id),
            "model": "ibm-granite/granite-3.0-2b-instruct",
            "experiment_name": "too-early",
        },
        ctx,
    )

    assert out.startswith("Error:")


# --- the optional autotune core ---------------------------------------------


async def test_get_config_template_returns_the_cores_template_or_a_clean_error(
    session_factory: async_sessionmaker[AsyncSession], provisioned_principal: Principal
) -> None:
    """Present-core and absent-core installs both yield a string, never a raise.

    The vendored ``fm-tune`` catalog is installed in CI, so this normally takes
    the happy path; the ``Error:`` branch keeps the test honest on an install
    without it (see ``test_get_supported_dataset_types_never_raises``).
    """
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_config_template", {}, ctx)

    assert out.startswith("Error:") or isinstance(json.loads(out), dict)


async def test_get_supported_dataset_types_names_the_catalog_when_the_core_is_absent(
    session_factory: async_sessionmaker[AsyncSession],
    provisioned_principal: Principal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This tool catches an unavailable core itself so the message names the catalog."""

    async def _unavailable(_self: AutotuneCoreAdapter) -> dict[str, object]:
        raise AutotuneCoreUnavailableError()

    monkeypatch.setattr(AutotuneCoreAdapter, "get_dataset_types", _unavailable)
    ctx = _ctx(session_factory, provisioned_principal)

    out = await run_tool("get_supported_dataset_types", {}, ctx)

    assert out == "Dataset-type catalog unavailable (autotune core not installed)."
