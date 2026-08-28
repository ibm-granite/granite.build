# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The AutoTuneX tool surface — the single source of truth for chat + MCP.

Each tool is a name, description, a Pydantic params model (its schema), and an
async handler that calls the Principal-scoped services from
:mod:`autotunex.services.chat.context`. Identity is always the context's
principal — no tool takes an email, so a caller can never act as someone else.
Handlers return human/agent-facing text; domain failures are converted to a
short error string for the model, never raised into a stream.

The registry (:data:`TOOLS`) is consumed two ways: the in-app chat agent calls
:func:`run_tool` directly and advertises :func:`openai_tool_specs` to the LLM;
the opt-in MCP server wraps each entry as an ``@mcp.tool``. Both paths go
through the exact same handler, so there is exactly one place a tool's
behavior — and its ownership scoping — is defined.

MUST NOT import fastapi or fastmcp.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autotunex.core.exceptions import AutotuneCoreUnavailableError, AutoTuneXError
from autotunex.core.logging import get_logger
from autotunex.models.configuration import ConfigurationCreate
from autotunex.models.job import JobCreate
from autotunex.services.autotune import AutotuneCoreAdapter
from autotunex.services.chat.context import ScopedServices, ToolContext

logger = get_logger(__name__)

_LIST_LIMIT = 50
"""Cap on items returned by list tools, keeping the chat context window lean."""

_LOG_LIMIT = 30
"""Cap on trial log lines returned by ``get_trial_logs``."""


class ToolError(Exception):
    """A handler-level failure whose message is safe to show the model.

    Distinct from :class:`~autotunex.core.exceptions.AutoTuneXError`: that
    hierarchy is for domain rules the services already enforce, while this is
    for tool-local failures (e.g. a malformed identifier) that have no service
    exception of their own. Both are converted to the same ``"Error: ..."``
    shape by :func:`run_tool`.
    """


@dataclass(slots=True)
class ToolSpec:
    """One entry in the tool registry: a name, schema, description, and handler."""

    name: str
    description: str
    params: type[BaseModel]
    handler: Callable[[ScopedServices, BaseModel], Awaitable[str]]


class _NoArgs(BaseModel):
    """Params for a tool that takes no arguments."""

    model_config = ConfigDict(extra="forbid")


class _JobIdParams(BaseModel):
    """Params for a tool keyed by job id."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="The job's UUID.")


class _TrialLogsParams(BaseModel):
    """Params for fetching one trial's log lines.

    Both ``job_id`` and ``trial_id`` are required: ``LogService`` scopes a
    trial's logs through its owning job (a trial id alone cannot be verified
    against ``trials.id`` — see the log-service module docstring), so the
    caller must supply both, unlike the 2025 tool that took only ``trial_id``.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="The owning job's UUID.")
    trial_id: str = Field(description="The trial's short opaque id.")


class _ConfigIdParams(BaseModel):
    """Params for a tool keyed by configuration id."""

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(description="The configuration's UUID.")


class _DatasetIdParams(BaseModel):
    """Params for a tool keyed by dataset id."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(description="The dataset's UUID.")


class _CreateConfigParams(BaseModel):
    """Params for creating a new hyperparameter configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name for the new configuration.")
    config_data: dict[str, Any] = Field(
        description="Hyperparameter search space / tuning settings as a non-empty JSON object."
    )
    tuner_type: str = Field(default="bayesian", description="HPO algorithm, e.g. 'bayesian'.")
    rl_tuner_type: str | None = Field(
        default=None, description="RL tuner type, if this configuration drives an RL run."
    )


class _StartTuningJobParams(BaseModel):
    """Params for submitting a new tuning job."""

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(description="The hyperparameter configuration's UUID.")
    dataset_id: str = Field(description="The training dataset's UUID.")
    model: str = Field(description="Model identifier, e.g. 'ibm-granite/granite-3.0-2b-instruct'.")
    experiment_name: str = Field(description="A unique name for this experiment.")
    model_source: Literal["huggingface", "custom_path"] = Field(
        default="huggingface", description="Where the model comes from."
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")


async def _list_jobs(svc: ScopedServices, _params: BaseModel) -> str:
    """List the caller's fine-tuning jobs as a pre-formatted markdown listing."""
    page = await svc.job.list(limit=_LIST_LIMIT, offset=0)
    if not page.items:
        return "No fine-tuning jobs found for this user."
    lines = [f"**{len(page.items)} job(s):**\n"]
    for i, j in enumerate(page.items, 1):
        lines.append(
            f"{i}. **{j.experiment_name}** (id: `{j.id}`) — "
            f"status: `{j.status.value}`, model: `{j.model}`"
        )
    return "\n".join(lines)


async def _get_job(svc: ScopedServices, params: BaseModel) -> str:
    """Return one job's key fields (without logs — use ``get_trial_logs`` for those)."""
    assert isinstance(params, _JobIdParams)
    job = await svc.job.get(UUID(params.job_id))
    return json.dumps(
        {
            "id": str(job.id),
            "experiment_name": job.experiment_name,
            "status": job.status.value,
            "model": job.model,
            "model_source": job.model_source,
            "num_trials": job.num_trials,
            "task_count": len(job.tasks),
            "tuning_type": job.tuning_type,
        },
        default=str,
    )


async def _get_job_trials(svc: ScopedServices, params: BaseModel) -> str:
    """Return one job's trials in summary shape (id, status), max 50."""
    assert isinstance(params, _JobIdParams)
    job = await svc.job.get(UUID(params.job_id))
    trials = [{"id": t.id, "status": t.status.value} for t in job.trials[:_LIST_LIMIT]]
    return json.dumps(trials, default=str)


async def _get_job_results(svc: ScopedServices, params: BaseModel) -> str:
    """Return one job's trials with their reported metrics, max 50.

    Metrics are sourced from each trial's own ``metric``/``metrics`` fields —
    :class:`~autotunex.models.trial.TrialRead` already merges the one-to-one
    ``results`` row for the caller, so there is no separate results lookup here.
    """
    assert isinstance(params, _JobIdParams)
    job = await svc.job.get(UUID(params.job_id))
    results = [
        {"trial_id": t.id, "status": t.status.value, "metric": t.metric, "metrics": t.metrics}
        for t in job.trials[:_LIST_LIMIT]
    ]
    return json.dumps(results, default=str)


async def _get_trial_logs(svc: ScopedServices, params: BaseModel) -> str:
    """Return the last 30 log lines for one trial of one job."""
    assert isinstance(params, _TrialLogsParams)
    page = await svc.logs.get_trial_logs(
        UUID(params.job_id), params.trial_id, before_id=0, limit=_LOG_LIMIT
    )
    lines = [
        {
            "level": entry.level,
            "message": entry.message,
            "iteration": entry.iteration,
            "epoch": entry.epoch,
        }
        for entry in page.logs
    ]
    return json.dumps(lines, default=str)


async def _list_configs(svc: ScopedServices, _params: BaseModel) -> str:
    """List the caller's hyperparameter configurations as a markdown listing."""
    page = await svc.config.list(limit=_LIST_LIMIT, offset=0)
    if not page.items:
        return "No configurations found for this user."
    lines = [f"**{len(page.items)} configuration(s):**\n"]
    for i, c in enumerate(page.items, 1):
        tuner_str = f"`{c.tuner_type or 'n/a'}`"
        if c.rl_tuner_type:
            tuner_str += f" / `{c.rl_tuner_type}`"
        lines.append(f"{i}. **{c.name}** (id: `{c.id}`) — tuner: {tuner_str}")
    return "\n".join(lines)


async def _get_config(svc: ScopedServices, params: BaseModel) -> str:
    """Return one configuration's fields (without its associated jobs)."""
    assert isinstance(params, _ConfigIdParams)
    configuration = await svc.config.get(UUID(params.config_id))
    return json.dumps(
        {
            "id": str(configuration.id),
            "name": configuration.name,
            "tuner_type": configuration.tuner_type,
            "rl_tuner_type": configuration.rl_tuner_type,
            "config_data": configuration.config_data,
        },
        default=str,
    )


async def _get_config_template(svc: ScopedServices, _params: BaseModel) -> str:
    """Return the autotune core's starter configuration template.

    Server-global, not caller-scoped (see ``ConfigurationService.get_template``).
    An unavailable autotune core raises ``AutotuneCoreUnavailableError``, which
    :func:`run_tool` converts to a short error string like every other domain
    error — no bespoke handling is needed here.
    """
    template = await svc.config.get_template()
    return json.dumps(template, default=str)


async def _create_config(svc: ScopedServices, params: BaseModel) -> str:
    """Create a new hyperparameter configuration owned by the caller."""
    assert isinstance(params, _CreateConfigParams)
    created = await svc.config.create(
        ConfigurationCreate(
            name=params.name,
            tuner_type=params.tuner_type,
            rl_tuner_type=params.rl_tuner_type,
            config_data=params.config_data,
        )
    )
    return json.dumps({"id": str(created.id), "name": created.name})


async def _list_datasets(svc: ScopedServices, _params: BaseModel) -> str:
    """List the caller's datasets as a pre-formatted markdown listing."""
    page = await svc.dataset.list(limit=_LIST_LIMIT, offset=0)
    if not page.items:
        return "No datasets found for this user."
    lines = [f"**{len(page.items)} dataset(s):**\n"]
    for i, d in enumerate(page.items, 1):
        train = d.train_records if d.train_records is not None else "?"
        val = d.validation_records if d.validation_records is not None else "?"
        lines.append(f"{i}. **{d.name}** (id: `{d.id}`) — train: {train}, val: {val}")
    return "\n".join(lines)


async def _get_dataset(svc: ScopedServices, params: BaseModel) -> str:
    """Return one dataset's fields (without a data preview)."""
    assert isinstance(params, _DatasetIdParams)
    dataset = await svc.dataset.get(UUID(params.dataset_id))
    return json.dumps(
        {
            "id": str(dataset.id),
            "name": dataset.name,
            "description": dataset.description,
            "data_format": dataset.data_format,
            "status": dataset.status.value,
            "train_file": dataset.train_file,
            "train_records": dataset.train_records,
            "train_file_size": dataset.train_file_size,
            "validation_file": dataset.validation_file,
            "validation_records": dataset.validation_records,
            "validation_file_size": dataset.validation_file_size,
        },
        default=str,
    )


async def _get_supported_dataset_types(_svc: ScopedServices, _params: BaseModel) -> str:
    """Return autotune's dataset-type catalog directly from the core adapter.

    ``ScopedServices`` exposes no dataset-type-catalog method (it belongs to
    the ``AutotuneCore`` seam, not any Principal-scoped service), so this is
    the one handler that builds an :class:`AutotuneCoreAdapter` itself. The
    catalog is server-global, so nothing here needs the principal. Unlike
    every other handler, an unavailable core is caught here rather than left
    to :func:`run_tool`, so the message can name this catalog specifically.
    """
    try:
        dataset_types = await AutotuneCoreAdapter().get_dataset_types()
    except AutotuneCoreUnavailableError:
        return "Dataset-type catalog unavailable (autotune core not installed)."
    return json.dumps(dataset_types, default=str)


async def _get_user_info(svc: ScopedServices, _params: BaseModel) -> str:
    """Return the calling principal's own identity — never another caller's."""
    principal = svc.principal
    return json.dumps(
        {
            "email": principal.email,
            "user_id": str(principal.user_id) if principal.user_id is not None else None,
            "is_admin": principal.is_admin,
        }
    )


async def _get_user_metadata(svc: ScopedServices, _params: BaseModel) -> str:
    """Return the calling principal's own job/configuration/dataset counts."""
    metadata = await svc.user.my_metadata()
    return json.dumps(metadata.model_dump())


async def _start_tuning_job(svc: ScopedServices, params: BaseModel) -> str:
    """Submit a new tuning job owned by the calling principal."""
    assert isinstance(params, _StartTuningJobParams)
    created = await svc.job.create(
        JobCreate(
            config_id=UUID(params.config_id),
            dataset_id=UUID(params.dataset_id),
            model=params.model,
            model_source=params.model_source,
            experiment_name=params.experiment_name,
            seed=params.seed,
        )
    )
    return json.dumps(
        {"status": "started", "experiment_name": created.experiment_name, "id": str(created.id)}
    )


TOOLS: dict[str, ToolSpec] = {
    "list_jobs": ToolSpec(
        name="list_jobs",
        description=(
            "List fine-tuning jobs for the calling user (most recent 50). Returns a "
            "pre-formatted text listing — present it to the user as-is."
        ),
        params=_NoArgs,
        handler=_list_jobs,
    ),
    "get_job": ToolSpec(
        name="get_job",
        description="Get details about a fine-tuning job (without logs — use get_trial_logs).",
        params=_JobIdParams,
        handler=_get_job,
    ),
    "get_job_trials": ToolSpec(
        name="get_job_trials",
        description="Get trials for a fine-tuning job (summary, max 50).",
        params=_JobIdParams,
        handler=_get_job_trials,
    ),
    "get_job_results": ToolSpec(
        name="get_job_results",
        description="Get results for a fine-tuning job (max 50 trials, core metrics only).",
        params=_JobIdParams,
        handler=_get_job_results,
    ),
    "get_trial_logs": ToolSpec(
        name="get_trial_logs",
        description="Get the last 30 training log entries for a trial of a job.",
        params=_TrialLogsParams,
        handler=_get_trial_logs,
    ),
    "list_configs": ToolSpec(
        name="list_configs",
        description=(
            "List hyperparameter configurations for the calling user (max 50). Returns "
            "a pre-formatted text listing — present it to the user as-is."
        ),
        params=_NoArgs,
        handler=_list_configs,
    ),
    "get_config": ToolSpec(
        name="get_config",
        description="Get a specific hyperparameter configuration (without associated jobs).",
        params=_ConfigIdParams,
        handler=_get_config,
    ),
    "get_config_template": ToolSpec(
        name="get_config_template",
        description=(
            "Get the default AutoTuneX configuration template showing all supported "
            "hyperparameters."
        ),
        params=_NoArgs,
        handler=_get_config_template,
    ),
    "create_config": ToolSpec(
        name="create_config",
        description="Create a new hyperparameter configuration for the calling user.",
        params=_CreateConfigParams,
        handler=_create_config,
    ),
    "list_datasets": ToolSpec(
        name="list_datasets",
        description=(
            "List datasets for the calling user (max 50). Returns a pre-formatted "
            "text listing — present it to the user as-is."
        ),
        params=_NoArgs,
        handler=_list_datasets,
    ),
    "get_dataset": ToolSpec(
        name="get_dataset",
        description="Get details about a specific dataset (without a data preview).",
        params=_DatasetIdParams,
        handler=_get_dataset,
    ),
    "get_supported_dataset_types": ToolSpec(
        name="get_supported_dataset_types",
        description="Get the list of supported dataset file types and formats.",
        params=_NoArgs,
        handler=_get_supported_dataset_types,
    ),
    "get_user_info": ToolSpec(
        name="get_user_info",
        description="Get the calling user's own profile information.",
        params=_NoArgs,
        handler=_get_user_info,
    ),
    "get_user_metadata": ToolSpec(
        name="get_user_metadata",
        description=(
            "Get the calling user's own statistics — number of jobs, configurations and datasets."
        ),
        params=_NoArgs,
        handler=_get_user_metadata,
    ),
    "start_tuning_job": ToolSpec(
        name="start_tuning_job",
        description="Start a new fine-tuning job for the calling user.",
        params=_StartTuningJobParams,
        handler=_start_tuning_job,
    ),
}
"""The tool registry: the single source of truth for chat and MCP alike."""

TOOL_LABELS: dict[str, str] = {
    "list_jobs": "Looking up your jobs…",
    "get_job": "Fetching job details…",
    "get_job_trials": "Loading job trials…",
    "get_job_results": "Loading job results…",
    "get_trial_logs": "Reading trial logs…",
    "list_configs": "Looking up your configurations…",
    "get_config": "Fetching configuration details…",
    "get_config_template": "Loading configuration template…",
    "create_config": "Creating your configuration…",
    "list_datasets": "Looking up your datasets…",
    "get_dataset": "Fetching dataset details…",
    "get_supported_dataset_types": "Checking supported dataset types…",
    "get_user_info": "Loading your profile…",
    "get_user_metadata": "Loading your account stats…",
    "start_tuning_job": "Starting your tuning job…",
}
"""Friendly, present-continuous status strings shown while a tool runs.

Ported from the 2025 ``chat_service.py``, minus ``get_job_assets`` and
``list_published_models`` (dropped tools — DMF and RITS asset listing are not
part of this registry).
"""

TOOL_REFRESH_TARGETS: dict[str, str] = {"start_tuning_job": "tunings", "create_config": "configs"}
"""Write tools whose success should trigger a UI refresh, and which view."""


async def run_tool(tool_name: str, arguments: dict[str, Any], ctx: ToolContext) -> str:
    """Validate args, run the tool as the context's principal, and never raise upstream.

    The three failure shapes a caller (chat agent or MCP client) might trigger —
    an unknown tool name, invalid arguments, or a domain rule violation deep
    inside a handler — all resolve to a plain ``"Error: ..."`` string rather
    than propagating, so a broken tool call never crashes a chat turn or an MCP
    response.
    """
    spec = TOOLS.get(tool_name)
    if spec is None:
        return f"Error: unknown tool '{tool_name}'."
    try:
        params = spec.params.model_validate(arguments)
    except ValidationError as exc:
        return f"Error: invalid arguments for {tool_name}: {exc.errors()}"
    try:
        async with ctx.services() as svc:
            return await spec.handler(svc, params)
    except AutoTuneXError as exc:
        return f"Error: {exc.detail}"
    except (ValueError, ToolError, TypeError, AttributeError) as exc:
        # ValueError/ToolError are expected, handler-local failures; a
        # TypeError or AttributeError here more likely means a genuine
        # handler bug (e.g. a wrong argument shape slipping past Pydantic
        # validation) — log it rather than silently masking it as a normal
        # tool-usage error.
        logger.warning("Tool %r raised %s: %s", tool_name, type(exc).__name__, exc)
        return f"Error: {exc}"


def openai_tool_specs() -> list[dict[str, Any]]:
    """Return the registry as OpenAI ``tools=[...]`` function-calling specs."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.params.model_json_schema(),
            },
        }
        for spec in TOOLS.values()
    ]
