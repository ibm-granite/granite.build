# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""
MCP Server for AutoTuneX — exposes the tuning platform as MCP tools.

Mount on FastAPI via:
    from mcp_server import mcp
    app.mount("/mcp", mcp.http_app(path="/mcp"))
"""

import asyncio
import logging

from fastmcp import FastMCP
from starlette.background import BackgroundTasks

import models as api
from services import (
    db_service,
    job_service,
    config_service,
    dataset_service,
    user_service,
)
from services.plugins import Seam, resolve

logger = logging.getLogger("mcp_server")

mcp = FastMCP("AutoTuneX")

# Max items returned by list tools (keeps context window lean for the chat agent)
_LIST_LIMIT = 50


# ---------------------------------------------------------------------------
# Service helpers — mirror dependencies.py without FastAPI Depends()
# ---------------------------------------------------------------------------


def _get_services():
    """Create a shared DB connection and all service instances for one tool call."""
    db = db_service.Database()
    return {
        "job": job_service.Job(db),
        "config": config_service.Config(db),
        "dataset": dataset_service.Dataset(db),
        "dmf": resolve(Seam.REGISTRY, db=db),
        "user": user_service.User(db),
    }


async def _resolve_user_id(svc: dict, email: str) -> str:
    """Resolve user_email → user_id, raising a clear error if not found."""
    user = await svc["user"].get_user(email)
    if not user:
        raise ValueError(f"User not found: {email}")
    return user["id"]


def _pick(obj, keys: list[str]) -> dict:
    """Extract only the specified keys from a dict or Pydantic model."""
    d = (
        obj.model_dump(mode="json")
        if hasattr(obj, "model_dump")
        else (obj if isinstance(obj, dict) else {})
    )
    return {k: d[k] for k in keys if k in d}


# ---------------------------------------------------------------------------
# Job tools
# ---------------------------------------------------------------------------

_JOB_SUMMARY = [
    "id",
    "experiment_name",
    "status",
    "model",
    "model_source",
    "created_at",
    "updated_at",
]
_JOB_DETAIL = [
    *_JOB_SUMMARY,
    "config_id",
    "dataset_id",
    "seed",
    "tuning_type",
    "num_trials",
    "config_name",
]
_TRIAL_SUMMARY = ["id", "job_id", "status"]
_LOG_KEYS = ["id", "trial_id", "level", "message", "iteration", "epoch", "timestamp"]
_RESULT_KEYS = ["id", "trial_id", "metric", "loss", "eval_loss", "total_time"]


@mcp.tool
async def list_jobs(user_email: str) -> str:
    """List fine-tuning jobs for a user (most recent 50).

    Returns a pre-formatted text listing — present it to the user as-is.
    """
    logger.info("[MCP TOOL] list_jobs(user_email=%s)", user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    jobs = await svc["job"].get_jobs(user_id)
    logger.info("[MCP TOOL] list_jobs: user_id=%s raw=%d", user_id, len(jobs))
    items = [_pick(j, _JOB_SUMMARY) for j in jobs[:_LIST_LIMIT]]
    if not items:
        return "No fine-tuning jobs found for this user."
    lines = [f"**{len(items)} job(s):**\n"]
    for i, j in enumerate(items, 1):
        name = j.get("experiment_name", "?")
        status = j.get("status", "?")
        model = j.get("model", "?")
        lines.append(
            f"{i}. **{name}** (id: `{j.get('id', '?')}`) — status: `{status}`, model: `{model}`"
        )
    return "\n".join(lines)


@mcp.tool
async def get_job(job_id: str, user_email: str) -> dict:
    """Get details about a fine-tuning job (without logs — use get_trial_logs for those)."""
    logger.info("[MCP TOOL] get_job(job_id=%s, user_email=%s)", job_id, user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    job = await svc["job"].get_job(job_id, user_id)
    summary = _pick(job, _JOB_DETAIL)
    logs = job.logs if hasattr(job, "logs") and job.logs else []
    summary["log_count"] = len(logs)
    return summary


@mcp.tool
async def get_job_trials(job_id: str) -> list:
    """Get trials for a fine-tuning job (summary, max 50)."""
    logger.info("[MCP TOOL] get_job_trials(job_id=%s)", job_id)
    svc = _get_services()
    trials = await svc["job"].get_trials_by_job_id(job_id)
    return [_pick(t, _TRIAL_SUMMARY) for t in trials[:_LIST_LIMIT]]


@mcp.tool
async def get_trial_logs(trial_id: str) -> list:
    """Get the last 30 training log entries for a trial."""
    logger.info("[MCP TOOL] get_trial_logs(trial_id=%s)", trial_id)
    svc = _get_services()
    logs = await svc["job"].get_trial_logs_by_id(trial_id)
    return [_pick(entry, _LOG_KEYS) for entry in logs[-30:]]


@mcp.tool
async def get_job_results(job_id: str) -> list:
    """Get results for a fine-tuning job (max 50 trials, core metrics only)."""
    logger.info("[MCP TOOL] get_job_results(job_id=%s)", job_id)
    svc = _get_services()
    results = await svc["job"].get_results_by_job_id(job_id)
    return [_pick(r, _RESULT_KEYS) for r in results[:_LIST_LIMIT]]


@mcp.tool
async def get_job_assets(job_id: str) -> list:
    """List downloadable result files (model checkpoints, metrics, etc.) for a job."""
    logger.info("[MCP TOOL] get_job_assets(job_id=%s)", job_id)
    svc = _get_services()
    return await svc["job"].list_job_assets(job_id)


@mcp.tool
async def start_tuning_job(
    config_id: str,
    dataset_id: str,
    model: str,
    experiment_name: str,
    user_email: str,
    model_source: str = "huggingface",
    seed: int = 42,
) -> dict:
    """Start a new fine-tuning job.

    Args:
        config_id: ID of the hyperparameter configuration to use.
        dataset_id: ID of the training dataset.
        model: Model identifier (e.g. 'meta-llama/Llama-2-7b-hf').
        experiment_name: Unique name for this experiment.
        user_email: Email of the user starting the job.
        model_source: 'huggingface' or 'dmf'. Defaults to 'huggingface'.
        seed: Random seed for reproducibility. Defaults to 42.
    """
    logger.info(
        "[MCP TOOL] start_tuning_job(config_id=%s, dataset_id=%s, model=%s, user_email=%s)",
        config_id,
        dataset_id,
        model,
        user_email,
    )
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)

    run_config = api.TuningConfig(
        config_id=config_id,
        dataset_id=dataset_id,
        model=model,
        experiment_name=experiment_name,
        model_source=api.ModelSource(model_source),
        seed=seed,
    )
    run_config.user_id = user_id

    bg = BackgroundTasks()
    await svc["job"].start(run_config, bg)

    # Execute the queued background task (runner.run) asynchronously
    for task in bg.tasks:
        if asyncio.iscoroutinefunction(task.func):
            asyncio.create_task(task.func(*task.args, **task.kwargs))
        else:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, task.func, *task.args, **task.kwargs)

    return {"status": "started", "experiment_name": experiment_name}


# ---------------------------------------------------------------------------
# Configuration tools
# ---------------------------------------------------------------------------

_CONFIG_SUMMARY = ["id", "name", "tuner_type", "rl_tuner_type"]
_CONFIG_DETAIL = [*_CONFIG_SUMMARY, "config_data"]


@mcp.tool
async def list_configs(user_email: str) -> str:
    """List hyperparameter configurations for a user (max 50).

    Returns a pre-formatted text listing — present it to the user as-is.
    """
    logger.info("[MCP TOOL] list_configs(user_email=%s)", user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    configs = await svc["config"].get_configs(user_id)
    logger.info("[MCP TOOL] list_configs: user_id=%s raw=%d", user_id, len(configs))
    items = [_pick(c, _CONFIG_SUMMARY) for c in configs[:_LIST_LIMIT]]
    if not items:
        return "No configurations found for this user."
    lines = [f"**{len(items)} configuration(s):**\n"]
    for i, c in enumerate(items, 1):
        tuner = c.get("tuner_type", "n/a")
        rl = c.get("rl_tuner_type")
        tuner_str = f"`{tuner}`" + (f" / `{rl}`" if rl else "")
        lines.append(
            f"{i}. **{c.get('name', '?')}** (id: `{c.get('id', '?')}`) — tuner: {tuner_str}"
        )
    return "\n".join(lines)


@mcp.tool
async def get_config(config_id: str, user_email: str) -> dict:
    """Get a specific hyperparameter configuration (without associated jobs)."""
    logger.info(
        "[MCP TOOL] get_config(config_id=%s, user_email=%s)", config_id, user_email
    )
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    cfg = await svc["config"].get_config(config_id, user_id)
    return _pick(cfg, _CONFIG_DETAIL)


@mcp.tool
def get_config_template() -> dict:
    """Get the default AutoTuneX configuration template showing all supported hyperparameters."""
    logger.info("[MCP TOOL] get_config_template()")
    svc = _get_services()
    return svc["config"].get_config_for_ui()


@mcp.tool
async def create_config(
    name: str,
    config_data: dict,
    user_email: str,
    tuner_type: str = "bayesian",
) -> dict:
    """Create a new hyperparameter configuration.

    Args:
        name: Name for the configuration.
        config_data: Hyperparameter search space definition (dict).
        user_email: Email of the user creating the config.
        tuner_type: HPO algorithm — 'bayesian', 'grid_search', or 'random_search'.
    """
    logger.info("[MCP TOOL] create_config(name=%s, user_email=%s)", name, user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    config_obj = api.Configuration(
        name=name,
        tuner_type=tuner_type,
        config_data=config_data,
        user_id=user_id,
    )
    result = await svc["config"].push_config(config_obj)
    return _pick(result, ["id", "name"])


# ---------------------------------------------------------------------------
# Dataset tools
# ---------------------------------------------------------------------------

_DATASET_SUMMARY = [
    "id",
    "name",
    "description",
    "train_records",
    "validation_records",
    "created_at",
]
_DATASET_DETAIL = [
    *_DATASET_SUMMARY,
    "train_file",
    "train_file_size",
    "validation_file",
    "validation_file_size",
    "updated_at",
]


@mcp.tool
async def list_datasets(user_email: str) -> str:
    """List datasets for a user (max 50).

    Returns a pre-formatted text listing — present it to the user as-is.
    """
    logger.info("[MCP TOOL] list_datasets(user_email=%s)", user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    datasets = await svc["dataset"].get_datasets(user_id)
    logger.info("[MCP TOOL] list_datasets: user_id=%s raw=%d", user_id, len(datasets))
    items = [_pick(d, _DATASET_SUMMARY) for d in datasets[:_LIST_LIMIT]]
    if not items:
        return "No datasets found for this user."
    lines = [f"**{len(items)} dataset(s):**\n"]
    for i, d in enumerate(items, 1):
        train = d.get("train_records", "?")
        val = d.get("validation_records", "?")
        lines.append(
            f"{i}. **{d.get('name', '?')}** (id: `{d.get('id', '?')}`) — train: {train}, val: {val}"
        )
    return "\n".join(lines)


@mcp.tool
async def get_dataset(dataset_id: str, user_email: str) -> dict:
    """Get details about a specific dataset (without data preview)."""
    logger.info(
        "[MCP TOOL] get_dataset(dataset_id=%s, user_email=%s)", dataset_id, user_email
    )
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    ds = await svc["dataset"].get_dataset(dataset_id, user_id)
    return _pick(ds, _DATASET_DETAIL)


@mcp.tool
def get_supported_dataset_types() -> dict:
    """Get the list of supported dataset file types and formats."""
    logger.info("[MCP TOOL] get_supported_dataset_types()")
    svc = _get_services()
    return svc["dataset"].get_autotune_dataset_types()


# ---------------------------------------------------------------------------
# DMF (Model Registry) tools
# ---------------------------------------------------------------------------

_MODEL_SUMMARY = ["model_id", "model_label", "base_model", "size", "created_at"]


@mcp.tool
async def list_published_models(user_email: str) -> list:
    """List models published to DMF (summary only, max 50)."""
    logger.info("[MCP TOOL] list_published_models(user_email=%s)", user_email)
    svc = _get_services()
    user_id = await _resolve_user_id(svc, user_email)
    models = await svc["dmf"].get_models(user_id)
    return [_pick(m, _MODEL_SUMMARY) for m in models[:_LIST_LIMIT]]


# ---------------------------------------------------------------------------
# User tools
# ---------------------------------------------------------------------------

_USER_KEYS = ["id", "email", "role", "created_at"]


@mcp.tool
async def get_user_info(user_email: str) -> dict:
    """Get user profile information."""
    logger.info("[MCP TOOL] get_user_info(user_email=%s)", user_email)
    svc = _get_services()
    user = await svc["user"].get_user(user_email)
    return _pick(user, _USER_KEYS)


@mcp.tool
async def get_user_metadata(user_email: str) -> dict:
    """Get user statistics — number of jobs, configs, datasets, etc."""
    logger.info("[MCP TOOL] get_user_metadata(user_email=%s)", user_email)
    svc = _get_services()
    user = await svc["user"].get_user(user_email)
    if not user:
        raise ValueError(f"User not found: {user_email}")
    user_id = user["id"] if isinstance(user, dict) else user.id
    metadata = await svc["user"].get_user_metadata(str(user_id))
    return (
        metadata.model_dump(mode="json")
        if hasattr(metadata, "model_dump")
        else metadata
    )
