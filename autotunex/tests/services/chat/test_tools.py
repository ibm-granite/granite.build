"""Tests for :mod:`autotunex.services.chat.tools`."""

from __future__ import annotations

import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.models.auth import Principal
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
