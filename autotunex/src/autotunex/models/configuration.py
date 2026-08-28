# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Configuration schemas.

A *configuration* is a named, reusable set of tuning settings, stored in the
schema-less ``configurations.config_data`` JSON column. Unlike jobs,
configurations are the one resource this API creates, updates and deletes — see
``CLAUDE.md`` open decision 6 for why job submission stayed retired while this
did not.

``config_data`` is *not* validated against
:mod:`autotunex.models.search_space` — that toy ``SearchSpace`` shape belongs to
the unbuilt search layer and does not match the far richer structure the tuning
pipeline actually writes (``tune_config`` / ``tuners_config`` /
``training_config`` / ``tuners_rl_config`` / ``training_rl_config``). The API
requires only that it be a non-empty JSON object.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from autotunex.models.status import RunStatus


class ConfigurationCreate(BaseModel):
    """The request body for creating or fully replacing a configuration.

    Reused for both ``POST`` and ``PUT``: a ``PUT`` is a full replacement, so it
    carries the same representation as a create — there is no partial update
    (``PATCH`` is deliberately not offered).

    ``config_data`` is typed as a bare JSON object; the service enforces only
    that it is non-empty (a domain 422,
    :class:`~autotunex.core.exceptions.InvalidConfigDataError`, rather than a
    stored no-op). ``user_id`` is deliberately absent: ownership is taken from
    the calling principal, never from the request, so a caller cannot create a
    configuration on another user's behalf.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    tuner_type: str | None = Field(default=None, max_length=50)
    rl_tuner_type: str | None = Field(default=None, max_length=50)
    config_data: dict[str, Any] = Field(
        description="Tuning settings as a non-empty JSON object; shape is not enforced."
    )


class ConfigurationJobRef(BaseModel):
    """A compact reference to a job ("tuning") launched from this configuration.

    Mirrors :class:`~autotunex.models.dataset.DatasetJobRef`: a full
    :class:`~autotunex.models.job.JobRead` would nest every tuning's trials and
    tasks under each configuration. Scoped to the caller's own jobs in the
    service (an admin with ``scope=all`` sees every referencing job). The UX
    renders one pill per entry, labelled by ``experiment_name``, and opens the
    tuning by ``id`` on click.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_name: str | None = None
    status: RunStatus


class ConfigurationRead(BaseModel):
    """A configuration as returned by every configuration endpoint.

    Reused for both the list and the detail response: a configuration has no
    heavy nested collections (unlike a job's trials and tasks), so there is
    nothing a compact ``Summary`` would usefully drop.

    ``config_data`` is ``| None`` on read even though it is required on write:
    the live database predates these endpoints and may hold rows with a null
    ``config_data``, and reading must not choke on them.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str = Field(description="Owner's id, from configurations.user_id.")
    name: str
    tuner_type: str | None = None
    rl_tuner_type: str | None = None
    config_data: dict[str, Any] | None = None
    associated_jobs: list[ConfigurationJobRef] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
