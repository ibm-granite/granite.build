# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""GbLogReader selection from settings, mirroring launch/registry.py."""

from __future__ import annotations

import os

from autotunex.core.config import Settings
from autotunex.services.gb_logs.disabled_reader import DisabledGbLogReader
from autotunex.services.gb_logs.gbcli_reader import GbcliLogReader
from autotunex.services.gb_logs.protocols import GbLogReader


def get_gb_log_reader(settings: Settings) -> GbLogReader:
    """Return the gbcli reader when gb is configured, else the disabled reader.

    gb log reading needs the same signals as launching on gb: the ``llmb`` backend
    and a token in the ``gb_token_env`` var (startup validation guarantees the
    latter when ``job_backend="llmb"``; the check here is defensive).
    """
    if settings.job_backend == "llmb" and os.environ.get(settings.gb_token_env):
        return GbcliLogReader(settings.gb_token_env)
    return DisabledGbLogReader()
