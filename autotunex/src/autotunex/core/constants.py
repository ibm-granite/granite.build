# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Well-known constants shared across layers.

``core/`` is importable from anywhere (see ``CLAUDE.md``), so a value the
repository and the services must agree on lives here rather than being
duplicated in each.
"""

from __future__ import annotations

from typing import Final
from uuid import UUID

SYSTEM_USER_ID: Final[UUID] = UUID("00000000-0000-0000-0000-000000000001")
"""Owner of the shared, curated starter configurations and datasets.

Rows owned by this reserved user are readable and usable by *every* caller (the
shared tier), but writable only by the system user itself or an admin via
``scope=all``. The same UUID is hard-coded in the tuning-pipeline DB layer
(``src/api-bridge/src/api_bridge/utils.py``) and the three frontend components
that render system rows; all four must agree, which is why this is a fixed
constant, not a setting.
"""
