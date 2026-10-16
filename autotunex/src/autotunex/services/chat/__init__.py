# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The chat assistant: a native tool-calling agent over the shared tool registry.

The tool registry (:mod:`autotunex.services.chat.tools`) is the single source of
truth for the operations both the in-app chat agent and the opt-in MCP server
(:mod:`autotunex.api.mcp`) expose. Nothing in this package imports ``fastapi`` or
``fastmcp`` except where explicitly noted.
"""

from __future__ import annotations
