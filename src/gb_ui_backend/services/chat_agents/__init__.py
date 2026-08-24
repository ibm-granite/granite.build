"""Factory for selecting a ChatAgentBackend implementation.

get_backend() is process-wide cached (mirrors config.get_config()) so the same
backend instance — and therefore the same in-memory session registry — is
reused across requests. api/chat.py must call get_backend(), not construct a
backend directly, or every request would spawn a fresh gbmcp subprocess and
multi-turn conversations would silently lose context.
"""

from __future__ import annotations

from functools import lru_cache

from gb_ui_backend.config import get_config
from gb_ui_backend.services.chat_agents.base import ChatAgentBackend


@lru_cache
def get_backend() -> ChatAgentBackend:
    config = get_config()
    if config.chat_backend == "tool_loop":
        from gb_ui_backend.services.chat_agents.tool_loop_backend import ToolLoopBackend

        return ToolLoopBackend(config)
    raise ValueError(f"Unknown GB_UI_CHAT_BACKEND: {config.chat_backend!r}")
