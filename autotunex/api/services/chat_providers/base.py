# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""ChatProvider seam contract: abstracts chat-model construction only.

The ReAct agent, MCP tool discovery, checkpointer, and system prompt in
chat_service.py are independent of which LLM object is used. A provider only
returns a LangChain chat model.
"""

from abc import ABC, abstractmethod


class ChatProvider(ABC):
    @abstractmethod
    def build_llm(self):  # -> a LangChain BaseChatModel
        ...
