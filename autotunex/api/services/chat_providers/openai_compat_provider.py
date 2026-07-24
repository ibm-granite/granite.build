# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat provider — for Ollama / LM Studio / OpenAI / any
endpoint speaking the OpenAI API. Selected by AUTOTUNEX_CHAT=openai_compatible
or by setting OPENAI_BASE_URL (fallback probe).
"""

import logging
import os

from services.chat_providers.base import ChatProvider

logger = logging.getLogger(__name__)


class OpenAICompatChatProvider(ChatProvider):
    def build_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv(
                "OPENAI_API_KEY", "not-needed"
            ),  # local servers ignore it
            base_url=os.getenv("OPENAI_BASE_URL"),
            max_tokens=4096,
            temperature=0,
        )
