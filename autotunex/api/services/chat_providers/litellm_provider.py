# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM chat provider — verbatim reproduction of the historical chat_service
LLM construction, including the pre-construction env guard.
"""

import logging
import os

from services.chat_providers.base import ChatProvider

logger = logging.getLogger(__name__)


class LiteLLMChatProvider(ChatProvider):
    def build_llm(self):
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("LITELLM_API_KEY")
        url = os.getenv("LITELLM_URL")
        if not api_key or not url:
            raise ValueError(
                "LITELLM_API_KEY and LITELLM_URL must be set in the .env file."
            )
        return ChatOpenAI(
            model=os.getenv("LITELLM_MODEL", "aws/claude-sonnet-4-6"),
            api_key=api_key,
            base_url=f"{url}/v1",
            max_tokens=4096,
            temperature=0,
        )
