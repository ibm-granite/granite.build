#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Telegram Bot API notification adapter."""

import logging

import aiohttp

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload

logger = logging.getLogger(__name__)


class TelegramAdapter(NotificationAdapter):
    """Delivers build event notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def _format_message(self, event: BuildEvent) -> str:
        """Format a BuildEvent into a human-readable Telegram message."""
        build_id = event.run_metadata.build_id or "unknown"
        target_name = event.run_metadata.target_name or "unknown"
        event_type = event.type.value

        lines = [
            f"*Build Event* `{build_id[:8]}`",
            f"Target: `{target_name}`",
            f"Event: `{event_type}`",
        ]

        if isinstance(event.payload, BuildEventStatusPayload):
            lines.append(f"Status: *{event.payload.status.value}*")
            lines.append(f"Message: {event.payload.msg}")

        return "\n".join(lines)

    async def deliver(self, event: BuildEvent) -> bool:
        """Deliver a build event notification via Telegram.

        Returns True if delivery succeeded, False otherwise.
        """
        try:
            message = self._format_message(event)
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    logger.warning(
                        "Telegram API returned status %d for build %s",
                        response.status,
                        event.run_metadata.build_id,
                    )
                    return False
        except Exception as exc:
            logger.warning(
                "Failed to deliver Telegram notification for build %s: %s",
                event.run_metadata.build_id,
                exc,
            )
            return False
