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

"""Ntfy push notification adapter.

Sends build event notifications via ntfy.sh (or a self-hosted ntfy server).
Users subscribe to a topic in the ntfy app (iOS/Android) to receive push notifications.

No API key, no account required for the public ntfy.sh server.
"""

import logging

import aiohttp

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload

logger = logging.getLogger(__name__)


class NtfyAdapter(NotificationAdapter):
    """Delivers build event notifications via ntfy push service."""

    def __init__(self, topic: str, server: str = "https://ntfy.sh") -> None:
        self._topic = topic
        self._server = server.rstrip("/")

    async def deliver(self, event: BuildEvent) -> bool:
        """Send a push notification to the ntfy topic."""
        build_id = event.run_metadata.build_id or "unknown"
        event_type = event.type.value

        # Build title and message
        title = f"Build {build_id[:8]} - {event_type}"
        message = self._format_message(event)

        # Set priority based on event content
        priority = "default"
        tags = "hammer"
        if isinstance(event.payload, BuildEventStatusPayload):
            status = event.payload.status.value
            title = f"Build {build_id[:8]} - {status}"
            if status in ("failed", "error"):
                priority = "high"
                tags = "x"
            elif status == "success":
                tags = "white_check_mark"

        url = f"{self._server}/{self._topic}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=message,
                    headers={
                        "Title": title,
                        "Priority": priority,
                        "Tags": tags,
                    },
                ) as response:
                    if response.status == 200:
                        return True
                    logger.warning(
                        "[NtfyAdapter] Delivery failed: %d %s",
                        response.status,
                        await response.text(),
                    )
                    return False
        except Exception as e:
            logger.warning("[NtfyAdapter] Delivery error: %s", e)
            return False

    def _format_message(self, event: BuildEvent) -> str:
        """Format a BuildEvent into a human-readable ntfy message."""
        build_id = event.run_metadata.build_id or "unknown"
        target = event.run_metadata.target_name or ""
        lines = [f"Build: {build_id}"]
        if target:
            lines.append(f"Target: {target}")
        lines.append(f"Event: {event.type.value}")
        if isinstance(event.payload, BuildEventStatusPayload):
            lines.append(f"Status: {event.payload.status.value}")
            if event.payload.msg:
                lines.append(f"Message: {event.payload.msg}")
        return "\n".join(lines)
