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

"""macOS notification adapter.

Sends build event notifications via macOS Notification Center using osascript.
Notifications can forward to iPhone if Handoff is enabled.

Only works on macOS.
"""

import asyncio
import logging
import platform

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload

logger = logging.getLogger(__name__)


class MacOSAdapter(NotificationAdapter):
    """Delivers build event notifications via macOS Notification Center."""

    def __init__(self, sound: str = "default") -> None:
        self._sound = sound

    async def deliver(self, event: BuildEvent) -> bool:
        """Display a macOS notification for the build event."""
        if platform.system() != "Darwin":
            logger.warning("[MacOSAdapter] Not running on macOS, skipping")
            return False

        title = self._build_title(event)
        message = self._format_message(event)

        # Escape for AppleScript
        title_escaped = title.replace('"', '\\"')
        message_escaped = message.replace('"', '\\"')

        script = (
            f'display notification "{message_escaped}" '
            f'with title "{title_escaped}" '
            f'sound name "{self._sound}"'
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True
            logger.warning(
                "[MacOSAdapter] osascript failed: %s",
                stderr.decode().strip(),
            )
            return False
        except Exception as e:
            logger.warning("[MacOSAdapter] Delivery error: %s", e)
            return False

    def _build_title(self, event: BuildEvent) -> str:
        build_id = event.run_metadata.build_id or "unknown"
        if isinstance(event.payload, BuildEventStatusPayload):
            return f"Build {build_id[:8]} - {event.payload.status.value}"
        return f"Build {build_id[:8]} - {event.type.value}"

    def _format_message(self, event: BuildEvent) -> str:
        target = event.run_metadata.target_name or ""
        lines = []
        if target:
            lines.append(f"Target: {target}")
        if isinstance(event.payload, BuildEventStatusPayload):
            if event.payload.msg:
                lines.append(event.payload.msg)
        return " | ".join(lines) if lines else event.type.value
