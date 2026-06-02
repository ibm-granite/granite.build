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

"""Email notification adapter.

Sends build event notifications via SMTP. Works with any email provider.
"""

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload

logger = logging.getLogger(__name__)


class EmailAdapter(NotificationAdapter):
    """Delivers build event notifications via email (SMTP)."""

    def __init__(
        self,
        to: str,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_addr: str = "",
        use_tls: bool = True,
    ) -> None:
        self._to = to
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._from_addr = from_addr or f"gbserver@{smtp_host}"
        self._use_tls = use_tls

    async def deliver(self, event: BuildEvent) -> bool:
        """Send an email notification for the build event."""
        subject = self._build_subject(event)
        body = self._format_message(event)

        try:
            msg = MIMEMultipart()
            msg["From"] = self._from_addr
            msg["To"] = self._to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Run SMTP in thread to avoid blocking the event loop
            await asyncio.to_thread(self._send_smtp, msg)
            return True
        except Exception as e:
            logger.warning("[EmailAdapter] Delivery error: %s", e)
            return False

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Synchronous SMTP send (called via asyncio.to_thread)."""
        if self._use_tls:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                if self._smtp_user:
                    server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._smtp_user:
                    server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)

    def _build_subject(self, event: BuildEvent) -> str:
        build_id = event.run_metadata.build_id or "unknown"
        if isinstance(event.payload, BuildEventStatusPayload):
            return f"[gbserver] Build {build_id[:8]} - {event.payload.status.value}"
        return f"[gbserver] Build {build_id[:8]} - {event.type.value}"

    def _format_message(self, event: BuildEvent) -> str:
        build_id = event.run_metadata.build_id or "unknown"
        target = event.run_metadata.target_name or ""
        lines = [
            f"Build ID: {build_id}",
            f"Event: {event.type.value}",
        ]
        if target:
            lines.append(f"Target: {target}")
        if isinstance(event.payload, BuildEventStatusPayload):
            lines.append(f"Status: {event.payload.status.value}")
            if event.payload.msg:
                lines.append(f"Message: {event.payload.msg}")
        lines.append("")
        lines.append("-- gbserver standalone notification")
        return "\n".join(lines)
