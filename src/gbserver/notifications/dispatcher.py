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

"""Standalone notification dispatcher."""

import logging
from typing import Any, Dict, List, Optional

from gbserver.notifications.adapter import NotificationAdapter
from gbserver.notifications.config import load_notification_config
from gbserver.notifications.email_adapter import EmailAdapter
from gbserver.notifications.macos_adapter import MacOSAdapter
from gbserver.notifications.ntfy_adapter import NtfyAdapter
from gbserver.notifications.telegram_adapter import TelegramAdapter
from gbserver.types.buildevent import BuildEvent

logger = logging.getLogger(__name__)


class StandaloneDispatcher:
    """Dispatches build events to configured notification adapters.

    On initialization, loads the notification configuration and creates
    adapter instances for each configured notification entry.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        config_entries = load_notification_config(config_path)
        self._adapters: List[_AdapterEntry] = []

        for entry in config_entries:
            adapter = self._create_adapter(entry)
            if adapter is not None:
                events_filter = entry.get("events", ["*"])
                self._adapters.append(_AdapterEntry(adapter=adapter, events=events_filter))

    def _create_adapter(self, entry: Dict[str, Any]) -> Optional[NotificationAdapter]:
        """Create an adapter instance based on the entry's 'type' field."""
        adapter_type = entry.get("type")

        if adapter_type == "telegram":
            bot_token = entry.get("bot_token")
            chat_id = entry.get("chat_id")
            if not bot_token or not chat_id:
                logger.warning(
                    "Telegram adapter config missing bot_token or chat_id, skipping"
                )
                return None
            return TelegramAdapter(bot_token=bot_token, chat_id=chat_id)

        if adapter_type == "ntfy":
            topic = entry.get("topic", "")
            server = entry.get("server", "https://ntfy.sh")
            if not topic:
                logger.warning("Ntfy adapter config missing topic, skipping")
                return None
            return NtfyAdapter(topic=topic, server=server)

        if adapter_type == "email":
            to = entry.get("to", "")
            if not to:
                logger.warning("Email adapter config missing 'to' address, skipping")
                return None
            return EmailAdapter(
                to=to,
                smtp_host=entry.get("smtp_host", "localhost"),
                smtp_port=int(entry.get("smtp_port", 587)),
                smtp_user=entry.get("smtp_user", ""),
                smtp_password=entry.get("smtp_password", ""),
                from_addr=entry.get("from_addr", ""),
                use_tls=entry.get("use_tls", True),
            )

        if adapter_type == "macos":
            sound = entry.get("sound", "default")
            return MacOSAdapter(sound=sound)

        logger.warning("Unknown notification adapter type: %s", adapter_type)
        return None

    async def dispatch(self, event: BuildEvent) -> None:
        """Dispatch a build event to all matching adapters.

        For each configured adapter, checks if the event type matches the
        adapter's event filter. If matched, calls adapter.deliver(event).
        Exceptions from individual adapters are caught and logged.
        """
        event_type_value = event.type.value

        for adapter_entry in self._adapters:
            if not self._matches_filter(event_type_value, adapter_entry.events):
                continue

            try:
                await adapter_entry.adapter.deliver(event)
            except Exception as exc:
                logger.warning(
                    "Adapter %s failed to deliver event %s: %s",
                    type(adapter_entry.adapter).__name__,
                    event_type_value,
                    exc,
                )

    @staticmethod
    def _matches_filter(event_type_value: str, events: List[str]) -> bool:
        """Check if an event type matches the adapter's event filter list."""
        return "*" in events or event_type_value in events


class _AdapterEntry:
    """Internal holder for an adapter instance and its event filter."""

    def __init__(self, adapter: NotificationAdapter, events: List[str]) -> None:
        self.adapter = adapter
        self.events = events
