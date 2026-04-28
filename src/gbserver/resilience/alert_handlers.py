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

"""
Alert handlers for node health monitoring.

Provides pluggable alert delivery mechanisms for notifying administrators
when nodes exceed failure thresholds.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Self

import aiohttp

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class NodeHealthAlert:
    """
    Represents an alert about a problematic node.

    Attributes:
        node_name: Name of the Kubernetes node
        failure_count: Number of failures in the alert window
        threshold: The threshold that was exceeded
        window_minutes: The time window for counting failures
        failures: Recent failure events (last 5)
        timestamp: When the alert was triggered
        namespace: Kubernetes namespace where the failures occurred
        cluster: Kubernetes cluster context where the failures occurred
    """

    def __init__(
        self: Self,
        node_name: str,
        failure_count: int,
        threshold: int,
        window_minutes: int,
        failures: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
        namespace: str = "",
        cluster: str = "",
    ) -> None:
        self.node_name = node_name
        self.failure_count = failure_count
        self.threshold = threshold
        self.window_minutes = window_minutes
        self.failures = failures
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.namespace = namespace
        self.cluster = cluster

    def to_dict(self: Self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        cordon_cmd = "kubectl cordon"
        if self.cluster:
            cordon_cmd += f" --context {self.cluster}"
        if self.namespace:
            cordon_cmd += f" -n {self.namespace}"
        cordon_cmd += f" {self.node_name}"

        return {
            "alert_type": "node_health_threshold_exceeded",
            "node_name": self.node_name,
            "namespace": self.namespace,
            "cluster": self.cluster,
            "failure_count": self.failure_count,
            "threshold": self.threshold,
            "window_minutes": self.window_minutes,
            "recent_failures": self.failures,
            "recommended_action": cordon_cmd,
            "timestamp": self.timestamp.isoformat(),
            "severity": "warning",
        }


class AlertHandler(ABC):
    """
    Abstract base class for alert handlers.

    Subclasses implement specific delivery mechanisms (webhook, Slack, email, etc.).
    """

    @abstractmethod
    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """
        Send an alert about a problematic node.

        Args:
            alert: The alert to send

        Returns:
            bool: True if alert was sent successfully
        """


class LoggingAlertHandler(AlertHandler):
    """
    Alert handler that logs alerts.

    Useful for testing or when external alerting is handled by log aggregation.
    """

    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """Log the alert at ERROR level."""
        logger.error(
            "[ALERT] Node %s exceeded failure threshold: %d failures in %d minutes. "
            "Recommended action: kubectl cordon %s",
            alert.node_name,
            alert.failure_count,
            alert.window_minutes,
            alert.node_name,
        )
        return True


class WebhookAlertHandler(AlertHandler):
    """
    Alert handler that sends alerts to a webhook URL.

    Works with generic webhooks, Slack incoming webhooks, Microsoft Teams,
    PagerDuty, OpsGenie, and other webhook-based systems.

    Parameters
    ----------
    webhook_url : str
        The URL to POST alerts to
    headers : Optional[Dict[str, str]]
        Additional headers to include (e.g., authorization)
    timeout_seconds : int
        Request timeout in seconds (default: 10)
    """

    def __init__(
        self: Self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 10,
    ) -> None:
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """Send alert to webhook URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        **self.headers,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    if response.status >= 400:
                        logger.error(
                            "[WebhookAlertHandler] Failed to send alert: HTTP %s - %s",
                            response.status,
                            await response.text(),
                        )
                        return False

                    logger.info(
                        "[WebhookAlertHandler] Alert sent for node %s",
                        alert.node_name,
                    )
                    return True

        except aiohttp.ClientError as e:
            logger.error("[WebhookAlertHandler] Network error sending alert: %s", e)
            return False
        except Exception as e:
            logger.error("[WebhookAlertHandler] Unexpected error sending alert: %s", e)
            return False


class SlackAlertHandler(AlertHandler):
    """
    Alert handler that sends formatted messages to Slack.

    Uses Slack's Block Kit for rich formatting with actionable information.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL
    channel : Optional[str]
        Override channel (if webhook allows)
    mention_users : Optional[List[str]]
        Slack user IDs to mention (e.g., ["U1234567890"])
    """

    def __init__(
        self: Self,
        webhook_url: str,
        channel: Optional[str] = None,
        mention_users: Optional[List[str]] = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.channel = channel
        self.mention_users = mention_users or []

    def _format_slack_message(self: Self, alert: NodeHealthAlert) -> Dict[str, Any]:
        """Format alert as Slack Block Kit message."""
        # Build mention string
        mentions = " ".join(f"<@{user}>" for user in self.mention_users)
        mention_text = f"{mentions} " if mentions else ""

        # Build failure type summary
        failure_types: Dict[str, int] = {}
        for failure in alert.failures:
            ft = failure.get("failure_type", "Unknown")
            failure_types[ft] = failure_types.get(ft, 0) + 1
        failure_summary = ", ".join(f"{k}: {v}" for k, v in failure_types.items())

        message: Dict[str, Any] = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Node Health Alert: {alert.node_name}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{mention_text}Node *{alert.node_name}* has exceeded "
                            f"the failure threshold.\n\n"
                            f"*Failures:* {alert.failure_count} in {alert.window_minutes} minutes "
                            f"(threshold: {alert.threshold})\n"
                            f"*Failure types:* {failure_summary}"
                        ),
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Recommended Action:*\n"
                            f"```kubectl cordon {alert.node_name}```"
                        ),
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Alert generated at {alert.timestamp.isoformat()}",
                        }
                    ],
                },
            ]
        }

        if self.channel:
            message["channel"] = self.channel

        return message

    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """Send formatted alert to Slack."""
        try:
            message = self._format_slack_message(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=message,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status >= 400:
                        logger.error(
                            "[SlackAlertHandler] Failed to send alert: HTTP %s - %s",
                            response.status,
                            await response.text(),
                        )
                        return False

                    logger.info(
                        "[SlackAlertHandler] Alert sent for node %s",
                        alert.node_name,
                    )
                    return True

        except Exception as e:
            logger.error("[SlackAlertHandler] Error sending alert: %s", e)
            return False


class RetryableAlertHandler(AlertHandler):
    """
    Alert handler wrapper that adds retry logic with exponential backoff.

    Implements:
    - Retry with exponential backoff
    - Dead letter queue for failed alerts
    - Circuit breaker pattern

    Parameters
    ----------
    handler : AlertHandler
        The underlying alert handler to wrap
    max_retries : int
        Maximum number of retry attempts (default: 3)
    initial_backoff : float
        Initial backoff in seconds (default: 1.0)
    max_backoff : float
        Maximum backoff in seconds (default: 60.0)
    dlq_max_size : int
        Maximum size of dead letter queue (default: 100)
    """

    def __init__(
        self: Self,
        handler: AlertHandler,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        dlq_max_size: int = 100,
    ) -> None:
        self.handler = handler
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.dlq_max_size = dlq_max_size

        # Dead letter queue for failed alerts
        self._dlq: Deque[Dict[str, Any]] = deque(maxlen=dlq_max_size)

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None

    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """Send alert with retry and circuit breaker."""
        # Check circuit breaker
        if self._circuit_open and self._circuit_open_until is not None:
            if datetime.now(timezone.utc) < self._circuit_open_until:
                logger.warning(
                    "[RetryableAlertHandler] Circuit breaker open, moving alert to DLQ"
                )
                self._add_to_dlq(alert, "circuit_breaker_open")
                return False

            # Circuit breaker timeout expired, reset it
            # Try to close circuit
            logger.info("[RetryableAlertHandler] Attempting to close circuit")
            self._circuit_open = False
            self._failure_count = 0

        # Attempt send with retries
        for attempt in range(self.max_retries + 1):
            try:
                success = await self.handler.send_alert(alert)
                if success:
                    # Reset failure count on success
                    self._failure_count = 0
                    logger.info(
                        "[RetryableAlertHandler] Alert sent successfully (attempt %d/%d)",
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    return True

                # Handler returned False (non-exception failure)
                if attempt < self.max_retries:
                    backoff = min(
                        self.initial_backoff * (2**attempt),
                        self.max_backoff,
                    )
                    logger.warning(
                        "[RetryableAlertHandler] Alert send returned False, "
                        "retrying in %.2f seconds (attempt %d/%d)",
                        backoff,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(
                    "[RetryableAlertHandler] Alert send failed with exception: %s "
                    "(attempt %d/%d)",
                    e,
                    attempt + 1,
                    self.max_retries + 1,
                )

                if attempt < self.max_retries:
                    backoff = min(
                        self.initial_backoff * (2**attempt),
                        self.max_backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    # Track failures for circuit breaker
                    self._failure_count += 1
                    if self._failure_count >= 5:
                        self._open_circuit()

        # All retries exhausted
        logger.error(
            "[RetryableAlertHandler] All retry attempts exhausted for alert %s",
            alert.node_name,
        )
        self._add_to_dlq(alert, "max_retries_exceeded")
        return False

    def _open_circuit(self: Self) -> None:
        """Open circuit breaker for 5 minutes."""
        self._circuit_open = True
        self._circuit_open_until = datetime.now(timezone.utc) + timedelta(minutes=5)
        logger.error(
            "[RetryableAlertHandler] Circuit breaker opened until %s",
            self._circuit_open_until.isoformat(),
        )

    def _add_to_dlq(self: Self, alert: NodeHealthAlert, reason: str) -> None:
        """Add failed alert to dead letter queue."""
        dlq_entry = {
            "alert": alert.to_dict(),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._dlq.append(dlq_entry)
        logger.error(
            "[RetryableAlertHandler] Added alert to DLQ (reason: %s). DLQ size: %d/%d",
            reason,
            len(self._dlq),
            self.dlq_max_size,
        )

    def get_dlq_entries(self: Self) -> List[Dict[str, Any]]:
        """Get all entries in the dead letter queue."""
        return list(self._dlq)

    def clear_dlq(self: Self) -> int:
        """Clear the dead letter queue and return number of entries cleared."""
        count = len(self._dlq)
        self._dlq.clear()
        logger.info("[RetryableAlertHandler] Cleared %d entries from DLQ", count)
        return count


class CompositeAlertHandler(AlertHandler):
    """
    Alert handler that sends alerts to multiple handlers.

    Useful for sending to both Slack and a logging system, for example.

    Parameters
    ----------
    handlers : List[AlertHandler]
        List of handlers to send alerts to
    require_all : bool
        If True, returns success only if all handlers succeed.
        If False, returns success if any handler succeeds.
    """

    def __init__(
        self: Self,
        handlers: List[AlertHandler],
        require_all: bool = False,
    ) -> None:
        self.handlers = handlers
        self.require_all = require_all

    async def send_alert(self: Self, alert: NodeHealthAlert) -> bool:
        """Send alert to all configured handlers."""
        if not self.handlers:
            return True

        results = []
        for handler in self.handlers:
            try:
                result = await handler.send_alert(alert)
                results.append(result)
            except Exception as e:
                logger.error(
                    "[CompositeAlertHandler] Handler %s failed: %s",
                    handler.__class__.__name__,
                    e,
                )
                results.append(False)

        if self.require_all:
            return all(results)
        return any(results)


def create_alert_handler_from_env() -> AlertHandler:
    """
    Create an alert handler based on environment variables.

    Implements graceful degradation - if external handlers fail to initialize,
    falls back to logging-only handler.

    Environment variables:
        GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL: Generic webhook URL
        GBSERVER_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL: Slack webhook URL
        GBSERVER_NODE_HEALTH_ALERT_SLACK_CHANNEL: Optional Slack channel override
        GBSERVER_NODE_HEALTH_ALERT_SLACK_MENTION_USERS: Comma-separated Slack user IDs

    Returns:
        AlertHandler (never None, at minimum returns LoggingAlertHandler)
    """
    handlers: List[AlertHandler] = []

    # Always add logging handler as fallback
    handlers.append(LoggingAlertHandler())

    from gbserver.types.constants import (
        ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_CHANNEL,
        ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_MENTION_USERS,
        ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL,
        ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL,
    )

    # Try to add generic webhook handler
    webhook_url = os.getenv(ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL)
    if webhook_url:
        try:
            handlers.append(WebhookAlertHandler(webhook_url=webhook_url))
            logger.info("[AlertHandler] Configured generic webhook alerting")
        except Exception as e:
            logger.error(
                "[AlertHandler] Failed to initialize WebhookAlertHandler: %s. "
                "Continuing with logging only.",
                e,
            )

    # Try to add Slack handler
    slack_webhook_url = os.getenv(ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL)
    if slack_webhook_url:
        try:
            slack_channel = os.getenv(ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_CHANNEL)
            mention_users_str = os.getenv(
                ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_MENTION_USERS, ""
            )
            mention_users = [
                u.strip() for u in mention_users_str.split(",") if u.strip()
            ]

            handlers.append(
                SlackAlertHandler(
                    webhook_url=slack_webhook_url,
                    channel=slack_channel,
                    mention_users=mention_users,
                )
            )
            logger.info("[AlertHandler] Configured Slack webhook alerting")
        except Exception as e:
            logger.error(
                "[AlertHandler] Failed to initialize SlackAlertHandler: %s. "
                "Continuing with logging only.",
                e,
            )

    if len(handlers) == 1:
        # Only logging handler, return it directly
        logger.info(
            "[AlertHandler] No external alert handlers configured. Using logging only."
        )
        return handlers[0]

    return CompositeAlertHandler(handlers=handlers)
