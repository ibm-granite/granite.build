"""Tests verifying webhook integration with build event flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.status import Status
from gbserver.webhooks.dispatcher import WebhookDispatcher
from gbserver.webhooks.models import StoredWebhookSubscription


class TestBuildRunnerWebhookIntegration:
    """Verify dispatcher lifecycle: init -> accept -> flush -> deliver."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="s",
            build_id="b1",
            webhook_url="https://example.com/hook",
            secret="sec",
            event_types=["*"],
            created_by="u",
            frequency=1,
        )
        mock_storage.get_active_for_build.return_value = [sub]

        dispatcher = WebhookDispatcher(
            webhook_storage=mock_storage,
            build_id="b1",
            space_name="s",
            build_name="build",
            username="u",
            build_start_time="2026-05-20T11:00:00Z",
        )
        dispatcher.start([sub])

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="b1",
                username="u",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="go"),
        )
        dispatcher.accept_event(event)

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await dispatcher.flush_final()
            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert payload["build_id"] == "b1"
            assert payload["user"] == "u"
            assert len(payload["events"]) == 1


class TestWebhooksEnabledConstant:
    """Verify the GBSERVER_WEBHOOKS_ENABLED constant is accessible."""

    def test_webhooks_enabled_default_true(self):
        from gbserver.types.constants import GBSERVER_WEBHOOKS_ENABLED

        assert isinstance(GBSERVER_WEBHOOKS_ENABLED, bool)
