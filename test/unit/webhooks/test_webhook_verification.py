"""Unit tests for webhook URL ownership verification challenge."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.webhooks.verification import verify_url_ownership


class TestURLVerification:
    """Tests for verify_url_ownership challenge flow."""

    @pytest.mark.asyncio
    async def test_successful_verification(self):
        """Endpoint echoes challenge back -> returns True."""
        challenge_token = "test-challenge-token"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"challenge": challenge_token})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch(
                "gbserver.webhooks.verification._generate_challenge",
                return_value=challenge_token,
            ):
                result = await verify_url_ownership("https://example.com/webhook")

        assert result is True

    @pytest.mark.asyncio
    async def test_wrong_challenge_response(self):
        """Endpoint returns wrong challenge -> returns False."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"challenge": "wrong-token"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch(
                "gbserver.webhooks.verification._generate_challenge",
                return_value="expected-token",
            ):
                result = await verify_url_ownership("https://example.com/webhook")

        assert result is False

    @pytest.mark.asyncio
    async def test_endpoint_returns_non_200(self):
        """Endpoint returns 500 -> returns False."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await verify_url_ownership("https://example.com/webhook")

        assert result is False

    @pytest.mark.asyncio
    async def test_endpoint_unreachable(self):
        """Connection error -> returns False."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await verify_url_ownership(
                "https://unreachable.example.com/webhook"
            )

        assert result is False
