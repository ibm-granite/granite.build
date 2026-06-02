"""Unit tests for notification configuration loader."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gbserver.notifications.config import load_notification_config


class TestLoadNotificationConfig:
    """Tests for load_notification_config."""

    def test_load_config_from_explicit_path(self, tmp_path):
        """Loads config from an explicitly provided file path."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: my-token\n"
            "    chat_id: '12345'\n"
            "    events: [status_event]\n"
        )

        result = load_notification_config(str(config_file))

        assert len(result) == 1
        assert result[0]["type"] == "telegram"
        assert result[0]["bot_token"] == "my-token"
        assert result[0]["chat_id"] == "12345"
        assert result[0]["events"] == ["status_event"]

    def test_resolves_env_keys_from_environment(self, tmp_path):
        """Keys ending in _env are resolved from environment variables."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token_env: MY_BOT_TOKEN\n"
            "    chat_id: '99999'\n"
            "    events: ['*']\n"
        )

        with patch.dict(os.environ, {"MY_BOT_TOKEN": "secret-token-value"}):
            result = load_notification_config(str(config_file))

        assert len(result) == 1
        assert result[0]["bot_token"] == "secret-token-value"
        assert "bot_token_env" not in result[0]
        assert result[0]["chat_id"] == "99999"

    def test_returns_empty_list_if_file_not_found(self):
        """Returns empty list when no config file exists."""
        result = load_notification_config("/nonexistent/path/notifications.yaml")
        assert result == []

    def test_returns_empty_list_if_no_default_paths_exist(self):
        """Returns empty list when no default config files exist."""
        with patch(
            "gbserver.notifications.config._DEFAULT_CONFIG_PATHS",
            [Path("/nonexistent/a.yaml"), Path("/nonexistent/b.yaml")],
        ):
            result = load_notification_config(None)
        assert result == []

    def test_env_key_resolves_to_none_if_var_unset(self, tmp_path):
        """If the referenced env var is not set, the resolved value is None."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token_env: DEFINITELY_NOT_SET_XYZ\n"
            "    chat_id: '111'\n"
        )

        with patch.dict(os.environ, {}, clear=False):
            # Ensure the var is not set
            os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
            result = load_notification_config(str(config_file))

        assert result[0]["bot_token"] is None

    def test_multiple_notifications(self, tmp_path):
        """Correctly parses multiple notification entries."""
        config_file = tmp_path / "notifications.yaml"
        config_file.write_text(
            "notifications:\n"
            "  - type: telegram\n"
            "    bot_token: token1\n"
            "    chat_id: '111'\n"
            "  - type: telegram\n"
            "    bot_token: token2\n"
            "    chat_id: '222'\n"
        )

        result = load_notification_config(str(config_file))

        assert len(result) == 2
        assert result[0]["bot_token"] == "token1"
        assert result[1]["bot_token"] == "token2"
