"""Tests for embedded nats-server in standalone command."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.g4os


class TestEmbeddedNatsServer:
    """Tests for the embedded nats-server lifecycle management."""

    def test_start_nats_server_when_binary_found(self, tmp_path):
        """Starts nats-server subprocess when binary is on PATH."""
        from gbserver.commands.command_standalone import _start_nats_server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is running

        with (
            patch("shutil.which", return_value="/usr/local/bin/nats-server"),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch(
                "gbserver.commands.command_standalone._wait_for_nats", return_value=True
            ),
        ):
            proc = _start_nats_server(str(tmp_path), port=4222)

        assert proc is mock_proc
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "nats-server" in call_args[0]
        assert "-js" in call_args

    def test_returns_none_when_binary_not_found(self, tmp_path):
        """Returns None when nats-server is not on PATH."""
        from gbserver.commands.command_standalone import _start_nats_server

        with patch("shutil.which", return_value=None):
            proc = _start_nats_server(str(tmp_path), port=4222)

        assert proc is None

    def test_stop_nats_server_sends_sigterm(self):
        """_stop_nats_server sends SIGTERM and waits."""
        from gbserver.commands.command_standalone import _stop_nats_server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        _stop_nats_server(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_stop_nats_server_noop_for_none(self):
        """_stop_nats_server is a no-op when proc is None."""
        from gbserver.commands.command_standalone import _stop_nats_server

        _stop_nats_server(None)  # Should not raise
