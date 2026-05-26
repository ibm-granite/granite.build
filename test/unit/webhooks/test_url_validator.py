"""Unit tests for webhook URL validation (SSRF protection)."""

import pytest

from gbserver.webhooks.url_validator import WebhookURLError, validate_webhook_url


class TestURLValidator:
    """Tests for validate_webhook_url."""

    def test_valid_https_url_with_ip(self):
        """Public HTTPS URL with public IP passes validation."""
        # 8.8.8.8 is Google DNS, always public
        validate_webhook_url("https://8.8.8.8/webhook")

    def test_rejects_http_by_default(self):
        """Plain HTTP is rejected by default."""
        with pytest.raises(WebhookURLError, match="HTTPS required"):
            validate_webhook_url("http://8.8.8.8/webhook")

    def test_allows_http_when_configured(self):
        """HTTP allowed when allow_http=True (for dev/testing)."""
        validate_webhook_url("http://8.8.8.8/webhook", allow_http=True)

    def test_rejects_loopback_ipv4(self):
        """127.x.x.x addresses are blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://127.0.0.1/webhook")

    def test_rejects_private_10(self):
        """10.x.x.x addresses are blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://10.0.0.1/webhook")

    def test_rejects_private_172(self):
        """172.16-31.x.x addresses are blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://172.16.0.1/webhook")

    def test_rejects_private_192(self):
        """192.168.x.x addresses are blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://192.168.1.1/webhook")

    def test_rejects_link_local(self):
        """169.254.x.x (cloud metadata) addresses are blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://169.254.169.254/latest/meta-data/")

    def test_rejects_empty_url(self):
        """Empty string is rejected."""
        with pytest.raises(WebhookURLError):
            validate_webhook_url("")

    def test_rejects_no_host(self):
        """URL with no host is rejected."""
        with pytest.raises(WebhookURLError):
            validate_webhook_url("https:///path")

    def test_rejects_non_url(self):
        """Non-URL string is rejected."""
        with pytest.raises(WebhookURLError):
            validate_webhook_url("not-a-url")

    def test_allows_public_ip(self):
        """Public IP address is allowed."""
        validate_webhook_url("https://8.8.8.8/webhook")

    def test_rejects_localhost_hostname(self):
        """'localhost' hostname is blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://localhost/webhook")

    def test_rejects_zero_ip(self):
        """0.0.0.0 is blocked."""
        with pytest.raises(WebhookURLError, match="blocked"):
            validate_webhook_url("https://0.0.0.0/webhook")
