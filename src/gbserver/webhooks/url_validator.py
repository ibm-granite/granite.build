"""Webhook URL validation with SSRF protection.

Validates webhook endpoint URLs by checking:
- HTTPS enforcement (configurable for dev)
- IP address against blocked private/internal ranges
- Cloud metadata endpoint blocking (169.254.x.x)
"""

import ipaddress
import socket
from urllib.parse import urlparse

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}


class WebhookURLError(ValueError):
    """Raised when a webhook URL fails validation."""


def _is_ip_blocked(ip_str: str) -> bool:
    """Check if an IP address falls within a blocked network range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in network for network in BLOCKED_NETWORKS)


def validate_webhook_url(url: str, allow_http: bool = False) -> None:
    """Validate a webhook URL for safety (SSRF protection).

    Args:
        url: The webhook endpoint URL to validate.
        allow_http: If True, allow http:// URLs (for dev/testing only).

    Raises:
        WebhookURLError: If the URL fails any validation check.
    """
    if not url:
        raise WebhookURLError("URL cannot be empty")

    parsed = urlparse(url)

    if not parsed.scheme:
        raise WebhookURLError(f"Invalid URL (no scheme): {url}")
    if parsed.scheme not in ("http", "https"):
        raise WebhookURLError(f"Invalid URL scheme: {parsed.scheme}")
    if parsed.scheme == "http" and not allow_http:
        raise WebhookURLError("HTTPS required for webhook URLs")

    hostname = parsed.hostname
    if not hostname:
        raise WebhookURLError(f"Invalid URL (no host): {url}")

    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise WebhookURLError(
            f"Webhook URL blocked (private/blocked host): {hostname}"
        )

    # Check if hostname is a literal IP address
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_ip_blocked(str(addr)):
            raise WebhookURLError(
                f"Webhook URL blocked (private/blocked IP): {hostname}"
            )
        return
    except ValueError:
        pass

    # DNS resolution — check all resolved addresses
    try:
        addr_infos = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise WebhookURLError(f"DNS resolution failed for {hostname}: {e}") from e

    if not addr_infos:
        raise WebhookURLError(f"DNS resolution returned no results for {hostname}")

    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        if _is_ip_blocked(ip_str):
            raise WebhookURLError(
                f"Webhook URL blocked (private/blocked IP {ip_str} for host {hostname})"
            )
