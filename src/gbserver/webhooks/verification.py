"""Webhook URL ownership verification via challenge-response.

Before activating a webhook subscription, we POST a challenge token to
the endpoint. The endpoint must respond with the same token to prove
the registrant controls it.
"""

import secrets

import aiohttp

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

VERIFICATION_TIMEOUT = 10


def _generate_challenge() -> str:
    """Generate a cryptographically random challenge token."""
    return secrets.token_urlsafe(32)


async def verify_url_ownership(
    webhook_url: str,
    timeout: int = VERIFICATION_TIMEOUT,
) -> bool:
    """Send a verification challenge to a webhook URL.

    POSTs a JSON payload with a random challenge token. The endpoint
    must respond with 200 and echo the challenge back in the response
    body as {"challenge": "<token>"}.

    Args:
        webhook_url: The URL to verify ownership of.
        timeout: HTTP request timeout in seconds.

    Returns:
        True if the endpoint correctly echoed the challenge, False otherwise.
    """
    challenge = _generate_challenge()
    payload = {
        "type": "url_verification",
        "challenge": challenge,
    }

    headers = {
        "Content-Type": "application/json",
        "X-GB-Event": "verification",
    }

    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                webhook_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "Verification challenge to %s returned status %d",
                        webhook_url,
                        response.status,
                    )
                    return False

                body = await response.json()
                if body.get("challenge") != challenge:
                    logger.warning(
                        "Verification challenge to %s: wrong challenge response",
                        webhook_url,
                    )
                    return False

                logger.info("Verification challenge to %s succeeded", webhook_url)
                return True
    except Exception as e:
        logger.warning("Verification challenge to %s failed: %s", webhook_url, e)
        return False
