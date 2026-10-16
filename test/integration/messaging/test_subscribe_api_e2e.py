"""E2E test for the event subscription REST API against a deployed GB server.

Tests the full client flow:
  1. Call POST /api/v1/builds/{build_id}/events/subscribe (REST API)
  2. Connect to RabbitMQ with the returned scoped credentials
  3. Publish a simulated event (using admin credentials)
  4. Verify the scoped consumer receives it

Required environment variables:
    GBSERVER_REST_URL              - Deployed API base URL
                                     (e.g. https://api.llm-build-dev.vpc-int.res.ibm.com)
    GBSERVER_GITHUB_TOKEN          - GitHub token for Bearer auth against the REST API
    GBSERVER_RABBITMQ_MGMT_URL     - RabbitMQ Management API URL
    GBSERVER_RABBITMQ_MGMT_USER    - RabbitMQ admin username (for publishing test events)
    GBSERVER_RABBITMQ_MGMT_PASSWORD - RabbitMQ admin password

Optional:
    GBSERVER_RABBITMQ_AMQP_PORT    - AMQP port (default: 5672)
    RABBITMQ_HOST                  - AMQP host (default: derived from MGMT URL)
    RABBITMQ_TLS                   - Enable TLS for AMQP (default: true)
    RABBITMQ_CA_CERT               - CA cert path for TLS verification
    E2E_BUILD_ID                   - Specific build ID to test (default: uses build list)

Run with:
    pytest test/integration/messaging/test_subscribe_api_e2e.py -v -s -m ibm
"""

import asyncio
import json
import os
import ssl
import time
from urllib.parse import urlparse

import httpx
import pytest

# All required env vars for this test
_REQUIRED_VARS = [
    "GBSERVER_REST_URL",
    "GBSERVER_GITHUB_TOKEN",
    "GBSERVER_RABBITMQ_MGMT_URL",
    "GBSERVER_RABBITMQ_MGMT_USER",
    "GBSERVER_RABBITMQ_MGMT_PASSWORD",
]

_missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.ibm,
    pytest.mark.extended,
    pytest.mark.skipif(
        len(_missing) > 0,
        reason=f"Missing env vars: {', '.join(_missing)}",
    ),
]


def _get_config():
    """Load test configuration from environment."""
    rest_url = os.environ["GBSERVER_REST_URL"].rstrip("/")
    mgmt_url = os.environ["GBSERVER_RABBITMQ_MGMT_URL"]
    mgmt_host = urlparse(mgmt_url).hostname or "localhost"

    host = os.getenv("RABBITMQ_HOST") or mgmt_host
    port = int(os.getenv("GBSERVER_RABBITMQ_AMQP_PORT", "5672"))
    tls = os.getenv("RABBITMQ_TLS", "true").lower() in ("true", "1")
    ca_cert = os.getenv("RABBITMQ_CA_CERT", "")

    return {
        "rest_url": rest_url,
        "token": os.environ["GBSERVER_GITHUB_TOKEN"],
        "mgmt_url": mgmt_url,
        "mgmt_user": os.environ["GBSERVER_RABBITMQ_MGMT_USER"],
        "mgmt_password": os.environ["GBSERVER_RABBITMQ_MGMT_PASSWORD"],
        "host": host,
        "port": port,
        "tls": tls,
        "ca_cert": ca_cert,
    }


def _make_ssl_context(ca_cert: str):
    """Create SSL context for AMQP connection."""
    if ca_cert and os.path.isfile(ca_cert):
        return ssl.create_default_context(cafile=ca_cert)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _get_build_id(rest_url: str, token: str) -> str:
    """Get a valid build ID from the API, or use E2E_BUILD_ID if set."""
    if os.getenv("E2E_BUILD_ID"):
        return os.environ["E2E_BUILD_ID"]

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.get(
            f"{rest_url}/api/v1/builds/",
            headers={"Authorization": f"Bearer {token}"},
            params={"page_size": 1, "show_all": "true"},
        )
        resp.raise_for_status()
        data = resp.json()
        # The API returns {"builds": [...], ...} via ListBuildResponse
        if isinstance(data, dict) and data.get("builds"):
            return data["builds"][0]["uuid"]
        pytest.skip("No builds found in the server to test against")


async def test_subscribe_api_returns_credentials():
    """POST /subscribe returns valid RabbitMQ credentials."""
    cfg = _get_config()
    build_id = await _get_build_id(cfg["rest_url"], cfg["token"])

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            f"{cfg['rest_url']}/api/v1/builds/{build_id}/events/subscribe",
            headers={"Authorization": f"Bearer {cfg['token']}"},
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Verify response structure
    assert data["delivery_type"] == "rabbitmq"
    assert data["host"]
    assert data["port"] > 0
    assert data["username"]
    assert data["password"]
    assert data["exchange"]
    assert data["routing_key"] == f"build.{build_id}.#"
    assert data["expires_at"] > int(time.time())


async def test_subscribe_and_receive_events():
    """Full flow: subscribe via API, connect, publish event, verify receipt."""
    import aio_pika

    cfg = _get_config()
    build_id = await _get_build_id(cfg["rest_url"], cfg["token"])

    # 1. Subscribe via REST API
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            f"{cfg['rest_url']}/api/v1/builds/{build_id}/events/subscribe",
            headers={"Authorization": f"Bearer {cfg['token']}"},
        )

    assert resp.status_code == 200, f"Subscribe failed: {resp.status_code}: {resp.text}"
    sub = resp.json()

    # Derive TLS settings from subscribe response, falling back to local config
    use_tls = sub.get("tls", cfg["tls"])
    ssl_ctx = _make_ssl_context(cfg["ca_cert"]) if use_tls else None

    # 2. Connect as scoped consumer using returned credentials
    consumer_conn = await aio_pika.connect(
        host=sub["host"],
        port=sub["port"],
        login=sub["username"],
        password=sub["password"],
        ssl=use_tls,
        ssl_context=ssl_ctx,
    )
    consumer_chan = await consumer_conn.channel()
    queue_name = sub.get("queue") or f"events.{build_id}.e2e-api-test"
    queue = await consumer_chan.declare_queue(queue_name, auto_delete=True)
    exchange = await consumer_chan.get_exchange(sub["exchange"], ensure=False)
    await queue.bind(exchange, routing_key=sub["routing_key"])

    received = []

    async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
        async with message.process():
            received.append(json.loads(message.body))

    await queue.consume(on_message)

    # 3. Publish a test event (using admin credentials, as the server would)
    pub_connect_kwargs = dict(
        host=cfg["host"],
        port=cfg["port"],
        login=cfg["mgmt_user"],
        password=cfg["mgmt_password"],
    )
    if use_tls:
        pub_connect_kwargs.update(ssl=True, ssl_context=ssl_ctx)

    pub_conn = await aio_pika.connect(**pub_connect_kwargs)
    pub_chan = await pub_conn.channel()
    pub_exchange = await pub_chan.declare_exchange(
        sub["exchange"], aio_pika.ExchangeType.TOPIC, durable=True
    )

    event_payload = {
        "build_id": build_id,
        "event_type": "status_event",
        "timestamp": int(time.time()),
        "target_name": "e2e-test-target",
        "step_name": "",
        "source": "e2e-api-test",
        "status": "running",
        "message": "E2E API integration test event",
    }
    msg = aio_pika.Message(json.dumps(event_payload).encode())
    await pub_exchange.publish(msg, routing_key=f"build.{build_id}.status_event")

    # 4. Wait for delivery
    for _ in range(20):
        if received:
            break
        await asyncio.sleep(0.5)

    # 5. Cleanup
    await consumer_conn.close()
    await pub_conn.close()

    # 6. Verify
    assert len(received) >= 1, f"Expected at least 1 event, got {len(received)}"
    evt = received[0]
    assert evt["build_id"] == build_id
    assert evt["event_type"] == "status_event"
    assert evt["source"] == "e2e-api-test"
    assert evt["message"] == "E2E API integration test event"


async def test_subscribe_invalid_build_returns_404():
    """Subscribe with a non-existent build ID returns 404."""
    cfg = _get_config()

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            f"{cfg['rest_url']}/api/v1/builds/nonexistent-build-00000/events/subscribe",
            headers={"Authorization": f"Bearer {cfg['token']}"},
        )

    assert resp.status_code == 404


async def test_subscribe_without_auth_returns_401():
    """Subscribe without auth header returns 401."""
    cfg = _get_config()

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            f"{cfg['rest_url']}/api/v1/builds/dummy-build-id-12345/events/subscribe",
        )

    assert resp.status_code == 401
