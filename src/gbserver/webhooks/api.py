"""FastAPI routes for webhook subscription management.

Provides endpoints to create, list, and delete webhook subscriptions
scoped to specific builds. Subscriptions allow external systems to
receive push notifications about build lifecycle events.

Routes:
    POST /{build_id}/subscriptions — Create a new subscription.
    GET /{build_id}/subscriptions — List active subscriptions for a build.
    DELETE /{webhook_id} — Deactivate a subscription (owner only).
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel

from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.constants import GBSERVER_WEBHOOKS_ALLOW_HTTP, GBSERVER_WEBHOOKS_MAX_PER_SPACE
from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import WEBHOOK_MIN_FREQUENCY, StoredWebhookSubscription
from gbserver.storage.sql.webhook_subscription_storage import create_webhook_storage
from gbserver.storage.webhook_subscription_storage import IWebhookStorage
from gbserver.webhooks.url_validator import WebhookURLError, validate_webhook_url

logger = get_logger(__name__)
webhooks_api = FastAPI()

# Module-level storage (lazily initialized)
_webhook_storage: Optional[IWebhookStorage] = None  # pylint: disable=invalid-name


def set_webhook_storage(storage: IWebhookStorage) -> None:
    """Set the module-level webhook storage instance.

    Used during application startup or in tests to inject the storage backend.

    Args:
        storage: An IWebhookStorage implementation.
    """
    global _webhook_storage  # pylint: disable=global-statement
    _webhook_storage = storage


def get_webhook_storage() -> IWebhookStorage:
    """Get the module-level webhook storage instance.

    Lazily initializes using the configured storage backend if not set.

    Returns:
        The configured IWebhookStorage implementation.
    """
    global _webhook_storage  # pylint: disable=global-statement
    if _webhook_storage is None:
        _webhook_storage = create_webhook_storage()
    return _webhook_storage


# ── Request / Response Models ─────────────────────────────────────────────


class CreateWebhookRequest(BaseModel):
    """Request body for creating a new webhook subscription.

    Args:
        webhook_url: The URL to POST event payloads to.
        secret: HMAC signing key for payload verification.
        event_types: Event types to subscribe to. Defaults to wildcard.
        excluded_types: Event types to always exclude.
        frequency: Batch flush interval in seconds.
        log_pattern: Optional regex for log line scanning.
        metadata: Arbitrary metadata dict.
    """

    webhook_url: str
    secret: str
    event_types: List[str] = ["*"]
    excluded_types: List[str] = []
    frequency: int = 30
    log_pattern: Optional[str] = None
    metadata: Dict[str, Any] = {}


class WebhookResponse(BaseModel):
    """Response model for a webhook subscription (secret excluded).

    Args:
        id: Unique subscription identifier.
        space_name: The space owning the subscription.
        webhook_url: The delivery endpoint URL.
        event_types: Subscribed event types.
        excluded_types: Excluded event types.
        frequency: Batch flush interval in seconds.
        log_pattern: Optional regex for log line scanning.
        active: Whether the subscription is currently active.
        status: Subscription lifecycle status.
        build_filter: Optional build UUID for per-build scoping.
        created_by: Username of the creator.
        created_time: ISO timestamp of creation.
    """

    id: str
    space_name: str
    webhook_url: str
    event_types: List[str]
    excluded_types: List[str]
    frequency: int
    log_pattern: Optional[str]
    active: bool
    status: str
    build_filter: Optional[str]
    created_by: str
    created_time: str


class ListWebhooksResponse(BaseModel):
    """Response model for listing webhook subscriptions.

    Args:
        subscriptions: List of active webhook subscriptions.
    """

    subscriptions: List[WebhookResponse]


# ── Helpers ───────────────────────────────────────────────────────────────


def _get_username(request: Request) -> str:
    """Extract the authenticated username from the request.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The username from the X-Forwarded-User header.

    Raises:
        HTTPException: 401 if the header is missing or empty.
    """
    username = request.headers.get("X-Forwarded-User")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Forwarded-User header",
        )
    return username


def _to_response(sub: StoredWebhookSubscription) -> WebhookResponse:
    """Convert a StoredWebhookSubscription to a WebhookResponse.

    The secret field is intentionally excluded from the response.

    Args:
        sub: The stored subscription to convert.

    Returns:
        A WebhookResponse with all public fields populated.
    """
    return WebhookResponse(
        id=sub.uuid,
        space_name=sub.space_name,
        webhook_url=sub.webhook_url,
        event_types=sub.event_types,
        excluded_types=sub.excluded_types,
        frequency=sub.frequency,
        log_pattern=sub.log_pattern,
        active=sub.active,
        status=sub.status,
        build_filter=sub.build_filter,
        created_by=sub.created_by,
        created_time=sub.created_time.isoformat(),
    )


def _check_rate_limit(storage: IWebhookStorage, space_name: str) -> None:
    """Raise 429 if space has too many active subscriptions."""
    existing = storage.get_by_space(space_name)
    active_count = sum(1 for s in existing if s.active)
    if active_count >= GBSERVER_WEBHOOKS_MAX_PER_SPACE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum {GBSERVER_WEBHOOKS_MAX_PER_SPACE} active subscriptions per space",
        )


# ── Routes ────────────────────────────────────────────────────────────────


@webhooks_api.post(
    "/{build_id}/subscriptions",
    status_code=status.HTTP_201_CREATED,
    response_model=WebhookResponse,
)
def create_subscription(
    build_id: str, body: CreateWebhookRequest, request: Request
) -> WebhookResponse:
    """Create a new webhook subscription for a build.

    Validates the authenticated user, checks that the build exists,
    enforces the minimum frequency constraint, then persists the
    subscription.

    Args:
        build_id: The build ID to subscribe to.
        body: The subscription creation request body.
        request: The incoming FastAPI request (for auth headers).

    Returns:
        WebhookResponse with the created subscription details.

    Raises:
        HTTPException: 401 if unauthenticated, 400 if frequency too low,
            404 if build not found.
    """
    username = _get_username(request)

    # Validate frequency
    if body.frequency < WEBHOOK_MIN_FREQUENCY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Frequency must be at minimum {WEBHOOK_MIN_FREQUENCY} seconds. "
                f"Got {body.frequency}."
            ),
        )

    # Validate webhook URL (SSRF protection)
    allow_http = GBSERVER_WEBHOOKS_ALLOW_HTTP
    try:
        validate_webhook_url(body.webhook_url, allow_http=allow_http)
    except WebhookURLError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid webhook URL: {e}",
        ) from e

    # Validate secret length
    if len(body.secret) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook secret must be at least 8 characters",
        )

    # Verify build exists
    admin_storage = get_admin_storage()
    build = admin_storage.build_storage.get_by_uuid(build_id)
    if build is None or isinstance(build, list):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build {build_id} not found",
        )

    # Rate limit check
    storage = get_webhook_storage()
    _check_rate_limit(storage, build.space_name)

    # Create and persist subscription
    subscription = StoredWebhookSubscription(
        space_name=build.space_name,
        build_filter=build_id,
        webhook_url=body.webhook_url,
        secret=body.secret,
        event_types=body.event_types,
        excluded_types=body.excluded_types,
        frequency=body.frequency,
        log_pattern=body.log_pattern,
        created_by=username,
        metadata=body.metadata,
        status="pending",
    )

    storage.add(subscription)

    logger.info(
        "Created webhook subscription %s for build %s by user %s",
        subscription.uuid,
        build_id,
        username,
    )

    return _to_response(subscription)


@webhooks_api.get(
    "/{build_id}/subscriptions",
    status_code=status.HTTP_200_OK,
    response_model=ListWebhooksResponse,
)
def list_subscriptions(build_id: str, request: Request) -> ListWebhooksResponse:
    """List active webhook subscriptions for a build.

    Args:
        build_id: The build ID to list subscriptions for.
        request: The incoming FastAPI request (for auth headers).

    Returns:
        ListWebhooksResponse containing active subscriptions.

    Raises:
        HTTPException: 401 if unauthenticated, 404 if build not found.
    """
    _get_username(request)

    # Verify build exists
    admin_storage = get_admin_storage()
    build = admin_storage.build_storage.get_by_uuid(build_id)
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Build {build_id} not found",
        )

    storage = get_webhook_storage()
    subscriptions = storage.get_active_for_build_filter(build_id)

    return ListWebhooksResponse(
        subscriptions=[_to_response(sub) for sub in subscriptions]
    )


@webhooks_api.post(
    "/spaces/{space_name}/subscriptions",
    status_code=status.HTTP_201_CREATED,
    response_model=WebhookResponse,
)
def create_space_subscription(
    space_name: str, body: CreateWebhookRequest, request: Request
) -> WebhookResponse:
    """Create a space-wide webhook subscription.

    Space-wide subscriptions receive events for ALL builds in the space,
    useful for monitoring dashboards or aggregate notification systems.

    Args:
        space_name: The space to subscribe to.
        body: The subscription creation request body.
        request: The incoming FastAPI request (for auth headers).

    Returns:
        WebhookResponse with the created subscription details.

    Raises:
        HTTPException: 401 if unauthenticated, 400 if frequency too low,
            404 if space not found.
    """
    username = _get_username(request)

    # Validate frequency
    if body.frequency < WEBHOOK_MIN_FREQUENCY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Frequency must be at minimum {WEBHOOK_MIN_FREQUENCY} seconds. "
                f"Got {body.frequency}."
            ),
        )

    # Validate webhook URL (SSRF protection)
    allow_http = GBSERVER_WEBHOOKS_ALLOW_HTTP
    try:
        validate_webhook_url(body.webhook_url, allow_http=allow_http)
    except WebhookURLError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid webhook URL: {e}",
        ) from e

    # Validate secret length
    if len(body.secret) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook secret must be at least 8 characters",
        )

    # Verify space exists
    admin_storage = get_admin_storage()
    spaces = admin_storage.space_storage.get_by_where({"name": space_name})
    if not spaces:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Space {space_name} not found",
        )

    # Rate limit check
    storage = get_webhook_storage()
    _check_rate_limit(storage, space_name)

    # Create and persist subscription (build_filter=None = space-wide)
    subscription = StoredWebhookSubscription(
        space_name=space_name,
        webhook_url=body.webhook_url,
        secret=body.secret,
        event_types=body.event_types,
        excluded_types=body.excluded_types,
        frequency=body.frequency,
        log_pattern=body.log_pattern,
        created_by=username,
        metadata=body.metadata,
        status="pending",
    )

    storage.add(subscription)

    logger.info(
        "Created space-wide webhook subscription %s for space %s by user %s",
        subscription.uuid,
        space_name,
        username,
    )

    return _to_response(subscription)


@webhooks_api.get(
    "/spaces/{space_name}/subscriptions",
    status_code=status.HTTP_200_OK,
    response_model=ListWebhooksResponse,
)
def list_space_subscriptions(space_name: str, request: Request) -> ListWebhooksResponse:
    """List active space-wide webhook subscriptions.

    Args:
        space_name: The space to list subscriptions for.
        request: The incoming FastAPI request (for auth headers).

    Returns:
        ListWebhooksResponse containing active space-wide subscriptions.

    Raises:
        HTTPException: 401 if unauthenticated.
    """
    _get_username(request)

    storage = get_webhook_storage()
    subscriptions = storage.get_active_for_space(space_name)

    return ListWebhooksResponse(
        subscriptions=[_to_response(sub) for sub in subscriptions]
    )


@webhooks_api.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_subscription(webhook_id: str, request: Request) -> Response:
    """Deactivate a webhook subscription.

    Only the subscription creator can deactivate it.

    Args:
        webhook_id: The UUID of the subscription to deactivate.
        request: The incoming FastAPI request (for auth headers).

    Returns:
        Empty 204 response on success.

    Raises:
        HTTPException: 401 if unauthenticated, 404 if subscription not found,
            403 if the caller is not the subscription owner.
    """
    username = _get_username(request)

    storage = get_webhook_storage()
    subscription = storage.get_by_uuid(webhook_id)
    if subscription is None or isinstance(subscription, list):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscription {webhook_id} not found",
        )

    if subscription.created_by != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the subscription creator can delete it",
        )

    storage.deactivate(webhook_id)

    logger.info(
        "Deactivated webhook subscription %s by user %s",
        webhook_id,
        username,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
