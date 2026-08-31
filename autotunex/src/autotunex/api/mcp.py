# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FastMCP server exposing the AutoTuneX tool registry over MCP.

Mounted at ``/mcp`` only when ``settings.enable_mcp`` is set (see
:func:`mount_mcp`). The base install never imports ``fastmcp`` at all: every
reference to it below is either lazy (inside a function body, resolved only
when that function actually runs) or confined to a ``TYPE_CHECKING`` block —
so ``import autotunex.api.mcp`` succeeds whether or not the ``[mcp]`` extra is
installed, and the base ``autotunex`` install never pays for a dependency most
deployments do not use.

Resolved version, verified against this repo while building this module:
``fastmcp==3.4.7`` (pulling in ``mcp==1.29.0``), from the ``fastmcp>=2.3,<4``
bound in ``pyproject.toml``'s ``[mcp]`` extra — see that bound's comment for
why it is a floor-and-ceiling rather than an exact pin, and confirmation that
resolving it forces no change to this project's exactly-pinned
fastapi/starlette/pydantic/httpx.

Four version-sensitive framework details this module depends on are pinned
here because they are not obvious from fastmcp's public docs, and were each
confirmed by reading fastmcp 3.4.7's own source and probing it directly rather
than assumed:

* **Reading the incoming HTTP request inside a tool.** A tool body running
  under ``http_app()`` can call
  ``fastmcp.server.dependencies.get_http_request()`` to get the current
  Starlette ``Request`` (it reads a context var fastmcp's ASGI layer sets per
  request) and read arbitrary headers off it directly — no ``Context``
  parameter is needed. It raises ``RuntimeError`` when no request is active,
  which is always true under the in-memory ``Client``/``FastMCPTransport``
  transport (it talks to the server over raw in-memory MCP JSON-RPC streams,
  with no ASGI ``Request`` at all) — so an authenticated *tool call* cannot be
  exercised through the in-memory client for this fastmcp version.
  ``tests/api/test_mcp.py`` tests :func:`resolve_mcp_principal` directly
  instead, and uses the in-memory client only to assert tool registration.
* **Registering a tool built from a runtime-derived parameter list.**
  ``mcp.tool()`` calls ``ParsedFunction.from_function()``, which raises
  ``ValueError`` for any function whose *reported* ``inspect.signature()``
  contains ``**kwargs`` — so the obvious ``async def wrapper(**kwargs: Any)``
  cannot be registered directly. Each tool wrapper below (see
  :func:`_make_tool_fn`) therefore keeps a real ``**kwargs`` body — callable
  with whatever keys a ``ToolSpec.params`` model defines — but overrides
  ``__signature__`` and ``__annotations__`` with a synthetic, keyword-only
  parameter list built from that model's fields. fastmcp's pydantic
  ``TypeAdapter``-based schema generation and argument validation then see a
  normal, richly-typed function with no ``VAR_KEYWORD``, so registration
  succeeds and the generated JSON schema carries each parameter's real name,
  type, description, default, and required-ness.
* **Signaling an authentication failure from inside a tool.** Raising
  ``fastmcp.exceptions.ToolError`` propagates to the MCP client with the given
  message intact and no traceback leak — confirmed directly. This is
  deliberately not fastmcp's OAuth-scopes authorization mechanism
  (``fastmcp.utilities.authorization.AuthCheck``/``AuthContext``, wired via
  ``mcp.tool(auth=...)``), which models per-tool ``AccessToken`` scopes — a
  different kind of authorization than this project's custom
  X-API-Key-to-domain-``Authenticator`` credential model.
* **Mounting the server at exactly ``/mcp`` under an existing FastAPI app.**
  ``FastMCP.http_app()`` defaults to registering its own internal route at
  ``fastmcp.settings.streamable_http_path`` ("/mcp"); mounting that default
  app at an outer prefix of "/mcp" (``app.mount("/mcp", mcp.http_app())``)
  therefore requires callers to hit "/mcp/mcp", confirmed directly against a
  running app. Passing ``path="/"`` roots the sub-app's own route at "/", so
  the mount prefix alone becomes the caller-visible path. Separately, the
  mounted sub-app's ``StreamableHTTPSessionManager`` needs its lifespan run —
  without it, every request fails with "Task group is not initialized" — but
  :func:`mount_mcp` receives an already-constructed ``FastAPI`` app (wiring it
  into ``create_app`` is a later task, not this module's job) whose lifespan
  is already fixed. ``fastmcp.utilities.lifespan.combine_lifespans`` is built
  for exactly this: confirmed directly that mutating
  ``app.router.lifespan_context`` in place, after construction, to combine the
  app's existing lifespan with the mounted sub-app's, works correctly.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.auth.disabled import STANDALONE_PROVIDER
from autotunex.core.auth.protocols import Authenticator
from autotunex.core.auth.registry import build_authenticator
from autotunex.core.config import ADMIN_ROLE, Settings
from autotunex.core.exceptions import AuthenticationError
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import UserRepository
from autotunex.db.repositories.sqlalchemy import SqlAlchemyUserRepository
from autotunex.db.session import get_session_factory
from autotunex.models.auth import Principal
from autotunex.services.chat.context import ToolContext
from autotunex.services.chat.tools import TOOLS, ToolSpec, run_tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from fastmcp import FastMCP

logger = get_logger(__name__)


async def resolve_mcp_principal(
    *,
    api_key: str | None,
    authenticator: Authenticator,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> Principal:
    """Resolve one MCP request's ``X-API-Key`` header into a scoped ``Principal``.

    Mirrors ``api/deps.get_authenticated_principal`` (stage one: the raw
    credential to a ``Principal`` with only ``email``/``provider`` set) and
    ``api/deps.get_principal`` (stage two: resolving ``user_id``/``is_admin``
    against ``users``) exactly, but takes its collaborators as parameters
    instead of FastAPI dependencies or ``app.state`` — there is no ``Request``
    object here to build a real dependency chain from, and this shape is what
    makes the function unit-testable with a fake ``Authenticator`` and an
    in-memory session factory. MCP has no bearer or session-cookie transport of
    its own, so only ``api_key`` is threaded through; ``bearer`` and ``session``
    are always ``None``, exactly as a caller who presented no such credential
    over HTTP would see.

    Raises:
        AuthenticationError: no credential was presented, or the one presented
            does not verify — the same exceptions
            ``get_authenticated_principal`` raises.
    """
    authenticated = await authenticator.authenticate(bearer=None, api_key=api_key, session=None)
    if authenticated.email is None:
        return authenticated

    async with session_factory() as session:
        user_repository: UserRepository = SqlAlchemyUserRepository(session)
        user = await user_repository.get_by_email(authenticated.email)
        should_provision = (
            settings.auto_provision_users or authenticated.provider == STANDALONE_PROVIDER
        )
        if user is None and should_provision:
            user = await user_repository.provision(authenticated.email)
        if authenticated.provider == STANDALONE_PROVIDER:
            is_admin = authenticated.is_admin
        else:
            is_admin = user is not None and user.role == ADMIN_ROLE
        return authenticated.model_copy(
            update={"user_id": user.id if user is not None else None, "is_admin": is_admin}
        )


def _tool_parameters(
    params_model: type[BaseModel],
) -> tuple[list[inspect.Parameter], dict[str, Any]]:
    """Derive a synthetic keyword-only parameter list from ``params_model``.

    See the module docstring's second bullet: fastmcp's schema-derivation path
    rejects any function whose reported ``inspect.signature()`` contains
    ``**kwargs``, so a tool wrapper's real, catch-all body cannot be
    registered as-is. This builds the synthetic signature pieces substituted
    in instead — one keyword-only ``inspect.Parameter`` per field of
    ``params_model``, carrying that field's real type, an ``Annotated``
    description (when the field declares one), and its default (or no default
    at all when the field is required) — alongside a matching annotations
    dict, so fastmcp's pydantic ``TypeAdapter``-based schema generation sees a
    normal, richly-typed function.
    """
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for field_name, field in params_model.model_fields.items():
        annotation: Any = field.annotation
        if field.description:
            annotation = Annotated[annotation, Field(description=field.description)]
        annotations[field_name] = annotation
        if field.is_required():
            parameters.append(
                inspect.Parameter(
                    field_name, kind=inspect.Parameter.KEYWORD_ONLY, annotation=annotation
                )
            )
        else:
            default = field.default if field.default is not PydanticUndefined else None
            parameters.append(
                inspect.Parameter(
                    field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=annotation,
                    default=default,
                )
            )
    return parameters, annotations


async def _resolve_request_principal(authenticator: Authenticator, settings: Settings) -> Principal:
    """Resolve the calling ``Principal`` from the active request's ``X-API-Key`` header.

    Lazily imports fastmcp's request-context helper and exception type (see
    the module docstring) — this only ever runs from inside a registered
    tool, i.e. only when fastmcp is installed and a request is being served.
    Both "no active request" (impossible under ``http_app()``, but always
    true under the in-memory test ``Client`` — see the module docstring) and a
    credential that fails to verify are converted to
    ``fastmcp.exceptions.ToolError``, confirmed to propagate cleanly to an MCP
    client with its message intact.
    """
    from fastmcp.exceptions import ToolError as FastMcpToolError
    from fastmcp.server.dependencies import get_http_request

    try:
        request = get_http_request()
    except RuntimeError as exc:
        raise FastMcpToolError("Authentication is required.") from exc

    try:
        return await resolve_mcp_principal(
            api_key=request.headers.get("x-api-key"),
            authenticator=authenticator,
            session_factory=get_session_factory(),
            settings=settings,
        )
    except AuthenticationError as exc:
        raise FastMcpToolError(exc.detail) from exc


def _make_tool_fn(
    spec: ToolSpec, authenticator: Authenticator, settings: Settings
) -> Callable[..., Awaitable[str]]:
    """Build one fastmcp-registrable wrapper delegating to ``run_tool`` for ``spec``.

    The wrapper's real body is a plain ``**kwargs`` catch-all — callable with
    whatever keys ``spec.params`` defines — while its *reported* signature
    (``__signature__``/``__annotations__``, from :func:`_tool_parameters`) is
    the synthetic one fastmcp's registration and schema-derivation machinery
    need (see the module docstring's second bullet). Authenticates the caller
    from the current request's ``X-API-Key`` header before every call — the
    single point where every registered tool is gated — then delegates to
    :func:`autotunex.services.chat.tools.run_tool`, the exact same handler the
    in-app chat agent calls, so a tool's behavior is defined in exactly one
    place.
    """

    async def _impl(**kwargs: Any) -> str:  # noqa: ANN401 — real args are validated by run_tool
        principal = await _resolve_request_principal(authenticator, settings)
        ctx = ToolContext.for_principal(principal, settings)
        return await run_tool(spec.name, kwargs, ctx)

    parameters, annotations = _tool_parameters(spec.params)
    annotations["return"] = str
    _impl.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=str
    )
    _impl.__annotations__ = annotations
    _impl.__name__ = spec.name
    _impl.__doc__ = spec.description
    return _impl


def build_mcp(settings: Settings) -> FastMCP:
    """Build the FastMCP server exposing every tool in ``TOOLS``.

    Registers each entry as a thin wrapper (see :func:`_make_tool_fn`) around
    the shared tool registry, so a tool's behavior — and its ownership
    scoping — is defined in exactly one place, consumed identically by the
    in-app chat agent and this server. fastmcp is imported here, not at
    module scope: importing ``autotunex.api.mcp`` never requires the ``[mcp]``
    extra; only calling this function (or :func:`mount_mcp`, when
    ``settings.enable_mcp`` is set) does.
    """
    from fastmcp import FastMCP

    authenticator = build_authenticator(settings)
    mcp = FastMCP("AutoTuneX")
    for spec in TOOLS.values():
        fn = _make_tool_fn(spec, authenticator, settings)
        mcp.tool(fn, name=spec.name, description=spec.description)
    return mcp


def mount_mcp(app: FastAPI, settings: Settings) -> None:
    """Mount the FastMCP server at ``/mcp`` when ``settings.enable_mcp`` is set.

    A no-op otherwise, so an app that never enables the feature never imports
    fastmcp. ``app`` arrives already fully constructed (wiring this call into
    ``create_app`` is a later task, not this module's), so its lifespan is
    combined with the mounted sub-app's by mutating ``app.router.lifespan_context``
    in place, and ``http_app(path="/")`` is used rather than the default — see
    the module docstring's last two bullets for why both are load-bearing,
    not stylistic choices.
    """
    if not settings.enable_mcp:
        return

    if "api_key" not in settings.auth_providers:
        logger.warning(
            "MCP server mounted at /mcp without the 'api_key' auth provider. External "
            "clients have no credential they can present as X-API-Key, so every MCP "
            "tool call will resolve through whatever provider IS configured — in "
            "standalone/disabled auth that means every caller resolves to the "
            "standalone system owner with no credential check at all. Add 'api_key' "
            "to auth_providers if MCP is meant to be reachable by external clients."
        )

    from fastmcp.utilities.lifespan import combine_lifespans

    mcp_app = build_mcp(settings).http_app(path="/")
    app.router.lifespan_context = combine_lifespans(app.router.lifespan_context, mcp_app.lifespan)
    app.mount("/mcp", mcp_app)
