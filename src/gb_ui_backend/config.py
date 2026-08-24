"""Central configuration — all env vars for the analytics service."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gbcommon.types.gbenvconfig import parse_boolean

# Default SQLite filename for the analytics database.
# Stored in ~/.granite.build/ alongside gbserver's llmb-server.db.
ANALYTICS_DB_FILENAME = "dashboard-analytics.db"


def has_anthropic_chat_config() -> bool:
    # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN are deliberately not GB_UI_-
    # prefixed — they're the standard Anthropic SDK credentials, read
    # directly from os.environ by the `anthropic` package itself. Either one
    # authenticates (API key for the direct Anthropic API, auth token for an
    # internal gateway/proxy). Shared by Config.chat_enabled and
    # tool_loop_backend.py's _build_provider() — both need the identical
    # check, one to report whether chat is configured at all, the other to
    # actually gate which provider gets constructed.
    return bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(
        os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


def has_openai_compat_chat_config(config: "Config") -> bool:
    return bool(config.resolved_chat_llm_base_url and config.resolved_chat_llm_api_key)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GB_UI_",
        # Look for .env in the repo root (../../ relative to src/gb_ui_backend/).
        # Falls back gracefully if the file doesn't exist.
        env_file=os.path.join(os.path.dirname(__file__), "../../../.env"),
        extra="ignore",
    )

    # Analytics database — SQLite or PostgreSQL.
    # Auto-derived by gbserver (see derive_analytics_database_url in
    # gbserver.types.constants) from the main store's own backend config when unset:
    # GBSERVER_METADATA_STORAGE=sql inherits GBSERVER_SQL_* (translated to an asyncpg
    # URL); GBSERVER_METADATA_STORAGE=sqlite defaults to its own SQLite file under
    # GB_HOME_DIR (see ANALYTICS_DB_FILENAME).
    database_url: str = Field(
        default="",
        description="SQLAlchemy async URL. Auto-derived by gbserver from the main store's backend config when unset.",
    )

    # Extra kwargs for create_async_engine(), JSON-encoded (e.g. {"ssl": ...} is built
    # from this by _get_engine() — see db_schema.py). Only gbserver's
    # _configure_analytics_env sets this (via GB_UI_DATABASE_CONNECT_ARGS), carrying
    # the main SQL store's TLS cert path across the env-var boundary since an
    # ssl.SSLContext object itself isn't JSON-serializable.
    database_connect_args_json: str = Field(
        default="{}", alias="GB_UI_DATABASE_CONNECT_ARGS"
    )

    @property
    def database_connect_args(self) -> dict:
        return json.loads(self.database_connect_args_json)

    # gbserver REST API
    gbserver_url: str = Field(default="http://localhost:8080")

    # GBMCP server (MCP-over-HTTP) — required for flight plans feature
    gbmcp_url: str = Field(
        default="",
        description="Streamable-HTTP MCP endpoint, e.g. http://localhost:3001/mcp",
    )

    # LLM — any OpenAI-compatible endpoint. Empty = AI analysis disabled.
    llm_base_url: str = Field(default="")
    llm_api_key: str = Field(default="")
    llm_models: str = Field(
        default="granite-4.0-h-small,granite-3.3-8b-instruct",
        description="Comma-separated model IDs to try in order (first = preferred).",
    )
    llm_timeout: int = Field(default=60)

    # gbserver database (for AI data collector — optional)
    gbserver_db_url: str = Field(default="")
    gbserver_db_schema: str = Field(default="public")

    # Optional cloud logging service (e.g. IBM Cloud Logs) — enables the Logs tab on running builds
    cloud_logs_url: str = Field(default="")
    cloud_logs_api_key: str = Field(default="")

    # S3-compatible object storage — enables the data processing pipeline page
    cos_endpoint: str = Field(default="")
    cos_access_key: str = Field(default="")
    cos_secret_key: str = Field(default="")
    cos_bucket: str = Field(default="")

    # CORS origins allowed to call this service
    cors_origins: list[str] = Field(default=["http://localhost:5173"])

    # Set to false to disable the AI analysis daemon without removing LLM credentials
    ai_analysis_enabled: bool = Field(default=True)

    # Chat assistant — a hand-rolled agentic tool-calling loop that calls
    # gbmcp's tools on the user's behalf, against either Claude directly or
    # any OpenAI-compatible model API (RITS, Ollama, etc.).
    chat_backend: str = Field(
        default="tool_loop",
        description="Which ChatAgentBackend implementation to use (see services/chat_agents/).",
    )
    # Deliberately backend-agnostic: this is a raw model identifier string
    # handed to whichever ChatAgentBackend is active. Its meaning (and
    # default, if unset) is entirely up to that backend/provider — for
    # tool_loop_backend.py's AnthropicProvider, it's an Anthropic model ID;
    # for its OpenAICompatProvider, it's an LLMClient-style model spec
    # (plain Ollama tag, or RITS's "slug:full/name" form).
    chat_model: str | None = Field(
        default=None,
        description="Model identifier passed to the active chat backend. Meaning is backend-specific.",
    )
    # Separate from AI Analysis's llm_base_url/llm_api_key by default (the
    # interactive agent may warrant a different/bigger model than the bulk
    # classification daemon), but falls back to them via the resolved_*
    # properties below so a single-model deployment doesn't need to
    # configure the same RITS/Ollama endpoint twice.
    chat_llm_base_url: str = Field(default="")
    chat_llm_api_key: str = Field(default="")

    # Explicit provider selection — see tool_loop_backend.py's _build_provider().
    # None (unset/blank) auto-detects: the OpenAI-compatible endpoint wins if
    # configured (the natural default for a self-hosted deployment — no
    # external API key, no request data leaving it), falling back to
    # Anthropic only if that's all that's configured. Setting this always
    # overrides auto-detection, and errors loudly if the selected provider's
    # own credentials aren't actually configured, rather than silently
    # falling through to the other one.
    chat_provider: str | None = Field(
        default=None,
        description="'openai_compatible' or 'anthropic' — overrides auto-detection when both are configured.",
    )

    @field_validator("chat_provider", mode="before")
    @classmethod
    def _parse_chat_provider(cls, v: object) -> object:
        if isinstance(v, str):
            if v.strip() == "":
                return (
                    None  # blank/unset k8s ConfigMap idiom — fall back to auto-detect
                )
            if v.strip() not in ("openai_compatible", "anthropic"):
                raise ValueError(
                    f"GB_UI_CHAT_PROVIDER must be 'openai_compatible' or 'anthropic', got {v!r}"
                )
            return v.strip()
        return v

    # Master on/off for the whole analytics subsystem, read from GB_UI_ANALYTICS_ENABLED.
    # Tri-state: None (unset) → auto-detect off the presence of compiled UI assets
    # (see analytics_is_enabled); explicit true/false wins. Lets a deployed, API-only
    # rest-server keep analytics off even though the gb_ui_backend package is installed.
    analytics_enabled: bool | None = Field(default=None)

    @field_validator("analytics_enabled", mode="before")
    @classmethod
    def _parse_analytics_enabled(cls, v: object) -> object:
        # Parse GB_UI_ANALYTICS_ENABLED into the tri-state without ever raising:
        # blank/whitespace (a common k8s/shell "unset" idiom) → None → auto-detect;
        # any other set value → a definite bool via the shared parser. A
        # ValidationError here would crash startup in the CLI parent and every
        # worker — the very failure this field exists to prevent.
        if isinstance(v, str):
            if v.strip() == "":
                return None
            return parse_boolean(v)
        return v

    @property
    def llm_models_list(self) -> list[str]:
        return [m.strip() for m in self.llm_models.split(",") if m.strip()]

    @property
    def ai_enabled(self) -> bool:
        return self.ai_analysis_enabled and bool(self.llm_base_url and self.llm_api_key)

    @property
    def db_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def resolved_chat_llm_base_url(self) -> str:
        return self.chat_llm_base_url or self.llm_base_url

    @property
    def resolved_chat_llm_api_key(self) -> str:
        return self.chat_llm_api_key or self.llm_api_key

    @property
    def chat_enabled(self) -> bool:
        # If neither Anthropic credential is set, any configured
        # OpenAI-compatible endpoint (RITS, Ollama, ...) also enables chat —
        # see tool_loop_backend.py's _build_provider() for the actual
        # provider-selection precedence (GB_UI_CHAT_PROVIDER if set, else
        # OpenAI-compatible wins auto-detection, falling back to Anthropic).
        return has_anthropic_chat_config() or has_openai_compat_chat_config(self)


class GitHubConfig(BaseSettings):
    """Reads GITHUB_* vars — no GB_UI_ prefix, so a separate settings class."""

    github_client_secret: str = Field(default="", alias="GITHUB_CLIENT_SECRET")

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../../.env"),
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_config() -> Config:
    return Config()


@lru_cache
def get_github_config() -> GitHubConfig:
    return GitHubConfig()


def analytics_is_enabled(ui_assets_present: bool) -> bool:
    """Resolve whether the analytics subsystem should run.

    An explicit GB_UI_ANALYTICS_ENABLED (config.analytics_enabled) always wins.
    Otherwise auto-detect: analytics runs only when the compiled frontend assets
    are present, since an API-only server has no dashboard to serve analytics to.

    ``ui_assets_present`` is passed in rather than computed here because the
    canonical UI directory (and its GBSERVER_UI_DIR override) lives in gbserver's
    root_api. This is called from both the CLI parent process and each uvicorn
    worker; both read the same inherited env and the same UI dir, so they agree.
    """
    override = get_config().analytics_enabled
    if override is not None:
        return override
    return ui_assets_present
