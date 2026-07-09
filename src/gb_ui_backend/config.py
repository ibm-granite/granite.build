"""Central configuration — all env vars for the analytics sidecar."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default SQLite filename for the sidecar's analytics database.
# Stored in ~/.granite.build/ alongside gbserver's llmb-server.db.
SIDECAR_DB_FILENAME = "dashboard-analytics.db"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GB_UI_",
        # Look for .env in the repo root (../../ relative to src/gb_ui_backend/).
        # Falls back gracefully if the file doesn't exist.
        env_file=os.path.join(os.path.dirname(__file__), "../../../.env"),
        extra="ignore",
    )

    # Sidecar analytics database — SQLite or PostgreSQL.
    # Auto-set to sqlite+aiosqlite:///<GB_HOME_DIR>/dashboard-analytics.db by gbserver if unset.
    database_url: str = Field(
        default="",
        description="SQLAlchemy async URL. Auto-configured by gbserver when running standalone.",
    )

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

    # CORS origins allowed to call this sidecar
    cors_origins: list[str] = Field(default=["http://localhost:5173"])

    # Set to false to disable the AI analysis daemon without removing LLM credentials
    ai_analysis_enabled: bool = Field(default=True)

    @property
    def llm_models_list(self) -> list[str]:
        return [m.strip() for m in self.llm_models.split(",") if m.strip()]

    @property
    def ai_enabled(self) -> bool:
        return self.ai_analysis_enabled and bool(self.llm_base_url and self.llm_api_key)

    @property
    def db_enabled(self) -> bool:
        return bool(self.database_url)


class GitHubConfig(BaseSettings):
    """Reads GITHUB_* vars — no GB_UI_ prefix, so a separate settings class."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../../.env"),
        extra="ignore",
    )
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
