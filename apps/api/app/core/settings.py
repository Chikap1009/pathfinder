"""Typed application settings — Pydantic v2 + pydantic-settings.

All env values are validated on startup; missing required keys fail fast.
Read order: process env > .env (repo root) > defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Top-level application settings.

    Field naming mirrors `.env.example`. Use SCREAMING_SNAKE in env, the model
    is case-insensitive on read.
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:3000"

    # --- Qdrant ---
    qdrant_url: HttpUrl = Field(default=HttpUrl("http://localhost:6333"))
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "profiles_v1"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("pathfinder")
    neo4j_database: str = "neo4j"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"

    # --- LiteLLM proxy ---
    litellm_master_key: SecretStr = SecretStr("sk-pathfinder-local-dev")
    litellm_proxy_url: str = "http://localhost:4000"

    # --- Provider keys (optional — guarded individually at call site) ---
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    hf_home: Path = Field(default_factory=lambda: REPO_ROOT / "hf_cache")

    # --- Observability ---
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: HttpUrl = Field(default=HttpUrl("https://cloud.langfuse.com"))
    otel_exporter_otlp_endpoint: HttpUrl | None = None
    otel_service_name: str = "pathfinder-api"

    # --- Frontend (read-back so /openapi reflects defaults) ---
    next_public_api_base_url: HttpUrl = Field(default=HttpUrl("http://localhost:8000"))
    next_public_app_name: str = "PathFinder"

    # --- Derived helpers ---
    @field_validator("api_cors_origins")
    @classmethod
    def _split_origins(cls, v: str) -> str:
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — call once per process."""
    return Settings()
