"""
CodeMentor AI - Application Configuration
==========================================
Uses Pydantic BaseSettings for type-safe, validated configuration loading.
All settings are sourced from environment variables or the .env file.

Design Rationale:
- Centralized config avoids magic strings scattered across the codebase.
- Pydantic validates types at startup — fail fast, not at runtime.
- The @lru_cache pattern ensures settings are loaded once (singleton).
- Computed fields (DATABASE_URL) are derived from parts to keep .env clean.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.
    Pydantic will raise a ValidationError at startup if any required
    variable is missing or has an incorrect type.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars silently
    )

    # ----------------------------------------------------------
    # Application
    # ----------------------------------------------------------
    app_name: str = Field(default="CodeMentor AI", description="Application name")
    app_version: str = Field(default="1.0.0", description="Semantic version")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment — controls behavior like debug mode",
    )
    debug: bool = Field(default=False, description="Enable FastAPI debug mode")
    api_prefix: str = Field(default="/api/v1", description="Global API route prefix")

    # ----------------------------------------------------------
    # Server
    # ----------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    port: int = Field(default=8000, description="Uvicorn bind port")
    workers: int = Field(default=1, description="Number of Uvicorn worker processes")

    # ----------------------------------------------------------
    # PostgreSQL
    # ----------------------------------------------------------
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(..., description="PostgreSQL database name")

    @computed_field  # type: ignore[misc]
    @property
    def database_url(self) -> str:
        """
        Async-compatible PostgreSQL URL using asyncpg driver.
        We use the asyncpg driver so SQLAlchemy never blocks the event loop.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def sync_database_url(self) -> str:
        """
        Synchronous URL for Alembic migrations.
        Alembic runs in a regular (non-async) context during migration commands.
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ----------------------------------------------------------
    # JWT
    # ----------------------------------------------------------
    jwt_secret_key: str = Field(..., description="Secret key for signing JWT tokens")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=60, description="Access token TTL in minutes"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, description="Refresh token TTL in days"
    )

    # ----------------------------------------------------------
    # Google Gemini
    # ----------------------------------------------------------
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-1.5-pro", description="Gemini model for chat completions"
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Gemini model for generating text embeddings",
    )

    # ----------------------------------------------------------
    # ChromaDB
    # ----------------------------------------------------------
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8001, description="ChromaDB port")
    chroma_ssl: bool = Field(default=False, description="Enable SSL/HTTPS for ChromaDB client")
    chroma_persist_dir: str = Field(
        default="./chroma_data",
        description="Local persistence directory for ChromaDB when not using HTTP mode",
    )
    chroma_api_key: str | None = Field(default=None, description="ChromaDB Cloud API key")
    chroma_tenant: str | None = Field(default=None, description="ChromaDB Cloud tenant")
    chroma_database: str | None = Field(default=None, description="ChromaDB Cloud database")

    # ----------------------------------------------------------
    # Valkey
    # ----------------------------------------------------------
    valkey_host: str = Field(default="localhost", description="Valkey host")
    valkey_port: int = Field(default=6379, description="Valkey port")
    valkey_password: str = Field(default="", description="Valkey password (empty = none)")
    valkey_db: int = Field(default=0, description="Valkey database number")

    @computed_field  # type: ignore[misc]
    @property
    def valkey_url(self) -> str:
        """Valkey connection URL."""
        if self.valkey_password:
            return f"valkey://:{self.valkey_password}@{self.valkey_host}:{self.valkey_port}/{self.valkey_db}"
        return f"valkey://{self.valkey_host}:{self.valkey_port}/{self.valkey_db}"

    @property
    def redis_url(self) -> str:
        """Compatibility property mapping redis_url to valkey_url (replacing redis protocol)."""
        return self.valkey_url.replace("valkey://", "redis://")

    @property
    def redis_host(self) -> str:
        return self.valkey_host

    @property
    def redis_port(self) -> int:
        return self.valkey_port

    # ----------------------------------------------------------
    # File Upload
    # ----------------------------------------------------------
    max_upload_size_mb: int = Field(
        default=50, description="Maximum allowed file upload size in megabytes"
    )
    upload_dir: str = Field(default="./uploads", description="Directory to store uploads")

    @computed_field  # type: ignore[misc]
    @property
    def max_upload_size_bytes(self) -> int:
        """Computed max upload size in bytes for validation."""
        return self.max_upload_size_mb * 1024 * 1024

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log output format"
    )

    # ----------------------------------------------------------
    # CORS Settings
    # ----------------------------------------------------------
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or * for all",
    )


    # ----------------------------------------------------------
    # Derived Helpers
    # ----------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Returns True if running in production environment."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Returns True if running in development environment."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.

    Using @lru_cache means the .env file is read exactly once at startup.
    FastAPI dependency injection uses this function via Depends(get_settings).

    Usage:
        from app.core.config import get_settings
        settings = get_settings()

    In FastAPI endpoints:
        from fastapi import Depends
        from app.core.config import Settings, get_settings

        @router.get("/info")
        def info(settings: Settings = Depends(get_settings)):
            return {"app": settings.app_name}
    """
    return Settings()
