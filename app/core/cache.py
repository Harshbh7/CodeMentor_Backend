"""
CodeMentor AI - Valkey Caching Manager
======================================
Provides helper functions to cache AI responses in Valkey
to reduce external API costs and latency.
"""

import hashlib
import json
from typing import Any
import valkey
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global Valkey connection instance
_valkey_client: valkey.Valkey | None = None


def get_valkey_client() -> valkey.Valkey | None:
    """Lazily initialize and return the Valkey client."""
    global _valkey_client
    if _valkey_client is None:
        try:
            _valkey_client = valkey.Valkey(
                host=settings.valkey_host,
                port=settings.valkey_port,
                password=settings.valkey_password or None,
                db=settings.valkey_db,
                socket_timeout=1.0,
                decode_responses=True,  # Automatically decode bytes to strings
            )
            # Ping to verify connection
            _valkey_client.ping()
            logger.info(
                "Valkey client connected successfully: %s:%s",
                settings.valkey_host,
                settings.valkey_port,
            )
        except Exception as exc:
            logger.warning("Failed to connect to Valkey cache: %s. Caching disabled.", exc)
            _valkey_client = None
    return _valkey_client


class ValkeyCache:
    """
    Manages key-value caching inside Valkey.
    Fails gracefully if Valkey is down.
    """

    @staticmethod
    def generate_key(query: str, prefix: str = "chat") -> str:
        """Generate a unique cache key based on the hashed query string."""
        normalized = query.strip().lower()
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"codementor:{prefix}:{query_hash}"

    @classmethod
    def get(cls, key: str) -> dict[str, Any] | None:
        """Retrieve a cached value from Valkey by key."""
        client = get_valkey_client()
        if not client:
            return None
        try:
            data = client.get(key)
            if data:
                logger.info("Cache HIT for key: %s", key)
                return json.loads(data)
        except Exception as exc:
            logger.warning("Valkey cache read error: %s", exc)
        return None

    @classmethod
    def set(cls, key: str, value: dict[str, Any], expire_seconds: int = 3600) -> None:
        """Store a value in Valkey with a TTL expiration."""
        client = get_valkey_client()
        if not client:
            return
        try:
            client.set(
                name=key,
                value=json.dumps(value),
                ex=expire_seconds,
            )
            logger.info("Cache SET for key: %s (TTL=%ds)", key, expire_seconds)
        except Exception as exc:
            logger.warning("Valkey cache write error: %s", exc)
