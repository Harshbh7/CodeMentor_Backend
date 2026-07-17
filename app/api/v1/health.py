"""
CodeMentor AI - Health Check Router
======================================
Provides a /health endpoint for load balancers, Docker health checks,
and monitoring systems (e.g., Kubernetes liveness/readiness probes).

Design Rationale:
- Health endpoints must be fast and non-blocking.
- We check each dependency independently to give fine-grained status.
- Returns HTTP 200 even when degraded — the `status` field indicates health.
  (Some load balancers only look at HTTP status, not body.)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import check_db_connection, get_db
from app.core.logging import get_logger
from app.schemas.common import HealthCheckResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Application Health Check",
    description=(
        "Returns the health status of the application and all connected services. "
        "Used by load balancers, container orchestrators, and monitoring systems."
    ),
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthCheckResponse:
    """
    Perform a health check on the application and its dependencies.

    Checks:
    - PostgreSQL database connectivity
    - Valkey connectivity (optional caching)
    - ChromaDB Cloud connectivity
    """
    services: dict[str, str] = {}

    # ----- PostgreSQL -----
    try:
        db_healthy = await check_db_connection()
        services["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as exc:
        logger.warning("Health check: database error: %s", exc)
        services["database"] = "unhealthy"

    # ----- Valkey (optional) -----
    try:
        from app.core.cache import get_valkey_client
        vk = get_valkey_client()
        if vk:
            vk.ping()
            services["valkey"] = "healthy"
        else:
            services["valkey"] = "not_configured"
    except Exception:
        services["valkey"] = "not_configured"

    # ----- ChromaDB -----
    try:
        from app.rag.vector_store import VectorStore
        store = VectorStore()
        client = store._get_client()
        client.list_collections()
        services["chromadb"] = "healthy"
    except Exception as exc:
        logger.warning("Health check: ChromaDB error: %s", exc)
        services["chromadb"] = "unhealthy"

    # Overall status: healthy only if ALL critical services are healthy
    critical_services = ["database", "chromadb"]
    overall_status = (
        "healthy"
        if all(services.get(s) == "healthy" for s in critical_services)
        else "degraded"
    )

    logger.info("Health check completed: status=%s", overall_status)

    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.app_env,
        services=services,
    )
