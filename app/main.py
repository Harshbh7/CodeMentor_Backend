"""
CodeMentor AI - FastAPI Application Entry Point
===============================================
Creates and configures the FastAPI application instance.

Design Rationale:
- Using `lifespan` (context manager) instead of deprecated `@app.on_event`.
- Middleware is added in the correct order (outer-most added first executes last).
- Exception handlers map custom exceptions to structured JSON responses.
- OpenAPI docs are disabled in production for security.
- CORS is configured to allow development frontend origins.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import close_db_connection, check_db_connection
from app.core.exceptions import CodeMentorException
from app.core.logging import get_logger, setup_logging

# ==============================================================
# Settings & Logger
# ==============================================================

settings = get_settings()
# Setup logging immediately so it's ready before the app starts
setup_logging()
logger = get_logger(__name__)


# ==============================================================
# Application Lifespan (Startup & Shutdown)
# ==============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages application startup and shutdown events.

    Startup:
    - Verify environment variables loaded correctly
    - Confirm database connectivity
    - Create upload directory if missing

    Shutdown:
    - Gracefully close DB connection pool
    """
    # ----- STARTUP -----
    logger.info("=" * 60)
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)
    logger.info("=" * 60)

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info("Upload directory ensured: %s", settings.upload_dir)

    # Ensure ChromaDB data directory exists
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    logger.info("ChromaDB persist directory ensured: %s", settings.chroma_persist_dir)

    # Verify database connectivity at startup
    db_connected = await check_db_connection()
    if not db_connected:
        logger.error(
            "STARTUP FAILED: Cannot connect to PostgreSQL at %s:%s/%s",
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
        )
        # We don't crash here — let health endpoint report degraded status
        # In production you might want to raise SystemExit(1)
    else:
        logger.info("PostgreSQL connection established.")

    logger.info("API available at: http://%s:%s%s", settings.host, settings.port, settings.api_prefix)
    logger.info(
        "Swagger UI: http://%s:%s/docs",
        settings.host if settings.host != "0.0.0.0" else "127.0.0.1",
        settings.port,
    )

    yield  # Application is now running

    # ----- SHUTDOWN -----
    logger.info("Shutting down %s...", settings.app_name)
    await close_db_connection()
    logger.info("Shutdown complete.")


# ==============================================================
# Application Factory
# ==============================================================

def create_application() -> FastAPI:
    """
    Application factory function — creates and configures the FastAPI app.

    Using a factory pattern allows:
    - Easy testing (create a test app instance with overridden settings)
    - Clean separation of configuration from instantiation
    """

    # Disable auto-generated OpenAPI docs in production (security)
    docs_url = "/docs" if not settings.is_production else None
    redoc_url = "/redoc" if not settings.is_production else None
    openapi_url = "/openapi.json" if not settings.is_production else None

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "CodeMentor AI is an AI-powered coding assistant with Agentic RAG. "
            "It helps developers with code generation, debugging, explanations, "
            "complexity analysis, and personalized learning roadmaps."
        ),
        openapi_tags=[
            {"name": "Health", "description": "Application health and readiness checks"},
            {"name": "Authentication", "description": "User registration, login, and token management"},
            {"name": "Chat", "description": "AI-powered coding assistant chat interface"},
            {"name": "Knowledge Base", "description": "Upload and manage programming documentation"},
        ],
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # ----------------------------------------------------------
    # Middleware (applied outer-to-inner)
    # ----------------------------------------------------------

    # CORS — Allow cross-origin requests from the frontend
    # In production, restrict `allow_origins` to your specific frontend domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------
    # Exception Handlers
    # ----------------------------------------------------------

    @app.exception_handler(CodeMentorException)
    async def codementor_exception_handler(
        request: Request, exc: CodeMentorException
    ) -> JSONResponse:
        """
        Maps all custom CodeMentorException subclasses to structured JSON responses.
        This keeps error handling out of route handlers.
        """
        logger.warning(
            "Application exception: %s | path=%s",
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for unexpected exceptions.
        Prevents raw tracebacks from leaking to clients in production.
        """
        logger.exception(
            "Unhandled exception: %s | path=%s | method=%s",
            str(exc),
            request.url.path,
            request.method,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "An unexpected error occurred. Please try again later.",
                "details": {},
            },
        )

    # ----------------------------------------------------------
    # Routers
    # ----------------------------------------------------------

    app.include_router(api_router, prefix=settings.api_prefix)

    # Mount the frontend static files (served at /ui/*)
    _frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.isdir(_frontend_dir):
        app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

    # Root redirect → frontend UI
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui/")

    return app


# ==============================================================
# Application Instance
# ==============================================================

app = create_application()
