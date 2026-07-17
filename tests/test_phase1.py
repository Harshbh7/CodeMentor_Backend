"""
CodeMentor AI - Phase 1 Integration Test
==========================================
Tests the core application setup:
- FastAPI app startup
- Health check endpoint
- Configuration loading
- Database engine creation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.core.config import get_settings


@pytest.fixture
def client():
    """Synchronous TestClient for simple endpoint testing."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestConfiguration:
    """Tests for Settings loading."""

    def test_settings_loads_correctly(self):
        """Settings should load without raising ValidationError."""
        settings = get_settings()
        assert settings.app_name == "CodeMentor AI"
        assert settings.api_prefix == "/api/v1"
        assert settings.database_url.startswith("postgresql+asyncpg://")

    def test_computed_database_url(self):
        """DATABASE_URL should be correctly computed from parts."""
        settings = get_settings()
        assert settings.postgres_user in settings.database_url
        assert settings.postgres_db in settings.database_url

    def test_computed_valkey_url(self):
        """Valkey URL should be computed correctly."""
        settings = get_settings()
        assert settings.valkey_host in settings.valkey_url
        assert str(settings.valkey_port) in settings.valkey_url


class TestRootEndpoint:
    """Tests for the root redirect endpoint."""

    def test_root_returns_app_info(self, client: TestClient):
        """Root endpoint should redirect to /ui/."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert "/ui/" in response.headers["location"]


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    @patch("app.api.v1.health.check_db_connection", new_callable=AsyncMock)
    def test_health_check_returns_200(self, mock_db_check, client: TestClient):
        """Health endpoint should always return 200."""
        mock_db_check.return_value = True
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    @patch("app.api.v1.health.check_db_connection", new_callable=AsyncMock)
    def test_health_check_healthy_status(self, mock_db_check, client: TestClient):
        """When DB is healthy, status should be 'healthy'."""
        mock_db_check.return_value = True
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["database"] == "healthy"

    @patch("app.api.v1.health.check_db_connection", new_callable=AsyncMock)
    def test_health_check_degraded_status(self, mock_db_check, client: TestClient):
        """When DB is down, status should be 'degraded'."""
        mock_db_check.return_value = False
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["database"] == "unhealthy"

    @patch("app.api.v1.health.check_db_connection", new_callable=AsyncMock)
    def test_health_response_schema(self, mock_db_check, client: TestClient):
        """Health response should match the expected schema."""
        mock_db_check.return_value = True
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "services" in data


class TestExceptionHandlers:
    """Tests for custom exception handling."""

    def test_not_found_returns_json(self, client: TestClient):
        """404 for unknown routes should return JSON (not HTML)."""
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
