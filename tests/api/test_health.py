import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from churner.api import app as app_module
from churner.api.dependencies import (
    PredictionServiceUnavailableError,
    get_prediction_service,
)
from churner.api.schemas import HealthResponse


def test_health_response_accepts_alive():
    response = HealthResponse(status="alive")

    assert response.status == "alive"


def test_health_response_accepts_ready():
    response = HealthResponse(status="ready")

    assert response.status == "ready"


def test_health_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        HealthResponse(status="healthy")


def test_liveness_endpoint_returns_alive(client):
    response = client.get("/health/liveness")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint_returns_ready(client):
    response = client.get("/health/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_endpoint_uses_prediction_service_dependency(settings, loader):
    application = app_module.create_app(settings)

    dependency_called = False

    def dependency():
        nonlocal dependency_called
        dependency_called = True
        return object()

    application.dependency_overrides[get_prediction_service] = dependency

    with TestClient(application) as client:
        response = client.get("/health/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert dependency_called


def test_readiness_endpoint_returns_503_when_dependency_fails(settings, loader):
    application = app_module.create_app(settings)

    def failing_dependency():
        raise PredictionServiceUnavailableError(
            "Prediction service unavailable"
        )

    application.dependency_overrides[get_prediction_service] = failing_dependency

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/health/readiness")

    assert response.status_code == 503
