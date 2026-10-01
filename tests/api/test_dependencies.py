"""Unit tests for how a route is handed the prediction service.

These tests establish the whole contract of ``get_prediction_service``: it
returns the service the running application already holds, the same one
every time it is asked, and it builds nothing. Repeated calls are what make
that a statement rather than a hope -- a provider that constructed a
service per request would pass a single call and fail here.

The dependency is called directly with a request carrying the application,
which is what FastAPI does for it, and avoids declaring a route this
milestone is not supposed to have.

Nothing here predicts, loads a model, or exercises an HTTP endpoint.
"""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from churner.api.app import create_app
from churner.api.dependencies import get_prediction_service


def request_to(application) -> Request:
    """Build the minimal request a dependency reads an application from."""
    return Request({"type": "http", "app": application, "headers": []})


# --- The dependency hands over what startup built ---


def test_the_dependency_returns_the_application_scoped_service(settings, loader):
    """What a route receives is the object sitting in application state."""
    application = create_app(settings)

    with TestClient(application):
        service = get_prediction_service(request_to(application))

        assert service is application.state.prediction_service


def test_repeated_calls_return_the_same_service(settings, loader):
    """Two requests are answered by one model, not by two copies of it."""
    application = create_app(settings)

    with TestClient(application):
        first = get_prediction_service(request_to(application))
        second = get_prediction_service(request_to(application))

    assert first is second


def test_the_dependency_does_not_load_a_model(settings, loader):
    """Asking for the service never reads the artifact again.

    The load that startup performed stays the only one, however many times
    the dependency is called.
    """
    application = create_app(settings)

    with TestClient(application):
        for _ in range(3):
            get_prediction_service(request_to(application))

    assert loader.call_count == 1


def test_each_application_hands_over_its_own_service(settings, loader):
    """The service comes from the request's application, not from a global."""
    first_application = create_app(settings)
    second_application = create_app(settings)

    with TestClient(first_application), TestClient(second_application):
        first = get_prediction_service(request_to(first_application))
        second = get_prediction_service(request_to(second_application))

    assert first is not second


# --- An application that never started has nothing to hand over ---


def test_an_unstarted_application_raises_rather_than_building_a_service(
    settings, loader
):
    """Without a lifespan there is no service, and none is improvised."""
    application = create_app(settings)

    with pytest.raises(RuntimeError, match="No prediction service"):
        get_prediction_service(request_to(application))

    assert loader.call_count == 0
