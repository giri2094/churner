"""Unit tests for the serving application's startup.

These tests establish what starting the application does: it reads the
configured artifact once, builds one ``PredictionService`` over it with the
configured version, and leaves that service in application state for the
run. They also establish what building the application does not do, since
loading at import time would be indistinguishable from the outside until
an import happened somewhere no artifact exists.

The application is entered through ``TestClient``, which runs the lifespan
on the way in and out. No request is made and no route exists yet: the
client is used here only as the thing that starts and stops an application.

Which version a service carries is read off a prediction rather than out of
the service's internals, because a stamped result is the observable form of
the pairing startup made.

Nothing here tests prediction, configuration, or the dependency that hands
the service to a route.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churner.api import app as app_module
from churner.api.app import build_prediction_service, create_app
from churner.serving.prediction import PredictionService

CUSTOMER_ID = "0001-AAAAA"


@pytest.fixture
def features() -> pd.DataFrame:
    """Build one feature row, for reading a service's version off a result."""
    return pd.DataFrame({"tenure": [12.0], "MonthlyCharges": [79.95]})


# --- Building an application loads nothing ---


def test_creating_the_application_does_not_load_a_model(settings, loader):
    """An application can be built where no artifact exists.

    The model belongs to a running application rather than to an imported
    module, so construction has to stay off the filesystem.
    """
    create_app(settings)

    assert loader.call_count == 0


def test_an_unstarted_application_holds_no_prediction_service(settings, loader):
    """Nothing is in application state until the lifespan has run."""
    application = create_app(settings)

    assert not hasattr(application.state, "prediction_service")


# --- Startup loads the configured model ---


def test_startup_loads_the_configured_artifact(settings, loader):
    """The path read is the configured one, not a default or a guess."""
    with TestClient(create_app(settings)):
        pass

    assert loader.requested_paths == [settings.model_path]


def test_the_model_is_loaded_once_for_the_life_of_the_application(settings, loader):
    """Startup reads the artifact a single time, however long the run lasts."""
    application = create_app(settings)

    with TestClient(application):
        assert loader.call_count == 1

    assert loader.call_count == 1


# --- Startup builds one service and stores it ---


def test_startup_creates_a_prediction_service(settings, loader):
    """What startup builds from the loaded model is a ``PredictionService``."""
    application = create_app(settings)

    with TestClient(application):
        assert isinstance(application.state.prediction_service, PredictionService)


def test_the_service_is_stored_in_application_state(settings, loader):
    """The service lives on the application, which is what lets it be shared."""
    application = create_app(settings)

    with TestClient(application):
        assert hasattr(application.state, "prediction_service")


def test_application_state_holds_one_service_rather_than_rebuilding_it(
    settings, loader
):
    """Reading the state twice gives the same object both times."""
    application = create_app(settings)

    with TestClient(application):
        first = application.state.prediction_service
        second = application.state.prediction_service

    assert first is second


def test_the_stored_service_predicts_with_the_configured_version(
    settings, loader, features
):
    """The service pairs the loaded model with the configured version.

    The version is read off a result because that is where it is visible: a
    service built with some other version would stamp that one instead.
    """
    application = create_app(settings)

    with TestClient(application):
        results = application.state.prediction_service.predict([CUSTOMER_ID], features)

    assert len(results) == 1
    assert results[0].customer_id == CUSTOMER_ID
    assert results[0].model_version == settings.model_version


# --- Two applications are independent ---


def test_each_application_builds_its_own_service(settings, loader):
    """Services are application-scoped, so two applications do not share one."""
    first_application = create_app(settings)
    second_application = create_app(settings)

    with TestClient(first_application), TestClient(second_application):
        assert (
            first_application.state.prediction_service
            is not second_application.state.prediction_service
        )

    assert loader.call_count == 2


# --- A model that cannot be loaded stops the application ---


def test_startup_fails_when_the_artifact_cannot_be_loaded(settings, failing_loader):
    """An application with no model does not start and serve anyway."""
    with pytest.raises(RuntimeError, match="Could not load the model artifact"):
        with TestClient(create_app(settings)):
            pass


def test_the_startup_failure_names_the_path_and_keeps_its_cause(
    settings, failing_loader
):
    """The report says which artifact was expected and what went wrong.

    A deployment acts on the path it configured, so the path is in the
    message, and the original error stays attached rather than being
    replaced by a summary of it.
    """
    with pytest.raises(RuntimeError) as failure:
        build_prediction_service(settings)

    assert str(settings.model_path) in str(failure.value)
    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_an_unreadable_artifact_fails_startup_as_a_missing_one_does(
    settings, monkeypatch
):
    """A file that is not a usable artifact is the same fatal condition.

    Deserialisation failures do not arrive as ``FileNotFoundError``, and an
    application that survived one would be running without a model.
    """

    def raise_deserialisation_error(path):
        raise ValueError("could not deserialise the artifact")

    monkeypatch.setattr(app_module, "load_model", raise_deserialisation_error)

    with pytest.raises(RuntimeError, match="Could not load the model artifact"):
        build_prediction_service(settings)
