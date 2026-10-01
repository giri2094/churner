"""The serving application, and the startup that gives it a model.

This module assembles what the other layers already provide: configuration
says which artifact to serve and what to call it, ``load_model`` reads that
artifact, and ``PredictionService`` turns it into answers. Assembly is all
that happens here.

The model is read once, during the application's lifespan startup, and the
service built from it is kept in ``app.state`` for as long as the process
runs. Reading per request would pay a deserialisation cost on every call
and would let a mid-deployment replacement of the file change what a single
request is answered by. Reading at import time instead of at startup would
tie the model to whoever imports the module, including a test collector,
and would leave a failure to load happening outside the lifecycle a server
reports on.

Failing to load is fatal here rather than deferred. An application that
started without a model would accept requests it cannot answer and would
report that as a per-request fault, which is a deployment error wearing the
costume of a runtime one. The error is raised during startup, naming the
path it tried and the variable that sets it, so the server stops with the
reason visible.

Nothing here predicts, validates a request, shapes a response, or declares
a route. The service this module stores is the whole of what routes get,
and they reach it through the dependency rather than through this module.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from churner.config.settings import MODEL_PATH_VARIABLE, Settings, load_settings
from churner.packaging.model import load_model
from churner.serving.prediction import PredictionService

APPLICATION_TITLE = "Churner"
APPLICATION_DESCRIPTION = "Customer churn prediction service."


def build_prediction_service(settings: Settings) -> PredictionService:
    """Load the configured artifact and bind it to the configured version.

    Parameters
    ----------
    settings : Settings
        The artifact to read and the version to stamp results with.

    Returns
    -------
    PredictionService
        A service over the fitted pipeline the artifact holds.

    Raises
    ------
    RuntimeError
        If the artifact cannot be read. Every reason for that -- a path
        pointing nowhere, a file that is not an artifact, one written by an
        incompatible environment -- leaves the application unable to serve,
        so they are reported as the one failure a deployment has to act on,
        with the original error kept as its cause.
    """
    try:
        fitted_model = load_model(settings.model_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not load the model artifact at {settings.model_path}, so "
            "the application has no model to serve. Set "
            f"{MODEL_PATH_VARIABLE} to an artifact written by save_model."
        ) from error

    return PredictionService(fitted_model, settings.model_version)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the serving application, configured but not yet started.

    Building the application loads nothing: the model is read when the
    returned application is started, which is what lets a caller construct
    one without an artifact on disk.

    Parameters
    ----------
    settings : Settings | None
        The configuration to run under. Defaults to what the environment
        says, which is what a deployment uses; a caller passes settings to
        run an application under a configuration of its own.

    Returns
    -------
    FastAPI
        An application whose startup puts one ``PredictionService`` in
        ``app.state.prediction_service``.
    """
    application_settings = settings if settings is not None else load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Hold one prediction service for as long as the application runs."""
        application.state.prediction_service = build_prediction_service(
            application_settings
        )

        yield

        # Shutdown. The service owns nothing outside the process -- no
        # connection, no file handle, no thread -- so there is nothing to
        # release, and this is where a resource that did would be released.

    return FastAPI(
        title=APPLICATION_TITLE,
        description=APPLICATION_DESCRIPTION,
        lifespan=lifespan,
    )


app = create_app()
