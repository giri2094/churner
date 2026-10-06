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

The one route the application declares is assembly of the same kind. It
names the customer from its path, has the validated features turned into a
frame by ``features_to_dataframe``, asks the injected service for one
answer, and reports it. Each of those steps belongs to a module that
already owns it, so the route itself holds no model, no frame building, and
no scoring, and there is nothing in it to go wrong independently of the
stages it calls.

The customer is read from the path rather than from the body, because the
address of the request already names which customer is being predicted
about; the body describes a customer without saying which one.

The service arrives through ``get_prediction_service`` even though this
module is what put it in ``app.state``. A handler reading that state
directly would be tied to how startup happens to store the service, and
would bypass the dependency override that lets a test, or another
application, supply a service of its own.

Nothing here predicts, preprocesses, or loads a model while serving a
request: the route reads what the dependency hands it, and the figures it
reports are the ones the service produced.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from churner.api.adapters import features_to_dataframe
from churner.api.dependencies import (
    PredictionServiceUnavailableError,
    get_prediction_service,
)
from churner.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from churner.config.settings import MODEL_PATH_VARIABLE, Settings, load_settings
from churner.packaging.model import load_model
from churner.serving.prediction import PredictionService

APPLICATION_TITLE = "Churner"
APPLICATION_DESCRIPTION = "Customer churn prediction service."

# --- The address a prediction is asked for at ---
# The customer is part of the path, so the request says which customer it
# is about before its body is read.
PREDICTION_PATH = "/customers/{customer_id}/prediction"

router = APIRouter()


@router.get("/health/liveness", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Report that the application process is alive."""
    return HealthResponse(status="alive")

async def prediction_service_unavailable_handler(
    request: Request,
    exc: PredictionServiceUnavailableError,
) -> JSONResponse:
    """Translate an unavailable prediction service into HTTP 503."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Prediction service unavailable"},
    )




@router.get("/health/readiness", response_model=HealthResponse)
def readiness(
    prediction_service: Annotated[
        PredictionService,
        Depends(get_prediction_service),
    ],
) -> HealthResponse:
    """Report that the prediction-serving dependency is available."""
    return HealthResponse(status="ready")


@router.post(PREDICTION_PATH, response_model=PredictionResponse)
def predict_customer_churn(
    customer_id: str,
    request: PredictionRequest,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    """Answer one prediction request about the customer the path names.

    The features are asked about as a single row, so the service returns a
    single result, and that result is the answer: the class, the
    probability, and the version are reported as it produced them rather
    than recomputed or reinterpreted here.

    Parameters
    ----------
    customer_id : str
        The customer the request is about, taken from the path. It is what
        the result is attributed to, and it is not read from the body.
    request : PredictionRequest
        The validated body, holding the customer's model features.
    prediction_service : PredictionService
        The application's service, supplied by the dependency rather than
        built here, so one model answers every request.

    Returns
    -------
    PredictionResponse
        The identifier, predicted class, probability of churn, and model
        version of the single result the service returned.
    """
    features = features_to_dataframe(request.features)
    result = prediction_service.predict([customer_id], features)[0]

    return PredictionResponse(
        customer_id=result.customer_id,
        churn_prediction=result.churn_prediction,
        churn_probability=result.churn_probability,
        model_version=result.model_version,
    )


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
        An application serving the prediction route, whose startup puts one
        ``PredictionService`` in ``app.state.prediction_service``.
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


    application = FastAPI(
        title=APPLICATION_TITLE,
        description=APPLICATION_DESCRIPTION,
        lifespan=lifespan,
    )

    application.add_exception_handler(
        PredictionServiceUnavailableError,
        prediction_service_unavailable_handler,
    )
    application.include_router(router)

    return application


app = create_app()
