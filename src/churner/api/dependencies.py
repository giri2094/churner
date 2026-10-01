"""How a route reaches the application's prediction service.

A route needs the service that startup built, and this is the whole of how
it asks for one. The dependency looks the service up; it does not make one.
A provider that constructed a service would read the artifact again on
every request and would answer two requests with two models during a
deployment, which is the opposite of what loading once at startup is for.

Looking it up through the request rather than importing the application
keeps the lookup bound to whichever application is serving the request, so
a test application and a deployed one each hand over their own service
without this module knowing either exists.

Nothing here predicts, loads a model, reads configuration, or decides what
a caller may ask for.
"""

from fastapi import Request

from churner.serving.prediction import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """Return the prediction service the running application holds.

    The same service is returned for every request, because it is the one
    object startup put in application state.

    Parameters
    ----------
    request : Request
        The request being served, read only for the application it belongs
        to.

    Returns
    -------
    PredictionService
        The application-scoped service, built once during startup.

    Raises
    ------
    RuntimeError
        If application state holds no service. Startup always puts one
        there, so this means the application is being used without its
        lifespan having run rather than that a model is missing.
    """
    try:
        return request.app.state.prediction_service
    except AttributeError as error:
        raise RuntimeError(
            "No prediction service in application state. One is created "
            "during lifespan startup, so reaching this without one means "
            "the application was not started through its lifespan."
        ) from error
