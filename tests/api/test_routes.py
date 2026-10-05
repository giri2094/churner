"""Unit tests for the prediction route, from request to response.

These tests establish what ``POST /customers/{customer_id}/prediction``
does with a request: it validates the body, asks the application's
prediction service for one answer about the customer the path names, and
reports that answer. What is under test is the orchestration -- which
object is asked, what it is handed, and what comes back out as JSON -- and
not what any of the stages it orchestrates do internally.

The service is replaced at the dependency boundary. ``get_prediction_service``
is overridden with one that records the arguments it was called with and
returns a predetermined result, which is what makes "exactly once", "this
customer", and "one row of these columns" observable. The replacement is
also what keeps these tests off a real model: nothing is fitted, nothing is
transformed, and no artifact beyond the stand-in the suite's ``loader``
fixture supplies is read.

Overriding the dependency is itself one of the assertions. A route that
reached into application state, or built a service of its own, would answer
with the version the stand-in artifact was configured with rather than the
one the recording service reports, so the version in the response is what
distinguishes the two.

The customer is named twice in every case -- once in the path and once in
what the recording service is asked -- and never in the body. Two
different paths are exercised for that reason: an answer that echoed a
fixed identifier, or one read from somewhere other than the path, agrees
with one of them and contradicts the other.

Nothing here loads a real artifact, fits or calls sklearn, checks
preprocessing, or inspects how the request body becomes a frame. The frame
is examined only for the shape the serving layer is promised: one row, the
canonical columns, in canonical order.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churner.api.app import create_app
from churner.api.dependencies import get_prediction_service
from churner.api.schemas import PredictionResponse
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from churner.serving.prediction import PredictionResult

# --- The features a frame has to carry, in the order it has to carry them ---
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# The one category the raw file records as a number rather than as text.
SENIOR_CITIZEN = "SeniorCitizen"

# --- The customer a request is about ---
# Named in the path and nowhere else. The second identifier exists so that
# two requests can be told apart by the answer they receive.
CUSTOMER_ID = "7590-VHVEG"
OTHER_CUSTOMER_ID = "3668-QPYBK"

# --- The body every case is built from ---
# Amounts that are distinct and unrounded, and categories each carrying a
# value that names their own column, so a frame whose columns were shuffled
# or shifted fails on the values as well as on the header.
NUMERICAL_VALUES = {
    "tenure": 9,
    "MonthlyCharges": 65.25,
    "TotalCharges": 593.30,
}

CATEGORICAL_VALUES = {
    field_name: 1 if field_name == SENIOR_CITIZEN else f"{field_name}-value"
    for field_name in CATEGORICAL_FEATURES
}

FEATURE_VALUES = {**NUMERICAL_VALUES, **CATEGORICAL_VALUES}

# --- What the recording service answers with ---
# None of these can be derived from the request, so a response carrying
# them can only have come from the service the dependency handed over. The
# version in particular differs from the one the suite's stand-in artifact
# is configured with, which is how a route that built its own service would
# be caught.
PREDICTED_CLASS = "Yes"
PREDICTED_PROBABILITY = 0.73
RECORDED_MODEL_VERSION = "churn-recorded-2026-10-05"

RESPONSE_FIELDS = {
    "customer_id",
    "churn_prediction",
    "churn_probability",
    "model_version",
}

UNPROCESSABLE_ENTITY = 422


class RecordingPredictionService:
    """A stand-in service that records what it was asked and answers fixed.

    Only ``predict`` is present, that being the whole of what a route uses.
    The identifiers it is given are echoed back in the results, as the real
    service does, so a response can be traced to the path it came from
    rather than to a value written into this class.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def predict(self, customer_ids, features) -> list[PredictionResult]:
        self.calls.append((customer_ids, features))
        return [
            PredictionResult(
                customer_id=customer_id,
                churn_prediction=PREDICTED_CLASS,
                churn_probability=PREDICTED_PROBABILITY,
                model_version=RECORDED_MODEL_VERSION,
            )
            for customer_id in customer_ids
        ]

    @property
    def call_count(self) -> int:
        """How many times a prediction was asked for."""
        return len(self.calls)

    @property
    def last_customer_ids(self):
        """The identifiers the most recent call was given."""
        return self.calls[-1][0]

    @property
    def last_features(self) -> pd.DataFrame:
        """The feature frame the most recent call was given."""
        return self.calls[-1][1]


def prediction_url(customer_id: str) -> str:
    """Address the prediction of one customer, as the contract spells it."""
    return f"/customers/{customer_id}/prediction"


def feature_values(**overrides) -> dict:
    """State every feature of the one customer, with the given ones replaced."""
    return {**FEATURE_VALUES, **overrides}


def request_body(**overrides) -> dict:
    """Nest a complete feature set under ``features``, as a body does."""
    return {"features": feature_values(**overrides)}


@pytest.fixture
def service() -> RecordingPredictionService:
    """Build the service the route will be handed instead of a real one."""
    return RecordingPredictionService()


@pytest.fixture
def client(settings, loader, service) -> TestClient:
    """Serve an application whose prediction service is the recording one.

    The ``loader`` fixture keeps startup off the filesystem, and the
    override is what puts the recording service where the dependency looks.
    """
    application = create_app(settings)
    application.dependency_overrides[get_prediction_service] = lambda: service

    with TestClient(application) as test_client:
        yield test_client


# --- A valid request is answered ---


def test_a_valid_request_is_answered_with_200(client):
    """A complete body for a named customer is a request the route serves."""
    response = client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert response.status_code == 200


def test_the_response_states_the_four_fields_the_contract_names(client):
    """An answer carries the customer, the class, the score, and the model."""
    response = client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert set(response.json()) == RESPONSE_FIELDS


def test_the_response_validates_as_a_prediction_response(client):
    """What is sent back is the response schema, not a shape resembling it."""
    response = client.post(prediction_url(CUSTOMER_ID), json=request_body())

    answer = PredictionResponse(**response.json())

    assert answer.churn_prediction == PREDICTED_CLASS
    assert answer.churn_probability == pytest.approx(PREDICTED_PROBABILITY)


# --- The customer comes from the path ---


@pytest.mark.parametrize("customer_id", [CUSTOMER_ID, OTHER_CUSTOMER_ID])
def test_the_answer_names_the_customer_the_path_named(client, customer_id):
    """The identifier in the response is the one in the URL.

    Two paths are exercised, so a response echoing a fixed identifier
    cannot satisfy both.
    """
    response = client.post(prediction_url(customer_id), json=request_body())

    assert response.json()["customer_id"] == customer_id


@pytest.mark.parametrize("customer_id", [CUSTOMER_ID, OTHER_CUSTOMER_ID])
def test_the_path_identifier_is_what_the_service_is_asked_about(
    client, service, customer_id
):
    """The path's customer is the one a prediction is requested for."""
    client.post(prediction_url(customer_id), json=request_body())

    assert list(service.last_customer_ids) == [customer_id]


def test_the_body_does_not_have_to_name_the_customer(client, service):
    """A body of features alone is enough, the path having named the customer."""
    response = client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert response.status_code == 200
    assert "customer_id" not in request_body()
    assert list(service.last_customer_ids) == [CUSTOMER_ID]


# --- The service the dependency hands over is the one used ---


def test_the_route_predicts_through_the_injected_service(client, service):
    """The overridden dependency is what answers, so that is how a route asks.

    A route reading application state directly, or constructing a service
    of its own, would leave this recording service untouched.
    """
    client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert service.call_count == 1


def test_the_answer_comes_from_the_injected_service(client, settings):
    """The version reported is the injected service's, not startup's.

    The stand-in artifact is configured with a different version, so this
    distinguishes an answer that came through the dependency from one
    produced by a service the route built for itself.
    """
    response = client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert response.json()["model_version"] == RECORDED_MODEL_VERSION
    assert response.json()["model_version"] != settings.model_version


def test_one_request_asks_for_exactly_one_prediction(client, service):
    """Serving a request calls ``predict`` once, not twice and not per field."""
    client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert service.call_count == 1


def test_each_request_asks_once_more(client, service):
    """Three requests are three predictions, each about its own customer."""
    for customer_id in (CUSTOMER_ID, OTHER_CUSTOMER_ID, CUSTOMER_ID):
        client.post(prediction_url(customer_id), json=request_body())

    assert service.call_count == 3
    assert [list(ids) for ids, _ in service.calls] == [
        [CUSTOMER_ID],
        [OTHER_CUSTOMER_ID],
        [CUSTOMER_ID],
    ]


def test_the_route_does_not_load_a_model_of_its_own(client, loader):
    """Serving requests reads no artifact; startup's single load is the only one."""
    for _ in range(3):
        client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert loader.call_count == 1


# --- What the service is handed ---


def test_the_service_is_given_one_identifier_for_one_customer(client, service):
    """Identifiers arrive as a one-element sequence, not as a bare string."""
    client.post(prediction_url(CUSTOMER_ID), json=request_body())

    customer_ids = service.last_customer_ids

    assert not isinstance(customer_ids, str)
    assert len(customer_ids) == 1
    assert list(customer_ids) == [CUSTOMER_ID]


def test_the_service_is_given_a_one_row_dataframe(client, service):
    """One customer is asked about with one row of features."""
    client.post(prediction_url(CUSTOMER_ID), json=request_body())

    features = service.last_features

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1


def test_the_frame_holds_the_canonical_features_in_canonical_order(client, service):
    """The columns are ``churner.schema.features``, in the order it states them."""
    client.post(prediction_url(CUSTOMER_ID), json=request_body())

    assert tuple(service.last_features.columns) == MODEL_FEATURES


def test_the_frame_holds_the_values_the_body_nested_under_features(client, service):
    """What was sent under ``features`` is what the service is asked about.

    A handful of columns is enough to establish that the nested object was
    read and placed; which value lands in which column is the adapter's
    contract and is tested where the adapter is.
    """
    client.post(
        prediction_url(CUSTOMER_ID),
        json=request_body(tenure=42, MonthlyCharges=99.95),
    )

    row = service.last_features.iloc[0]

    assert row["tenure"] == 42
    assert row["MonthlyCharges"] == pytest.approx(99.95)
    assert row["Contract"] == FEATURE_VALUES["Contract"]


# --- A request that cannot be served is refused before the service is reached ---


def test_a_body_without_features_is_unprocessable(client):
    """An empty body states no customer to predict about."""
    response = client.post(prediction_url(CUSTOMER_ID), json={})

    assert response.status_code == UNPROCESSABLE_ENTITY


@pytest.mark.parametrize("omitted_feature", MODEL_FEATURES)
def test_a_body_missing_any_feature_is_unprocessable(client, omitted_feature):
    """Each of the 19 features is required, and a body of 18 is refused."""
    incomplete_features = {
        field_name: value
        for field_name, value in feature_values().items()
        if field_name != omitted_feature
    }

    response = client.post(
        prediction_url(CUSTOMER_ID), json={"features": incomplete_features}
    )

    assert response.status_code == UNPROCESSABLE_ENTITY


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("tenure", "not a number of months"),
        ("MonthlyCharges", "not an amount"),
        ("TotalCharges", "not an amount"),
        (SENIOR_CITIZEN, "not a flag"),
    ],
)
def test_a_feature_of_the_wrong_type_is_unprocessable(
    client, field_name, invalid_value
):
    """A value that is not of the contracted type is refused."""
    response = client.post(
        prediction_url(CUSTOMER_ID),
        json=request_body(**{field_name: invalid_value}),
    )

    assert response.status_code == UNPROCESSABLE_ENTITY


def test_an_unservable_request_never_reaches_the_service(client, service):
    """Validation happens before a prediction is asked for.

    A model is not consulted about a request that was never valid, so the
    recording service stays untouched by all three refusals.
    """
    client.post(prediction_url(CUSTOMER_ID), json={})
    client.post(prediction_url(CUSTOMER_ID), json=request_body(tenure="nine months"))
    client.post(
        prediction_url(CUSTOMER_ID),
        json={"features": {"tenure": FEATURE_VALUES["tenure"]}},
    )

    assert service.call_count == 0
