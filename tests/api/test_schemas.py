"""Unit tests for the shape of a prediction request and of the answer to one.

These tests establish the public contract of the API schemas: which fields
a caller sends, which it is sent back, what type each one carries, and
which values are refused. That is the whole of what the schemas promise,
and it is the whole of what is checked here.

Two of those promises are deliberate permissiveness rather than oversight,
so they are stated as tests of their own. An unknown category is accepted
because the fitted ``OneHotEncoder`` is configured with
``handle_unknown="ignore"`` and already has defined behaviour for one, and
a missing ``TotalCharges`` is accepted because the pipeline's median
imputer is what stands in for it. Both reasons live downstream, and neither
is exercised here: these tests check that the boundary lets such a request
through, not what the model then makes of it.

The accepted feature fields are compared against
``churner.schema.features`` rather than against a list written out again
here. The schemas restate that canonical schema by hand, so a column added
to one and not the other is the drift this suite is positioned to catch.

``customer_id`` is checked for its absence from the request body, because
the route names the customer in its path --
``POST /customers/{customer_id}/prediction`` -- and a body field would be a
second, contradictable statement of the same thing.

Nothing here declares or calls a route, builds a prediction service, loads
a model, or transforms a feature frame.
"""

import pytest
from pydantic import ValidationError

from churner.api.schemas import CustomerFeatures, PredictionRequest, PredictionResponse
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# --- The features a request has to carry ---
# The canonical schema, in the order it states them, so the cases below are
# parametrised over the columns the pipeline actually selects.
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- A complete, ordinary feature set ---
# Values a customer in the Telco dataset plausibly holds: a new subscriber
# on a month-to-month DSL contract. Nothing here is meant to be a hard
# case; it is the baseline the awkward cases below are varied from.
VALID_FEATURE_PAYLOAD = {
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

# --- Categories the training data does not hold ---
# A service, a contract length, and a payment method that no fitted encoder
# has seen. Each names a plausible future offering rather than nonsense,
# which is the case this permissiveness exists for.
UNKNOWN_CATEGORIES = {
    "InternetService": "Satellite",
    "Contract": "Three year",
    "PaymentMethod": "Cryptocurrency",
    "gender": "Prefer not to say",
    "StreamingTV": "Streaming TV Plus",
}

# --- A complete response ---
# The probability sits away from 0.50 and from the two bounds, so a value
# that was rounded, inverted, or replaced is a different number.
RESPONSE_PAYLOAD = {
    "customer_id": "7590-VHVEG",
    "churn_prediction": "Yes",
    "churn_probability": 0.73,
    "model_version": "churn-logistic-2026-10-01",
}

RESPONSE_FIELDS = (
    "customer_id",
    "churn_prediction",
    "churn_probability",
    "model_version",
)


def feature_payload(**overrides) -> dict:
    """Build a complete feature payload, with the given fields replaced."""
    return {**VALID_FEATURE_PAYLOAD, **overrides}


def feature_payload_without(field_name: str) -> dict:
    """Build a feature payload that states every field but one."""
    payload = feature_payload()
    del payload[field_name]
    return payload


def request_body(**overrides) -> dict:
    """Build a request body holding a complete, ordinary feature set."""
    return {"features": feature_payload(**overrides)}


def error_locations(validation_error: ValidationError) -> set[tuple]:
    """Collect the field paths a validation error blames, as tuples."""
    return {tuple(error["loc"]) for error in validation_error.errors()}


# --- A complete feature set is accepted ---


def test_a_complete_feature_set_is_accepted():
    """All 19 features, stated together, validate into one object."""
    features = CustomerFeatures(**feature_payload())

    assert features.tenure == 1
    assert features.MonthlyCharges == pytest.approx(29.85)
    assert features.TotalCharges == pytest.approx(29.85)
    assert features.Contract == "Month-to-month"
    assert features.PaymentMethod == "Electronic check"


def test_the_accepted_fields_are_the_canonical_model_features():
    """The request's features are exactly the columns the project models on.

    The schemas restate ``churner.schema.features`` by hand, so this is
    what fails when a column is added to one of the two and not the other.
    """
    assert tuple(CustomerFeatures.model_fields) == MODEL_FEATURES
    assert len(MODEL_FEATURES) == 19


@pytest.mark.parametrize("excluded_column", ["customerID", "Churn"])
def test_the_identifier_and_the_target_are_not_feature_fields(excluded_column):
    """Neither the customer's id nor the outcome is something a caller sends."""
    assert excluded_column not in CustomerFeatures.model_fields


# --- The numerical features arrive as numbers ---


def test_the_numerical_features_arrive_as_the_types_they_are_modelled_as():
    """Tenure is a whole number of months; both charges are amounts."""
    features = CustomerFeatures(**feature_payload())

    assert type(features.tenure) is int
    assert type(features.MonthlyCharges) is float
    assert type(features.TotalCharges) is float


def test_a_whole_number_charge_is_still_read_as_an_amount():
    """JSON writes 70 for 70.00, and the field holds it as a float.

    Without this the two charge columns would be handed on as integers
    whenever a caller happened to send a round figure.
    """
    features = CustomerFeatures(**feature_payload(MonthlyCharges=70, TotalCharges=840))

    assert type(features.MonthlyCharges) is float
    assert type(features.TotalCharges) is float
    assert features.MonthlyCharges == pytest.approx(70.0)


@pytest.mark.parametrize("not_a_number", ["twenty", "", None])
def test_a_monthly_charge_that_is_not_an_amount_fails_validation(not_a_number):
    """An unreadable charge is refused rather than passed to the model."""
    with pytest.raises(ValidationError) as refusal:
        CustomerFeatures(**feature_payload(MonthlyCharges=not_a_number))

    assert ("MonthlyCharges",) in error_locations(refusal.value)


# --- A customer with no accumulated charges ---


def test_an_absent_total_charges_is_accepted():
    """``TotalCharges`` may be null, because the pipeline imputes it.

    The dataset documents the column as blank for customers who have
    accumulated nothing yet, and the median imputer is what stands in for
    that, so the boundary has to be able to express the absence.
    """
    features = CustomerFeatures(**feature_payload(TotalCharges=None))

    assert features.TotalCharges is None


def test_an_absent_total_charges_is_still_a_field_that_has_to_be_stated():
    """Null is a value the field accepts, not a default it falls back to."""
    with pytest.raises(ValidationError) as refusal:
        CustomerFeatures(**feature_payload_without("TotalCharges"))

    assert ("TotalCharges",) in error_locations(refusal.value)


# --- SeniorCitizen is the category the raw file records as 0 and 1 ---


@pytest.mark.parametrize("recorded_value", [0, 1])
def test_senior_citizen_accepts_the_integers_it_is_recorded_as(recorded_value):
    """Both values the dataset holds are accepted, as integers."""
    features = CustomerFeatures(**feature_payload(SeniorCitizen=recorded_value))

    assert features.SeniorCitizen == recorded_value
    assert type(features.SeniorCitizen) is int


# --- Categories the model was not fitted on ---


@pytest.mark.parametrize(
    ("field_name", "unknown_value"),
    sorted(UNKNOWN_CATEGORIES.items()),
)
def test_an_unknown_category_is_accepted(field_name, unknown_value):
    """A value no encoder has seen passes the boundary unchanged.

    ``OneHotEncoder(handle_unknown="ignore")`` already defines what the
    model does with one, so refusing it here would reject a request the
    model is able to answer.
    """
    features = CustomerFeatures(**feature_payload(**{field_name: unknown_value}))

    assert getattr(features, field_name) == unknown_value


def test_every_textual_categorical_field_accepts_an_unseen_value():
    """No categorical field is restricted to the categories it was fitted on.

    ``SeniorCitizen`` is left out, since it is recorded as a number and has
    a case of its own above.
    """
    unseen_value = "Not a category in the dataset"
    unseen_categories = {
        field_name: unseen_value
        for field_name in CATEGORICAL_FEATURES
        if field_name != "SeniorCitizen"
    }

    features = CustomerFeatures(**feature_payload(**unseen_categories))

    assert [getattr(features, field_name) for field_name in unseen_categories] == [
        unseen_value
    ] * len(unseen_categories)


# --- Every feature has to be stated ---


@pytest.mark.parametrize("omitted_feature", MODEL_FEATURES)
def test_omitting_any_feature_fails_validation(omitted_feature):
    """A feature left out is a missing column, and is refused as one.

    Every one of the 19 is required, so a payload that states 18 of them
    fails naming the one it left out rather than being filled in.
    """
    with pytest.raises(ValidationError) as refusal:
        CustomerFeatures(**feature_payload_without(omitted_feature))

    assert (omitted_feature,) in error_locations(refusal.value)


def test_an_empty_feature_payload_names_every_missing_feature():
    """Nothing is optional, so nothing at all fails on all 19 at once."""
    with pytest.raises(ValidationError) as refusal:
        CustomerFeatures()

    assert error_locations(refusal.value) == {(name,) for name in MODEL_FEATURES}


# --- The request body holds features, and only features ---


def test_a_request_body_holding_a_feature_object_is_accepted():
    """A body nesting the features under ``features`` validates."""
    request = PredictionRequest.model_validate(request_body())

    assert isinstance(request.features, CustomerFeatures)
    assert request.features.tenure == 1
    assert request.features.Contract == "Month-to-month"


def test_a_request_accepts_an_already_built_feature_object():
    """The nested field takes a ``CustomerFeatures``, not only a mapping."""
    features = CustomerFeatures(**feature_payload())

    request = PredictionRequest(features=features)

    assert request.features == features


def test_the_features_are_the_only_thing_a_request_body_states():
    """``features`` is the whole of the body's contract."""
    assert tuple(PredictionRequest.model_fields) == ("features",)


def test_the_customer_id_is_not_part_of_the_request_body():
    """The identifier belongs to the route's path, not to the JSON body.

    ``POST /customers/{customer_id}/prediction`` already names the
    customer, so a body field would be a second statement of it that a
    caller could contradict.
    """
    request = PredictionRequest.model_validate(request_body())

    assert "customer_id" not in PredictionRequest.model_fields
    assert "customer_id" not in request.model_dump()


def test_a_body_without_features_fails_validation():
    """An empty body states no customer to predict about, and is refused."""
    with pytest.raises(ValidationError) as refusal:
        PredictionRequest.model_validate({})

    assert ("features",) in error_locations(refusal.value)


def test_an_incomplete_nested_feature_set_fails_naming_the_feature():
    """A fault inside the nested object is reported at its own path.

    The nesting is what makes the location two levels deep, which is how a
    caller is told which feature of which object was missing.
    """
    incomplete_body = {"features": feature_payload_without("tenure")}

    with pytest.raises(ValidationError) as refusal:
        PredictionRequest.model_validate(incomplete_body)

    assert ("features", "tenure") in error_locations(refusal.value)


# --- The response names the customer, the class, the score, and the model ---


def test_a_complete_response_is_accepted():
    """All four fields, stated together, validate into one answer."""
    response = PredictionResponse(**RESPONSE_PAYLOAD)

    assert response.customer_id == "7590-VHVEG"
    assert response.churn_prediction == "Yes"
    assert response.churn_probability == pytest.approx(0.73)
    assert response.model_version == "churn-logistic-2026-10-01"


def test_the_response_states_the_four_fields_the_contract_names():
    """Nothing more and nothing less comes back from a prediction."""
    assert tuple(PredictionResponse.model_fields) == RESPONSE_FIELDS


def test_the_response_fields_keep_the_types_they_are_reported_in():
    """The class is a label, the probability an amount, both identifiers text.

    The class stays a string rather than becoming a boolean, so the
    response reports the label the model was fitted on instead of a reading
    of it.
    """
    response = PredictionResponse(**RESPONSE_PAYLOAD)

    assert type(response.customer_id) is str
    assert type(response.churn_prediction) is str
    assert type(response.churn_probability) is float
    assert type(response.model_version) is str


@pytest.mark.parametrize("omitted_field", RESPONSE_FIELDS)
def test_omitting_any_response_field_fails_validation(omitted_field):
    """An answer is not partial: a prediction without its version, or a
    version without its prediction, is not a response.
    """
    payload = {
        name: value for name, value in RESPONSE_PAYLOAD.items() if name != omitted_field
    }

    with pytest.raises(ValidationError) as refusal:
        PredictionResponse(**payload)

    assert (omitted_field,) in error_locations(refusal.value)
