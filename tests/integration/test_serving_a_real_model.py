"""Integration test for serving a real artifact through the real endpoint.

Every other test of the serving layer replaces something. The route tests
override the prediction service at the dependency boundary, and the startup
tests replace the loader with a recorder returning a stand-in pipeline, so
that what each of them examines is the orchestration and not the stages it
orchestrates. That leaves one statement none of them makes: that the
pieces fit together. A frame built from a request the project's own schema
validated is a frame the project's own fitted pipeline accepts, and the
answer it produces satisfies the response contract.

So nothing is replaced here. The pipeline is fitted by ``train_logistic``,
written by ``save_model``, read by the application's own ``load_model``
during a lifespan ``TestClient`` runs, and asked for a prediction through
an HTTP request to the declared route. A break anywhere along that path --
a feature the schema spells one way and the pipeline selects by another, an
artifact the loader cannot read, a probability the response model refuses
-- surfaces here rather than at a deployment.

The artifact is written into ``tmp_path`` and nowhere else. A test reading
``models/churn_model.joblib`` would pass or fail on whether somebody had
run the training script, which is not a property of the code, and writing
there would leave a model behind in the checkout.

The training population is the controlled six customers the packaging and
training tests each fit on, written out here as they write it out there. It
is small enough to fit a pipeline in a moment and it holds every one of the
19 canonical features, which is what the fitted pipeline has to accept. It
is restated rather than imported because a fixture in another test module
is that module's own, and reaching into one would couple this test to a
population it does not control.

No predicted class is asserted. What six customers teach a logistic model
is not a contract: a different sklearn version, or one changed value in the
fixture, could move the answer without anything being wrong. What is
asserted is that an answer came back, that it is about the customer the
path named, that it is stamped with the version the deployment configured,
and that the class and the probability are values the vocabulary and the
unit interval admit.

Nothing here tests what the model predicts, how good it is, how the request
becomes a frame, how long the service lives, or what happens when the
artifact cannot be read.
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churner.api.app import create_app
from churner.api.schemas import PredictionResponse
from churner.config.settings import Settings
from churner.packaging.model import save_model
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from churner.training.train import train_logistic

FEATURE_COLUMNS = list(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)

# --- What the deployment under test is configured with ---
# A temporary artifact, and a version no other test uses, so the stamp on
# the answer can only have come from these settings.
ARTIFACT_NAME = "churn_model.joblib"
MODEL_VERSION = "churn-integration-2026-10-12"

# --- The customer the request is about ---
# Named in the path, as the contract spells it, and nowhere in the body.
CUSTOMER_ID = "7590-VHVEG"
PREDICTION_PATH = f"/customers/{CUSTOMER_ID}/prediction"

# The vocabulary the data dictionary documents ``Churn`` in, which is what
# a pipeline fitted on these labels predicts in.
CHURN_LABELS = {"No", "Yes"}

# --- Controlled training population ---
# Six customers in the shape ``prepare_modeling_data`` hands on, holding
# every canonical feature, with the target beside them as the separate
# series a classifier is fitted against. Nothing is calculated off it here;
# it exists to give the pipeline something real to learn from.
TRAINING_CUSTOMERS = {
    "tenure": [1, 5, 9, 13, 34, 72],
    "MonthlyCharges": [20.05, 42.30, 55.00, 70.00, 89.10, 105.50],
    "TotalCharges": [20.05, 211.50, 500.00, 900.00, 3028.50, 7590.75],
    "gender": ["Female", "Male", "Male", "Female", "Male", "Female"],
    "SeniorCitizen": [0, 0, 1, 0, 1, 0],
    "Partner": ["No", "Yes", "No", "Yes", "No", "Yes"],
    "Dependents": ["No", "No", "Yes", "No", "Yes", "Yes"],
    "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes", "Yes"],
    "InternetService": ["DSL", "DSL", "Fiber optic", "Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No", "No", "Yes", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No", "Yes", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "Yes", "No", "Yes", "No internet service"],
    "TechSupport": ["No", "No", "Yes", "No", "Yes", "No internet service"],
    "StreamingTV": ["No", "No", "Yes", "Yes", "No", "No internet service"],
    "StreamingMovies": ["No", "Yes", "Yes", "No", "No", "No internet service"],
    "Contract": [
        "Month-to-month",
        "Month-to-month",
        "One year",
        "Month-to-month",
        "Two year",
        "Two year",
    ],
    "PaperlessBilling": ["Yes", "No", "Yes", "Yes", "No", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Electronic check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Mailed check",
    ],
}

TRAINING_CHURN = ["Yes", "Yes", "No", "Yes", "No", "No"]

# --- The body the request carries ---
# All 19 features of one customer who was not in the training population,
# under the dataset's own field names, nested under ``features`` as
# ``PredictionRequest`` declares. Each category is a value the training
# population also holds, so the request asks about a customer the fitted
# encoder has categories for rather than exercising ``handle_unknown``.
REQUEST_BODY = {
    "features": {
        "tenure": 3,
        "MonthlyCharges": 45.55,
        "TotalCharges": 136.65,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
    }
}


@pytest.fixture
def artifact_path(tmp_path) -> Path:
    """Fit the project's pipeline on the population above and persist it.

    The fit and the write are the real ones, so what the application later
    reads is an artifact of the kind a training run produces. It goes into
    the temporary directory pytest supplies, which is what keeps the test
    off any model the checkout happens to contain.
    """
    training_features = pd.DataFrame(TRAINING_CUSTOMERS)[FEATURE_COLUMNS]
    training_churn = pd.Series(TRAINING_CHURN, name="Churn")

    destination = tmp_path / ARTIFACT_NAME
    save_model(train_logistic(training_features, training_churn), destination)

    return destination


@pytest.fixture
def client(artifact_path) -> TestClient:
    """Serve a real application configured to load the temporary artifact.

    Entering the client runs the lifespan, so the model is loaded and the
    prediction service built by the application's own startup. No loader is
    patched and no dependency is overridden.
    """
    settings = Settings(model_path=artifact_path, model_version=MODEL_VERSION)

    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_a_real_fitted_pipeline_answers_through_the_prediction_endpoint(client):
    """A request is served end to end by the real model the artifact holds.

    The answer is checked against the contract rather than against a
    figure: the fields ``PredictionResponse`` declares, the customer the
    path named, the version the settings configured, a class from the
    Yes/No vocabulary, and a probability in the unit interval.
    """
    response = client.post(PREDICTION_PATH, json=REQUEST_BODY)

    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == set(PredictionResponse.model_fields)

    answer = PredictionResponse(**payload)
    assert answer.customer_id == CUSTOMER_ID
    assert answer.model_version == MODEL_VERSION
    assert answer.churn_prediction in CHURN_LABELS
    assert 0.0 <= answer.churn_probability <= 1.0
