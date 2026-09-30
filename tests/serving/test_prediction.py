"""Unit tests for predicting churn for named customers from a fitted model.

These tests establish the public contract of ``PredictionService``: it pairs
identifiers with feature rows by position, reports the probability of
``"Yes"`` as the fitted model orders its classes, stamps every result with
the version it was constructed with, and does all of that without fitting
anything.

Most cases use a stand-in estimator rather than a trained pipeline, because
what is under test is which numbers the service reads and where it puts
them, and a stand-in is the only way to state those numbers in advance. The
class ordering is part of what the stand-in supplies: the same scores are
predicted through a model that lists ``"Yes"`` last and one that lists it
first, so a column index assumed rather than looked up fails one of them.

One case uses a genuine fitted ``Pipeline``, to show the service works
against a real estimator's ``classes_`` and not only a hand-made one.

Nothing here trains, evaluates, or loads a model, and nothing here tests
preprocessing: the stand-in ignores the frame it is handed, and the fitted
pipeline is a bare classifier fitted on two numeric columns.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churner.serving.prediction import PredictionResult, PredictionService

# --- The version the service is constructed with ---
# A value with no meaning to the service, so a result carrying it can only
# have copied it rather than derived it from the model.
MODEL_VERSION = "churn-logistic-2026-09-30"
OTHER_MODEL_VERSION = "churn-tree-2026-10-01"

# --- Class orderings a fitted model may present ---
# sklearn sorts labels, so "No", "Yes" is the ordering a Yes/No model
# ordinarily has. The reversed ordering is not reachable from those two
# labels, but it is reachable from a model fitted on differently spelled
# ones, and the service is not supposed to depend on either.
YES_LAST = ["No", "Yes"]
YES_FIRST = ["Yes", "No"]

# Labels of a model that does not know the Yes/No vocabulary at all.
CLASSES_WITHOUT_YES = ["Churned", "Retained"]

# --- Controlled single-customer case ---
SINGLE_CUSTOMER_ID = "0001-AAAAA"
SINGLE_PREDICTION = "Yes"
SINGLE_POSITIVE_SCORE = 0.73

# --- Controlled four-customer case ---
# Identifiers are deliberately not in sorted order, so results returned in
# input order can be told apart from results returned sorted by id. Every
# score sits away from 0.50, so reading the wrong probability column gives a
# different number rather than the same one.
CUSTOMER_IDS = ["0003-CCCCC", "0001-AAAAA", "0004-DDDDD", "0002-BBBBB"]
PREDICTIONS = ["Yes", "No", "Yes", "No"]
POSITIVE_SCORES = [0.91, 0.12, 0.68, 0.34]

# What reading column 1 regardless of ``classes_`` would report for the
# four customers above when the model lists "Yes" first. Stated here so the
# wrong-column outcome is visible rather than implied.
COMPLEMENT_SCORES = [0.09, 0.88, 0.32, 0.66]


def probability_matrix(positive_scores, classes) -> np.ndarray:
    """Lay out P(No) and P(Yes) columns in the order ``classes`` names them.

    This is what makes a class ordering testable: the same scores can be
    presented by a model that lists ``"Yes"`` last and by one that lists it
    first, and only the column positions differ.
    """
    yes_scores = np.asarray(positive_scores, dtype=float)
    columns = {"Yes": yes_scores, "No": 1.0 - yes_scores}
    return np.column_stack([columns[label] for label in classes])


class StubClassifier:
    """A stand-in estimator returning predetermined labels and probabilities.

    ``classes_`` is supplied by the test and ``predict_proba`` returns its
    columns in that same order, so a test decides where ``"Yes"`` sits.
    ``fit`` is present only so that a call to it can be detected; predicting
    is not supposed to make one.
    """

    def __init__(self, classes, predictions, probabilities):
        self.classes_ = np.array(classes)
        self._predictions = np.array(predictions)
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.fit_call_count = 0

    def fit(self, X, y=None):
        self.fit_call_count += 1
        return self

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        return self._probabilities


def stub_service(classes, predictions, positive_scores, model_version=MODEL_VERSION):
    """Build a service over a stand-in that reports exactly these figures."""
    model = StubClassifier(
        classes, predictions, probability_matrix(positive_scores, classes)
    )
    return PredictionService(model, model_version)


@pytest.fixture
def single_customer_features() -> pd.DataFrame:
    """Build the one feature row of the single-customer case."""
    return pd.DataFrame({"tenure": [12.0], "MonthlyCharges": [79.95]})


@pytest.fixture
def features() -> pd.DataFrame:
    """Build four feature rows, one per customer of the controlled case."""
    return pd.DataFrame(
        {
            "tenure": [1.0, 60.0, 4.0, 33.0],
            "MonthlyCharges": [85.10, 25.40, 99.65, 55.00],
        }
    )


@pytest.fixture
def fitted_pipeline() -> Pipeline:
    """Fit a bare classifier pipeline on two numeric columns.

    No project preprocessing is involved: this exists to supply a real
    estimator's ``classes_`` and probability layout, not to exercise a
    pipeline that another module's tests already cover.
    """
    training_features = pd.DataFrame(
        {
            "tenure": [1.0, 2.0, 5.0, 40.0, 55.0, 70.0],
            "MonthlyCharges": [95.0, 90.0, 88.0, 30.0, 25.0, 20.0],
        }
    )
    training_churn = ["Yes", "Yes", "Yes", "No", "No", "No"]
    return Pipeline([("classifier", LogisticRegression())]).fit(
        training_features, training_churn
    )


# --- One customer ---


def test_single_customer_yields_exactly_one_result(single_customer_features):
    """One identifier and one feature row produce one result."""
    service = stub_service(YES_LAST, [SINGLE_PREDICTION], [SINGLE_POSITIVE_SCORE])

    results = service.predict([SINGLE_CUSTOMER_ID], single_customer_features)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], PredictionResult)


def test_single_result_holds_the_id_class_probability_and_version(
    single_customer_features,
):
    """All four fields are filled from what the service was given."""
    service = stub_service(YES_LAST, [SINGLE_PREDICTION], [SINGLE_POSITIVE_SCORE])

    result = service.predict([SINGLE_CUSTOMER_ID], single_customer_features)[0]

    assert result.customer_id == SINGLE_CUSTOMER_ID
    assert result.churn_prediction == SINGLE_PREDICTION
    assert result.churn_probability == pytest.approx(SINGLE_POSITIVE_SCORE)
    assert result.model_version == MODEL_VERSION


def test_single_result_reports_plain_python_types(single_customer_features):
    """The class is a ``str`` and the probability a ``float``, not numpy scalars."""
    service = stub_service(YES_LAST, [SINGLE_PREDICTION], [SINGLE_POSITIVE_SCORE])

    result = service.predict([SINGLE_CUSTOMER_ID], single_customer_features)[0]

    assert type(result.churn_prediction) is str
    assert type(result.churn_probability) is float


# --- Order of the results ---


def test_results_are_returned_in_input_order(features):
    """Four rows come back as four results, in the order they were given."""
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    results = service.predict(CUSTOMER_IDS, features)

    assert len(results) == len(CUSTOMER_IDS)
    assert [result.customer_id for result in results] == CUSTOMER_IDS


def test_each_result_keeps_its_own_customers_prediction(features):
    """A customer's class and probability stay with that customer's id.

    The identifiers are unsorted and every score differs, so a result set
    that was reordered or shifted by one would not match this triple.
    """
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    results = service.predict(CUSTOMER_IDS, features)

    assert [
        (result.customer_id, result.churn_prediction, result.churn_probability)
        for result in results
    ] == [
        (customer_id, prediction, pytest.approx(score))
        for customer_id, prediction, score in zip(
            CUSTOMER_IDS, PREDICTIONS, POSITIVE_SCORES
        )
    ]


def test_results_follow_row_order_rather_than_the_frames_index(features):
    """Pairing is positional, so a non-default frame index changes nothing."""
    reindexed_features = features.set_axis([40, 10, 30, 20])
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    results = service.predict(CUSTOMER_IDS, reindexed_features)

    assert [result.customer_id for result in results] == CUSTOMER_IDS
    assert [result.churn_probability for result in results] == pytest.approx(
        POSITIVE_SCORES
    )


# --- Which probability column is reported ---


@pytest.mark.parametrize("classes", [YES_LAST, YES_FIRST], ids=["yes_last", "yes_first"])
def test_probability_is_the_one_the_model_assigned_to_yes(classes, features):
    """The reported score follows ``classes_``, whichever column holds "Yes".

    Both orderings describe the same four customers with the same scores.
    Only the column positions differ, so a service that read a fixed column
    would agree with one ordering and contradict the other.
    """
    service = stub_service(classes, PREDICTIONS, POSITIVE_SCORES)

    results = service.predict(CUSTOMER_IDS, features)

    assert [result.churn_probability for result in results] == pytest.approx(
        POSITIVE_SCORES
    )


def test_probability_is_not_the_complement_when_yes_comes_first(features):
    """Reading column 1 from a "Yes"-first model would report P(No) instead.

    The complement scores below are what that mistake produces. Naming them
    is what distinguishes a looked-up column from a lucky one.
    """
    service = stub_service(YES_FIRST, PREDICTIONS, POSITIVE_SCORES)

    reported_scores = [
        result.churn_probability for result in service.predict(CUSTOMER_IDS, features)
    ]

    assert reported_scores == pytest.approx(POSITIVE_SCORES)
    assert reported_scores != pytest.approx(COMPLEMENT_SCORES)


def test_probability_of_yes_is_read_from_a_real_fitted_pipeline(
    fitted_pipeline, features
):
    """A genuine estimator's own P(Yes) column is what the service reports."""
    yes_column = list(fitted_pipeline.classes_).index("Yes")
    expected_scores = fitted_pipeline.predict_proba(features)[:, yes_column]
    expected_classes = list(fitted_pipeline.predict(features))
    service = PredictionService(fitted_pipeline, MODEL_VERSION)

    results = service.predict(CUSTOMER_IDS, features)

    assert [result.churn_prediction for result in results] == expected_classes
    assert [result.churn_probability for result in results] == pytest.approx(
        expected_scores
    )


# --- The version the results carry ---


def test_every_result_carries_the_supplied_model_version(features):
    """The version is copied onto each result, unchanged."""
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    results = service.predict(CUSTOMER_IDS, features)

    assert [result.model_version for result in results] == [MODEL_VERSION] * len(
        CUSTOMER_IDS
    )


def test_two_services_stamp_their_own_versions(features):
    """The version comes from the service that predicted, not from a default."""
    first = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES, MODEL_VERSION)
    second = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES, OTHER_MODEL_VERSION)

    assert first.predict(CUSTOMER_IDS, features)[0].model_version == MODEL_VERSION
    assert (
        second.predict(CUSTOMER_IDS, features)[0].model_version == OTHER_MODEL_VERSION
    )


# --- Identifiers and rows have to line up ---


def test_fewer_ids_than_feature_rows_raises_value_error(features):
    """Two identifiers cannot be attributed to four rows."""
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    with pytest.raises(ValueError, match="customer ids for"):
        service.predict(CUSTOMER_IDS[:2], features)


def test_more_ids_than_feature_rows_raises_value_error(single_customer_features):
    """Four identifiers cannot be attributed to one row either."""
    service = stub_service(YES_LAST, [SINGLE_PREDICTION], [SINGLE_POSITIVE_SCORE])

    with pytest.raises(ValueError, match="customer ids for"):
        service.predict(CUSTOMER_IDS, single_customer_features)


def test_no_ids_for_populated_feature_rows_raises_value_error(features):
    """An empty identifier list against four rows is the same failure."""
    service = stub_service(YES_LAST, PREDICTIONS, POSITIVE_SCORES)

    with pytest.raises(ValueError, match="customer ids for"):
        service.predict([], features)


# --- The fitted model is predicted with, never fitted ---


def test_constructing_the_service_does_not_fit_the_model():
    """Construction binds the model and the version; it trains nothing."""
    model = StubClassifier(
        YES_LAST, PREDICTIONS, probability_matrix(POSITIVE_SCORES, YES_LAST)
    )

    PredictionService(model, MODEL_VERSION)

    assert model.fit_call_count == 0


def test_predicting_does_not_fit_the_model(features):
    """Predicting is inference only: the model's ``fit`` is never called."""
    model = StubClassifier(
        YES_LAST, PREDICTIONS, probability_matrix(POSITIVE_SCORES, YES_LAST)
    )
    service = PredictionService(model, MODEL_VERSION)

    service.predict(CUSTOMER_IDS, features)
    service.predict(CUSTOMER_IDS, features)

    assert model.fit_call_count == 0


def test_predicting_through_a_fitted_pipeline_does_not_refit_it(
    fitted_pipeline, features
):
    """A real pipeline is not refitted either, not even on the frame it scores.

    Its ``fit`` is replaced rather than counted, so a refit fails the test
    at the moment it happens.
    """
    original_coefficients = fitted_pipeline.named_steps["classifier"].coef_.copy()

    def refuse_fit(*args, **kwargs):
        raise AssertionError("PredictionService must not fit the model it was given")

    fitted_pipeline.fit = refuse_fit
    PredictionService(fitted_pipeline, MODEL_VERSION).predict(CUSTOMER_IDS, features)

    assert np.array_equal(
        fitted_pipeline.named_steps["classifier"].coef_, original_coefficients
    )


# --- A returned result cannot be edited ---


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("customer_id", "9999-ZZZZZ"),
        ("churn_prediction", "No"),
        ("churn_probability", 0.01),
        ("model_version", OTHER_MODEL_VERSION),
    ],
)
def test_a_returned_result_refuses_field_assignment(
    field_name, new_value, single_customer_features
):
    """Each field of a returned result is frozen against rewriting."""
    service = stub_service(YES_LAST, [SINGLE_PREDICTION], [SINGLE_POSITIVE_SCORE])
    result = service.predict([SINGLE_CUSTOMER_ID], single_customer_features)[0]

    with pytest.raises(FrozenInstanceError):
        setattr(result, field_name, new_value)

    assert getattr(result, field_name) != new_value


# --- A model that has no "Yes" class ---


def test_a_model_without_a_yes_class_raises_value_error(features):
    """Prediction fails rather than reporting some other class's probability.

    The stand-in is built directly, since ``probability_matrix`` cannot lay
    out labels it has no column for. Its probabilities are ones a service
    reading a fixed column would have returned without complaint, and no
    result escapes at all.
    """
    model = StubClassifier(
        CLASSES_WITHOUT_YES,
        ["Churned", "Retained", "Churned", "Retained"],
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
    )
    service = PredictionService(model, MODEL_VERSION)

    with pytest.raises(ValueError, match="Yes"):
        service.predict(CUSTOMER_IDS, features)
