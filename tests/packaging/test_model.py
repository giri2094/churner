"""Unit tests for saving a fitted model pipeline and loading it back.

These tests establish the public contract of ``save_model`` and
``load_model``: saving writes an artifact where it was asked to, loading
returns the fitted pipeline that was written, and a round trip changes
neither the predictions the model makes nor the model that was handed over.

The pipelines under test are produced by the real trainers rather than by
hand-built stand-ins, because what has to survive serialisation is a
genuinely fitted pipeline: a preprocessor that learned medians, categories,
and scales, and a classifier fitted on what it produced. Both baselines are
put through the round trip, since packaging is not supposed to know which
estimator it is carrying.

Predictions are the evidence of a faithful round trip rather than an
equality check on the objects, because two fitted pipelines have no
meaningful ``==`` and comparing their internals would test sklearn's
attributes rather than this module's contract. The training population is
written out here, as it is in the training tests, so a fit that learned
from different records fails rather than agrees with itself.

Nothing here trains for its own sake, evaluates a model, selects one, or
decides where an artifact belongs: every path is a temporary one supplied
by the test.
"""

import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from churner.packaging.model import load_model, save_model
from churner.training.train import train_logistic, train_tree

# --- The structure a round-tripped pipeline is expected to keep ---
# Restated here rather than imported, so a loaded object that lost its
# preprocessing fails instead of agreeing with whatever it was built from.
PREPROCESSOR_STEP = "preprocessor"
CLASSIFIER_STEP = "classifier"
NUMERICAL_BRANCH = "numerical"
IMPUTER_STEP = "imputer"

TRAINERS = [train_logistic, train_tree]

ARTIFACT_NAME = "model.joblib"

NUMERICAL_FEATURE_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERICAL_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

# --- Controlled training population ---
# The same six customers the training tests fit on, holding the shape
# ``prepare_modeling_data`` hands on. Every amount is distinct and none is
# missing, so each column's median is fixed by the fixture alone:
#
#   column           sorted middle pair      median
#   tenure           9 and 13                 11.00
#   MonthlyCharges   55.00 and 70.00          62.50
#   TotalCharges     500.00 and 900.00       700.00
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

CHURN_LABELS = {"No", "Yes"}

TRAINING_MEDIANS = {"tenure": 11.00, "MonthlyCharges": 62.50, "TotalCharges": 700.00}

# --- Controlled population a round-tripped pipeline is asked to predict for ---
# Two customers who were not in the training fixture, holding only values
# the training fixture also holds.
HELD_OUT_CUSTOMERS = {
    "tenure": [3, 60],
    "MonthlyCharges": [45.55, 95.25],
    "TotalCharges": [136.65, 5715.00],
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "Yes"],
    "MultipleLines": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic"],
    "OnlineSecurity": ["Yes", "No"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "TechSupport": ["Yes", "No"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Mailed check", "Electronic check"],
}


@pytest.fixture
def training_df() -> pd.DataFrame:
    """Build the controlled six-customer training population described above."""
    return pd.DataFrame(TRAINING_CUSTOMERS)[FEATURE_COLUMNS]


@pytest.fixture
def training_churn() -> pd.Series:
    """Build the churn labels of the training population, as a separate series."""
    return pd.Series(TRAINING_CHURN, name="Churn")


@pytest.fixture
def held_out_df() -> pd.DataFrame:
    """Build the two customers a round-tripped pipeline is asked to predict for."""
    return pd.DataFrame(HELD_OUT_CUSTOMERS)[FEATURE_COLUMNS]


@pytest.fixture
def fitted_pipeline(training_df, training_churn) -> Pipeline:
    """Fit one baseline pipeline, for the cases that need only a fitted model."""
    return train_logistic(training_df, training_churn)


def classifier_of(pipeline: Pipeline):
    """Return the estimator a pipeline ends in."""
    return pipeline.named_steps[CLASSIFIER_STEP]


def preprocessor_of(pipeline: Pipeline):
    """Return the transformer a pipeline runs its features through."""
    return pipeline.named_steps[PREPROCESSOR_STEP]


def numerical_imputer_of(pipeline: Pipeline):
    """Return the fitted numerical imputer, whether or not it sits in a branch pipeline."""
    numerical = preprocessor_of(pipeline).named_transformers_[NUMERICAL_BRANCH]
    if isinstance(numerical, Pipeline):
        return numerical.named_steps[IMPUTER_STEP]
    return numerical


# --- Saving writes the artifact that was asked for ---


def test_save_creates_the_requested_artifact(fitted_pipeline, tmp_path):
    """The file appears at the path given, and nowhere else."""
    artifact_path = tmp_path / ARTIFACT_NAME

    save_model(fitted_pipeline, artifact_path)

    assert artifact_path.is_file()
    assert artifact_path.stat().st_size > 0
    assert [entry.name for entry in tmp_path.iterdir()] == [ARTIFACT_NAME]


def test_save_creates_missing_parent_directories(fitted_pipeline, tmp_path):
    """A caller does not have to prepare the directory the artifact goes in."""
    artifact_path = tmp_path / "artifacts" / "run-1" / ARTIFACT_NAME

    save_model(fitted_pipeline, artifact_path)

    assert artifact_path.is_file()


def test_save_accepts_the_path_as_a_string(fitted_pipeline, tmp_path):
    """Either a ``str`` or a ``Path`` names the destination."""
    artifact_path = tmp_path / ARTIFACT_NAME

    save_model(fitted_pipeline, str(artifact_path))

    assert artifact_path.is_file()


def test_saving_twice_replaces_the_earlier_artifact(training_df, training_churn, tmp_path):
    """One path holds one model: the second save is what a later load reads.

    Saving a tree over a logistic model is the readable version of that,
    since the estimator the loaded pipeline ends in says which write won.
    """
    artifact_path = tmp_path / ARTIFACT_NAME

    save_model(train_logistic(training_df, training_churn), artifact_path)
    save_model(train_tree(training_df, training_churn), artifact_path)

    assert isinstance(classifier_of(load_model(artifact_path)), DecisionTreeClassifier)


# --- Loading returns the fitted pipeline that was written ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_load_returns_a_fitted_pipeline(trainer, training_df, training_churn, tmp_path):
    """What comes back is a ``Pipeline``, already fitted, not an unfitted shell."""
    artifact_path = tmp_path / ARTIFACT_NAME
    save_model(trainer(training_df, training_churn), artifact_path)

    loaded = load_model(artifact_path)

    assert isinstance(loaded, Pipeline)
    check_is_fitted(loaded)
    check_is_fitted(preprocessor_of(loaded))
    check_is_fitted(classifier_of(loaded))


@pytest.mark.parametrize(
    ("trainer", "expected_classifier"),
    [(train_logistic, LogisticRegression), (train_tree, DecisionTreeClassifier)],
    ids=["logistic", "tree"],
)
def test_load_returns_the_estimator_that_was_saved(
    trainer, expected_classifier, training_df, training_churn, tmp_path
):
    """Packaging carries whichever estimator it was handed, not a fixed one."""
    artifact_path = tmp_path / ARTIFACT_NAME
    save_model(trainer(training_df, training_churn), artifact_path)

    assert isinstance(classifier_of(load_model(artifact_path)), expected_classifier)


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_load_restores_the_whole_pipeline_not_only_the_classifier(
    trainer, training_df, training_churn, tmp_path
):
    """The preprocessing and what it learned come back with the estimator.

    The medians below were learned from the training fixture during the
    fit that preceded the save. Recovering them from the loaded object is
    what shows the preprocessor was serialised fitted rather than dropped
    or rebuilt.
    """
    artifact_path = tmp_path / ARTIFACT_NAME
    save_model(trainer(training_df, training_churn), artifact_path)

    loaded = load_model(artifact_path)

    assert list(loaded.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP]
    assert numerical_imputer_of(loaded).statistics_.tolist() == pytest.approx(
        [
            TRAINING_MEDIANS["tenure"],
            TRAINING_MEDIANS["MonthlyCharges"],
            TRAINING_MEDIANS["TotalCharges"],
        ]
    )
    assert set(classifier_of(loaded).classes_) == CHURN_LABELS


def test_load_accepts_the_path_as_a_string(fitted_pipeline, tmp_path):
    """Either a ``str`` or a ``Path`` names the artifact to read."""
    artifact_path = tmp_path / ARTIFACT_NAME
    save_model(fitted_pipeline, artifact_path)

    assert isinstance(load_model(str(artifact_path)), Pipeline)


# --- A missing artifact is reported as one ---


def test_loading_a_missing_artifact_raises_file_not_found(tmp_path):
    """A wrong path fails immediately, saying which file was expected."""
    missing_path = tmp_path / "never-written.joblib"

    with pytest.raises(FileNotFoundError, match="No model artifact to load at"):
        load_model(missing_path)


def test_loading_from_a_missing_directory_raises_file_not_found(tmp_path):
    """Loading creates nothing, so an absent directory is the same failure."""
    missing_path = tmp_path / "no-such-directory" / ARTIFACT_NAME

    with pytest.raises(FileNotFoundError, match="No model artifact to load at"):
        load_model(missing_path)

    assert not missing_path.parent.exists()


# --- A round trip preserves what the model predicts ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_round_trip_preserves_predictions(trainer, training_df, training_churn, held_out_df, tmp_path):
    """The loaded model predicts what the saved one predicted, from a raw frame.

    ``held_out_df`` is passed as it comes rather than preprocessed, so the
    predictions agreeing means the packaged preprocessing ran too.
    """
    artifact_path = tmp_path / ARTIFACT_NAME
    original = trainer(training_df, training_churn)
    expected_predictions = original.predict(held_out_df)

    save_model(original, artifact_path)
    loaded = load_model(artifact_path)

    assert_array_equal(loaded.predict(held_out_df), expected_predictions)
    assert set(loaded.predict(held_out_df)) <= CHURN_LABELS


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_round_trip_preserves_predicted_probabilities(
    trainer, training_df, training_churn, held_out_df, tmp_path
):
    """Probabilities survive too, so the fitted coefficients and splits came across.

    Labels alone could agree between two models that differ; the
    probabilities behind them are the stricter statement.
    """
    artifact_path = tmp_path / ARTIFACT_NAME
    original = trainer(training_df, training_churn)
    expected_probabilities = original.predict_proba(held_out_df)

    save_model(original, artifact_path)

    assert_allclose(
        load_model(artifact_path).predict_proba(held_out_df), expected_probabilities
    )


# --- The caller's model and frame are left as they were ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_saving_leaves_the_model_fitted_and_predicting_the_same(
    trainer, training_df, training_churn, held_out_df, tmp_path
):
    """Saving reads the model; it does not fit, refit, or empty it."""
    original = trainer(training_df, training_churn)
    predictions_before = original.predict(held_out_df)
    imputer_statistics_before = numerical_imputer_of(original).statistics_.tolist()

    save_model(original, tmp_path / ARTIFACT_NAME)

    check_is_fitted(original)
    assert_array_equal(original.predict(held_out_df), predictions_before)
    assert numerical_imputer_of(original).statistics_.tolist() == pytest.approx(
        imputer_statistics_before
    )


def test_loading_returns_an_object_separate_from_the_one_saved(
    fitted_pipeline, held_out_df, tmp_path
):
    """The loaded pipeline is a reconstruction, sharing nothing with the original.

    Predicting with the copy therefore cannot reach back into the model the
    caller still holds.
    """
    artifact_path = tmp_path / ARTIFACT_NAME
    save_model(fitted_pipeline, artifact_path)

    loaded = load_model(artifact_path)
    loaded.predict(held_out_df)

    assert loaded is not fitted_pipeline
    assert preprocessor_of(loaded) is not preprocessor_of(fitted_pipeline)
    assert classifier_of(loaded) is not classifier_of(fitted_pipeline)
    check_is_fitted(fitted_pipeline)


def test_round_trip_leaves_the_callers_frame_alone(fitted_pipeline, held_out_df, tmp_path):
    """Predicting through a round-tripped model writes nothing back to the frame."""
    artifact_path = tmp_path / ARTIFACT_NAME
    original_held_out_df = held_out_df.copy(deep=True)

    save_model(fitted_pipeline, artifact_path)
    load_model(artifact_path).predict(held_out_df)

    assert_frame_equal(held_out_df, original_held_out_df)
