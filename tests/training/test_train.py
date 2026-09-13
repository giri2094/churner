"""Unit tests for the baseline model training functions.

These tests establish the public contract of ``train_logistic`` and
``train_tree``: each returns a fitted sklearn ``Pipeline``, carrying the
estimator it is named for, learned from the training records it was handed,
and able to predict from a raw feature frame of the same shape.

The training medians below are written out here rather than imported from
the modules under test or read off the fixture, so a test fails if the fit
learned from different records rather than agreeing with itself. Every
expectation is hand-calculable from the six-customer training fixture
defined in this module.

Nothing here splits data, scores a model, or writes one to disk. The
functions under test are handed training records directly.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from churner.training.train import train_logistic, train_tree

# --- The structure the fitted pipelines are expected to have ---
# Restated here on purpose: comparing a fitted pipeline against these names
# is what makes its shape testable, which comparing it against the constants
# it was built from would not.
PREPROCESSOR_STEP = "preprocessor"
CLASSIFIER_STEP = "classifier"
NUMERICAL_BRANCH = "numerical"
IMPUTER_STEP = "imputer"

TRAINERS = [train_logistic, train_tree]

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
# Six customers holding the shape ``prepare_modeling_data`` hands on, with
# the target kept beside them as the separate series a classifier is fitted
# against. Every amount is distinct and none is missing, so each column's
# median is fixed by the fixture alone:
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

# A uniform shift applied to tenure so a second fit can be shown to have
# learned from a different training frame rather than from a cached one.
TENURE_SHIFT = 100.0
SHIFTED_TENURE_MEDIAN = TRAINING_MEDIANS["tenure"] + TENURE_SHIFT

# --- Controlled population the fitted pipelines are asked to predict for ---
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
    """Build the two customers the fitted pipelines are asked to predict for."""
    return pd.DataFrame(HELD_OUT_CUSTOMERS)[FEATURE_COLUMNS]


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


# --- What the trainers return ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_trainer_returns_a_pipeline(trainer, training_df, training_churn):
    """Each trainer hands back a ``Pipeline`` and nothing else."""
    assert isinstance(trainer(training_df, training_churn), Pipeline)


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_returned_pipeline_is_fitted(trainer, training_df, training_churn):
    """The object returned has already been fitted; the caller does not fit it again."""
    pipeline = trainer(training_df, training_churn)

    check_is_fitted(pipeline)
    check_is_fitted(preprocessor_of(pipeline))
    check_is_fitted(classifier_of(pipeline))


def test_logistic_pipeline_ends_in_a_logistic_regression(training_df, training_churn):
    """The logistic trainer's last step is the estimator it is named for."""
    classifier = classifier_of(train_logistic(training_df, training_churn))

    assert isinstance(classifier, LogisticRegression)
    assert not isinstance(classifier, DecisionTreeClassifier)


def test_tree_pipeline_ends_in_a_decision_tree(training_df, training_churn):
    """The tree trainer's last step is the estimator it is named for."""
    classifier = classifier_of(train_tree(training_df, training_churn))

    assert isinstance(classifier, DecisionTreeClassifier)
    assert not isinstance(classifier, LogisticRegression)


# --- The fit used the records it was handed ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_fit_learns_from_the_supplied_training_records(trainer, training_df, training_churn):
    """Imputation statistics and class labels come from this training frame.

    A trainer that fitted on some other population, or that returned an
    unfitted pipeline, would not recover these medians or these two labels.
    """
    pipeline = trainer(training_df, training_churn)
    imputer = numerical_imputer_of(pipeline)

    assert imputer.statistics_.tolist() == pytest.approx(
        [
            TRAINING_MEDIANS["tenure"],
            TRAINING_MEDIANS["MonthlyCharges"],
            TRAINING_MEDIANS["TotalCharges"],
        ]
    )
    assert set(classifier_of(pipeline).classes_) == CHURN_LABELS


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_a_different_training_frame_produces_different_fitted_state(
    trainer, training_df, training_churn
):
    """Shifting tenure shifts the median the imputer learns.

    That is the supplied-data contract restated as a contrast: two fits on
    two frames do not share a cached statistic.
    """
    original = trainer(training_df, training_churn)
    shifted = trainer(training_df.assign(tenure=training_df["tenure"] + TENURE_SHIFT), training_churn)

    assert numerical_imputer_of(original).statistics_[0] == pytest.approx(
        TRAINING_MEDIANS["tenure"]
    )
    assert numerical_imputer_of(shifted).statistics_[0] == pytest.approx(SHIFTED_TENURE_MEDIAN)


# --- Prediction from raw feature frames ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_fitted_pipeline_predicts_from_a_raw_feature_frame(
    trainer, training_df, training_churn, held_out_df
):
    """``predict`` takes the frame as it comes, not a preprocessed matrix."""
    pipeline = trainer(training_df, training_churn)

    predictions = pipeline.predict(held_out_df)

    assert len(predictions) == len(held_out_df)
    assert set(predictions) <= CHURN_LABELS


# --- Independence ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_separate_calls_create_independent_fitted_pipelines(
    trainer, training_df, training_churn
):
    """Two calls share no object, so fitting one cannot reach the other."""
    first = trainer(training_df, training_churn)
    second = trainer(training_df, training_churn)

    assert first is not second
    assert preprocessor_of(first) is not preprocessor_of(second)
    assert classifier_of(first) is not classifier_of(second)
    check_is_fitted(first)
    check_is_fitted(second)


def test_training_one_baseline_leaves_the_other_independent(training_df, training_churn, held_out_df):
    """Fitting the logistic pipeline does not fit, unfit, or alias the tree.

    Each trainer builds its own pipeline, so the two fitted objects remain
    distinct down to their estimators, and predicting with one leaves the
    other fitted.
    """
    logistic_pipeline = train_logistic(training_df, training_churn)
    tree_pipeline = train_tree(training_df, training_churn)

    assert logistic_pipeline is not tree_pipeline
    assert preprocessor_of(logistic_pipeline) is not preprocessor_of(tree_pipeline)
    assert classifier_of(logistic_pipeline) is not classifier_of(tree_pipeline)
    assert isinstance(classifier_of(logistic_pipeline), LogisticRegression)
    assert isinstance(classifier_of(tree_pipeline), DecisionTreeClassifier)

    logistic_pipeline.predict(held_out_df)
    check_is_fitted(tree_pipeline)
    tree_pipeline.predict(held_out_df)
    check_is_fitted(logistic_pipeline)


# --- Guarantees about the caller's frames ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_training_leaves_the_callers_data_alone(trainer, training_df, training_churn):
    """The trainer reads the frames it is given and writes nothing back."""
    original_training_df = training_df.copy(deep=True)
    original_training_churn = training_churn.copy(deep=True)

    trainer(training_df, training_churn)

    assert_frame_equal(training_df, original_training_df)
    assert_series_equal(training_churn, original_training_churn)
