"""Unit tests for the model-agnostic evaluation function.

These tests establish the public contract of ``evaluate_model``: it scores
an already-fitted classifier on held-out records, returns the five
baseline metrics as an ``EvaluationResult``, and does so without refitting
the model or writing back to the test frames.

The known-metric case below is written out here rather than computed by
the module under test, so a test fails if the evaluator uses the wrong
inputs — class labels where probabilities belong, for example — rather
than agreeing with itself. Every expectation is hand-calculable from the
four-row scores defined in this module.

Nothing here splits data or trains a baseline, except where a fitted
logistic or tree pipeline is required to show that the same evaluator
accepts both.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from churner.evaluation.evaluate import EvaluationResult, evaluate_model
from churner.training.train import train_logistic, train_tree

# --- Metric names the result is expected to hold ---
# Restated here on purpose: comparing a result against these names is what
# makes the contract testable, which comparing it against the dataclass
# fields it was built from would not.
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")

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

# --- Controlled four-row scoring case ---
# Two retained and two churned customers, with predetermined class
# predictions and positive-class probabilities. The 0.5 decision threshold
# on those scores yields one false positive and one false negative:
#
#   true   pred   P(Yes)
#   No     No     0.2
#   No     Yes    0.8
#   Yes    No     0.3
#   Yes    Yes    0.9
#
#   accuracy  = (1 TN + 1 TP) / 4           = 0.50
#   precision = 1 TP / (1 TP + 1 FP)        = 0.50
#   recall    = 1 TP / (1 TP + 1 FN)        = 0.50
#   f1        = 2 * 0.50 * 0.50 / 1.00      = 0.50
#
# ROC-AUC from the four P(Yes) scores, over the four positive-negative
# pairs: 0.3 beats 0.2, 0.3 loses to 0.8, 0.9 beats 0.2, 0.9 beats 0.8,
# so three of four pairs are correctly ranked and AUC = 0.75.
#
# The same labels scored as 0/1 class predictions [0, 1, 0, 1] would
# rank only two of those four pairs correctly (counting ties as half),
# so that AUC would be 0.50. A result of 0.75 is therefore evidence that
# ROC-AUC used the continuous scores.
CONTROLLED_TRUE_LABELS = ["No", "No", "Yes", "Yes"]
CONTROLLED_PREDICTIONS = ["No", "Yes", "No", "Yes"]
CONTROLLED_POSITIVE_SCORES = [0.2, 0.8, 0.3, 0.9]

EXPECTED_ACCURACY = 0.50
EXPECTED_PRECISION = 0.50
EXPECTED_RECALL = 0.50
EXPECTED_F1 = 0.50
EXPECTED_ROC_AUC = 0.75
BINARY_PREDICTION_ROC_AUC = 0.50

# --- Controlled training population for the real pipelines ---
# Six customers holding the shape ``prepare_modeling_data`` hands on.
# Every amount is distinct and none is missing, so each column's median
# is fixed by the fixture alone:
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

TRAINING_MEDIANS = {"tenure": 11.00, "MonthlyCharges": 62.50, "TotalCharges": 700.00}

# Two customers who were not in the training fixture. Their tenure and
# charge amounts sit well away from the training medians, so a preprocessor
# accidentally refitted on this frame would not recover those medians.
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

HELD_OUT_CHURN = ["Yes", "No"]


class FixedScoreClassifier:
    """A stand-in estimator that returns predetermined labels and scores.

    ``fit`` is present so a call to it can be detected. Evaluation is not
    supposed to call it.
    """

    classes_ = np.array(["No", "Yes"])

    def __init__(self, predictions, positive_scores):
        self._predictions = np.asarray(predictions)
        self._positive_scores = np.asarray(positive_scores, dtype=float)
        self.fit_call_count = 0

    def fit(self, X, y):
        self.fit_call_count += 1
        return self

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        yes_scores = self._positive_scores
        no_scores = 1.0 - yes_scores
        return np.column_stack([no_scores, yes_scores])


@pytest.fixture
def controlled_model() -> FixedScoreClassifier:
    """Build the four-row stand-in classifier described above."""
    return FixedScoreClassifier(CONTROLLED_PREDICTIONS, CONTROLLED_POSITIVE_SCORES)


@pytest.fixture
def controlled_X() -> pd.DataFrame:
    """Build a four-row predictor frame the stand-in classifier ignores."""
    return pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def controlled_y() -> pd.Series:
    """Build the four known labels of the scoring case."""
    return pd.Series(CONTROLLED_TRUE_LABELS, name="Churn")


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
    """Build the two customers the fitted pipelines are scored on."""
    return pd.DataFrame(HELD_OUT_CUSTOMERS)[FEATURE_COLUMNS]


@pytest.fixture
def held_out_churn() -> pd.Series:
    """Build the churn labels of the held-out customers."""
    return pd.Series(HELD_OUT_CHURN, name="Churn")


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


# --- What evaluate_model returns ---


def test_evaluate_model_returns_an_evaluation_result(
    controlled_model, controlled_X, controlled_y
):
    """The evaluator hands back an ``EvaluationResult`` and nothing else."""
    result = evaluate_model(controlled_model, controlled_X, controlled_y)

    assert isinstance(result, EvaluationResult)


def test_evaluation_result_holds_all_five_metrics_as_numbers(
    controlled_model, controlled_X, controlled_y
):
    """Each of the five baseline metrics is present and a real number."""
    result = evaluate_model(controlled_model, controlled_X, controlled_y)

    for metric_name in METRIC_NAMES:
        assert hasattr(result, metric_name)
        value = getattr(result, metric_name)
        assert type(value) is float


# --- Known values ---


def test_metrics_match_the_controlled_scoring_case(
    controlled_model, controlled_X, controlled_y
):
    """The five metrics recover the hand-calculated values of the four-row case."""
    result = evaluate_model(controlled_model, controlled_X, controlled_y)

    assert result.accuracy == pytest.approx(EXPECTED_ACCURACY)
    assert result.precision == pytest.approx(EXPECTED_PRECISION)
    assert result.recall == pytest.approx(EXPECTED_RECALL)
    assert result.f1 == pytest.approx(EXPECTED_F1)
    assert result.roc_auc == pytest.approx(EXPECTED_ROC_AUC)


def test_roc_auc_uses_continuous_scores_rather_than_class_predictions(
    controlled_model, controlled_X, controlled_y
):
    """ROC-AUC is 0.75 from the probabilities, not 0.50 from the 0/1 labels.

    The class predictions of this fixture rank the four pairs no better than
    chance once ties are counted as half. The probabilities rank three of
    the four pairs correctly. A result of 0.75 cannot have come from the
    labels alone.
    """
    result = evaluate_model(controlled_model, controlled_X, controlled_y)

    assert result.roc_auc == pytest.approx(EXPECTED_ROC_AUC)
    assert result.roc_auc != pytest.approx(BINARY_PREDICTION_ROC_AUC)


# --- Both baseline pipelines ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_evaluate_model_scores_a_fitted_baseline_pipeline(
    trainer, training_df, training_churn, held_out_df, held_out_churn
):
    """The same evaluator accepts a fitted logistic pipeline and a fitted tree.

    The held-out frame is still a raw feature frame: scoring it at all means
    the fitted pipeline applied the preprocessing it learned during training.
    """
    pipeline = trainer(training_df, training_churn)

    result = evaluate_model(pipeline, held_out_df, held_out_churn)

    assert isinstance(result, EvaluationResult)
    for metric_name in METRIC_NAMES:
        value = getattr(result, metric_name)
        assert type(value) is float


def test_logistic_and_tree_pipelines_are_both_scored_as_themselves(
    training_df, training_churn, held_out_df, held_out_churn
):
    """Evaluating one baseline does not require a model-specific evaluator."""
    logistic_pipeline = train_logistic(training_df, training_churn)
    tree_pipeline = train_tree(training_df, training_churn)

    assert isinstance(classifier_of(logistic_pipeline), LogisticRegression)
    assert isinstance(classifier_of(tree_pipeline), DecisionTreeClassifier)

    logistic_result = evaluate_model(logistic_pipeline, held_out_df, held_out_churn)
    tree_result = evaluate_model(tree_pipeline, held_out_df, held_out_churn)

    assert isinstance(logistic_result, EvaluationResult)
    assert isinstance(tree_result, EvaluationResult)
    assert isinstance(classifier_of(logistic_pipeline), LogisticRegression)
    assert isinstance(classifier_of(tree_pipeline), DecisionTreeClassifier)


# --- The fitted pipeline is used as it was handed over ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_evaluation_uses_the_fitted_pipeline_directly(
    trainer, training_df, training_churn, held_out_df, held_out_churn
):
    """Scoring calls ``predict`` and ``predict_proba`` on the object it was given.

    That is the fitted-pipeline contract: evaluation does not rebuild the
    model, extract the classifier, or wrap a different estimator around the
    same steps.
    """
    pipeline = trainer(training_df, training_churn)
    original_predict = pipeline.predict
    original_predict_proba = pipeline.predict_proba
    called = {"predict": False, "predict_proba": False}

    def predict(X):
        called["predict"] = True
        return original_predict(X)

    def predict_proba(X):
        called["predict_proba"] = True
        return original_predict_proba(X)

    pipeline.predict = predict
    pipeline.predict_proba = predict_proba

    evaluate_model(pipeline, held_out_df, held_out_churn)

    assert called["predict"]
    assert called["predict_proba"]


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_evaluation_does_not_refit_the_model(
    trainer, training_df, training_churn, held_out_df, held_out_churn
):
    """The preprocessing statistics and the classifier parameters stay put.

    The held-out amounts are far from the training medians, so a preprocessor
    accidentally fitted on the test frame would not recover them. ``fit``
    itself is also replaced, so a refit cannot hide behind unchanged numbers.
    """
    pipeline = trainer(training_df, training_churn)
    imputer = numerical_imputer_of(pipeline)
    classifier = classifier_of(pipeline)
    original_statistics = imputer.statistics_.copy()
    if isinstance(classifier, LogisticRegression):
        original_parameters = classifier.coef_.copy()
    else:
        original_parameters = classifier.tree_.value.copy()

    def refuse_fit(*args, **kwargs):
        raise AssertionError("evaluate_model must not refit the fitted model")

    pipeline.fit = refuse_fit

    evaluate_model(pipeline, held_out_df, held_out_churn)

    assert imputer.statistics_.tolist() == pytest.approx(
        [
            TRAINING_MEDIANS["tenure"],
            TRAINING_MEDIANS["MonthlyCharges"],
            TRAINING_MEDIANS["TotalCharges"],
        ]
    )
    assert imputer.statistics_.tolist() == original_statistics.tolist()
    if isinstance(classifier, LogisticRegression):
        assert np.array_equal(classifier.coef_, original_parameters)
    else:
        assert np.array_equal(classifier.tree_.value, original_parameters)


def test_evaluation_does_not_refit_the_controlled_classifier(
    controlled_model, controlled_X, controlled_y
):
    """The stand-in classifier's ``fit`` is never called."""
    evaluate_model(controlled_model, controlled_X, controlled_y)

    assert controlled_model.fit_call_count == 0


# --- Guarantees about the caller's frames ---


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_evaluation_leaves_the_callers_test_features_alone(
    trainer, training_df, training_churn, held_out_df, held_out_churn
):
    """A missing amount is filled inside the pipeline only.

    The caller's test frame still records no amount afterwards.
    """
    X_test = held_out_df.assign(tenure=[np.nan, 60])
    original_X_test = X_test.copy(deep=True)

    pipeline = trainer(training_df, training_churn)
    evaluate_model(pipeline, X_test, held_out_churn)

    assert_frame_equal(X_test, original_X_test)


@pytest.mark.parametrize("trainer", TRAINERS, ids=["logistic", "tree"])
def test_evaluation_leaves_the_callers_test_labels_alone(
    trainer, training_df, training_churn, held_out_df, held_out_churn
):
    """The evaluator reads ``y_test`` and writes nothing back."""
    original_y_test = held_out_churn.copy(deep=True)

    pipeline = trainer(training_df, training_churn)
    evaluate_model(pipeline, held_out_df, held_out_churn)

    assert_series_equal(held_out_churn, original_y_test)
