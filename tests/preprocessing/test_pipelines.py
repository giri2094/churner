"""Unit tests for the baseline model pipelines.

These tests establish the public contract of ``create_logistic_pipeline`` and
``create_tree_pipeline``: the two steps each pipeline holds, which preprocessor
and which estimator it got, how those estimators are configured, the state the
pipeline is in before it is fitted, and what one ``fit`` followed by one
``predict`` does to raw feature frames.

The step names, the estimator settings, and the medians below are written out
here rather than imported from the modules under test or read off the fixture,
so a test fails if one of them changes rather than agreeing with itself. Every
expectation is hand-calculable from the six-customer training fixture defined
in this module, which holds the same columns and the same recorded amounts as
the preprocessor tests do, plus the churn labels a classifier needs.

Nothing here claims that a ``Pipeline`` prevents leakage. It fits every step on
whatever frame it is handed; what the tests below pin down is that one frame
fits both steps and that a later frame is preprocessed by what that fit
learned, which is what lets a caller fit on training records alone.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from churner.preprocessing.pipelines import (
    create_logistic_pipeline,
    create_tree_pipeline,
)
from churner.preprocessing.preprocessors import CATEGORICAL_BRANCH, NUMERICAL_BRANCH

# --- The structure the pipelines are expected to have ---
# Restated here on purpose: comparing a pipeline against these names is what
# makes its shape testable, which comparing it against the constants it was
# built from would not.
EXPECTED_STEP_NAMES = ["preprocessor", "classifier"]

PREPROCESSOR_STEP = "preprocessor"
CLASSIFIER_STEP = "classifier"

FACTORIES = [create_logistic_pipeline, create_tree_pipeline]

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
# Six customers holding the shape ``prepare_modeling_data`` hands on, with the
# target kept beside them as the separate series a classifier is fitted
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
    # The last customer has no internet service, so every add-on records that
    # rather than a "No" that would read as a declined subscription.
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

# Both labels are present, three each, because an estimator cannot be fitted on
# a single class. They are the recorded "Yes"/"No" strings
# ``prepare_modeling_data`` hands back, left unencoded, since encoding the
# target is no part of what these pipelines do.
TRAINING_CHURN = ["Yes", "Yes", "No", "Yes", "No", "No"]

CHURN_LABELS = {"No", "Yes"}

TRAINING_MEDIANS = {"tenure": 11.00, "MonthlyCharges": 62.50, "TotalCharges": 700.00}

# --- Controlled population the fitted pipelines are asked to predict for ---
# Two customers who were not in the training fixture, holding only values the
# training fixture also holds, so predicting for them meets no unknown category
# unless a test introduces one.
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


def step_names_of(pipeline: Pipeline) -> list[str]:
    """List the names a pipeline reports its steps under, in order."""
    return [name for name, _ in pipeline.steps]


def preprocessor_of(pipeline: Pipeline) -> ColumnTransformer:
    """Return the transformer a pipeline runs its features through."""
    return pipeline.named_steps[PREPROCESSOR_STEP]


def classifier_of(pipeline: Pipeline):
    """Return the estimator a pipeline ends in."""
    return pipeline.named_steps[CLASSIFIER_STEP]


def branch_of(preprocessor: ColumnTransformer, branch_name: str):
    """Return the ``(transformer, columns)`` a preprocessor gives to one branch."""
    branches = {
        name: (transformer, columns) for name, transformer, columns in preprocessor.transformers
    }
    assert branch_name in branches, f"No '{branch_name}' branch in {sorted(branches)}."
    return branches[branch_name]


def steps_of(transformer) -> list:
    """List the transformers a branch applies, whether or not it is a pipeline.

    A branch holding a single transformation needs no pipeline around it, so
    the tests describe what a branch does without depending on which of the two
    shapes it took.
    """
    if isinstance(transformer, Pipeline):
        return [step for _, step in transformer.steps]
    return [transformer]


def encoder_of(pipeline: Pipeline) -> OneHotEncoder:
    """Return the one-hot encoder from a pipeline's categorical branch."""
    categorical_transformer, _ = branch_of(preprocessor_of(pipeline), CATEGORICAL_BRANCH)
    encoders = [
        step for step in steps_of(categorical_transformer) if isinstance(step, OneHotEncoder)
    ]
    assert len(encoders) == 1, "Expected exactly one one-hot encoder in the categorical branch."
    return encoders[0]


def as_dense(matrix) -> np.ndarray:
    """Return a transformed matrix as a dense array, sparse or not.

    Whether a ``ColumnTransformer`` returns a sparse matrix depends on how
    dense the encoded columns turn out to be, which is a property of the
    fixture rather than of the design under test.
    """
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def preprocessed(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Return what a fitted pipeline's preprocessing step makes of a frame.

    Reaching into the fitted step is how a test reads the matrix the estimator
    was handed, which the pipeline otherwise keeps to itself.
    """
    return as_dense(preprocessor_of(pipeline).transform(df))


# --- What the factories return ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_factory_returns_a_pipeline(factory):
    """Each factory hands back a ``Pipeline`` and nothing else.

    The caller is given one object to fit and predict with; that it is made of
    a preprocessor and an estimator is this module's concern.
    """
    assert isinstance(factory(), Pipeline)


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_pipeline_holds_exactly_a_preprocessor_then_a_classifier(factory):
    """Two steps, under the agreed names, in the order the data flows through.

    The names are part of the contract: they are how a caller reaches the
    fitted preprocessing or the fitted model, so a rename would break callers
    that inspect either.
    """
    assert step_names_of(factory()) == EXPECTED_STEP_NAMES


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_preprocessing_step_is_a_column_transformer(factory):
    """The first step is the branching transformer the preprocessors module builds."""
    assert isinstance(preprocessor_of(factory()), ColumnTransformer)


# --- Which preprocessor each pipeline got ---


def test_logistic_pipeline_imputes_then_standardises_its_amounts():
    """The logistic pipeline carries the preprocessor built for Logistic Regression.

    Its numerical branch is the distinguishing part: the amounts are imputed
    and then standardised, so they enter the model's weighted sum on comparable
    terms.
    """
    numerical_transformer, numerical_columns = branch_of(
        preprocessor_of(create_logistic_pipeline()), NUMERICAL_BRANCH
    )
    steps = steps_of(numerical_transformer)

    assert list(numerical_columns) == NUMERICAL_FEATURE_COLUMNS
    assert [type(step) for step in steps] == [SimpleImputer, StandardScaler]
    assert steps[0].strategy == "median"


def test_tree_pipeline_imputes_its_amounts_without_standardising():
    """The tree pipeline carries the preprocessor built for tree models.

    Its numerical branch imputes and stops there, and no branch of it scales
    anything: a split threshold moves along with any rescaling, so
    standardising would change the numbers a split is expressed in without
    changing which splits are available.
    """
    preprocessor = preprocessor_of(create_tree_pipeline())
    numerical_transformer, numerical_columns = branch_of(preprocessor, NUMERICAL_BRANCH)
    steps = steps_of(numerical_transformer)

    assert list(numerical_columns) == NUMERICAL_FEATURE_COLUMNS
    assert [type(step) for step in steps] == [SimpleImputer]
    assert steps[0].strategy == "median"
    for _, transformer, _ in preprocessor.transformers:
        assert not any(isinstance(step, StandardScaler) for step in steps_of(transformer))


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_pipeline_imputes_then_encodes_its_categories(factory):
    """Both pipelines carry the same categorical treatment, in the same order.

    Encoding first would turn each missing value into a category of its own,
    leaving nothing for the imputer to fill. Dropping the first category and
    ignoring unknown values are both settings the preprocessors module argues
    for; they are checked here so a pipeline cannot quietly be given a
    differently configured encoder.
    """
    pipeline = factory()
    categorical_transformer, categorical_columns = branch_of(
        preprocessor_of(pipeline), CATEGORICAL_BRANCH
    )
    steps = steps_of(categorical_transformer)
    encoder = encoder_of(pipeline)

    assert list(categorical_columns) == CATEGORICAL_FEATURE_COLUMNS
    assert [type(step) for step in steps] == [SimpleImputer, OneHotEncoder]
    assert steps[0].strategy == "most_frequent"
    assert encoder.drop == "first"
    assert encoder.handle_unknown == "ignore"


def test_logistic_pipeline_hands_its_estimator_standardised_amounts(training_df, training_churn):
    """What the logistic preprocessing does, read off the matrix it produces.

    Fitted and transformed on the same records, each amount comes out with a
    mean of zero and a standard deviation of one. This is the configuration
    test above restated as behaviour, so the pipeline is shown to standardise
    rather than merely to hold a scaler.
    """
    pipeline = create_logistic_pipeline().fit(training_df, training_churn)

    amounts = preprocessed(pipeline, training_df)[:, : len(NUMERICAL_FEATURE_COLUMNS)]

    assert amounts.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert amounts.std(axis=0) == pytest.approx([1.0, 1.0, 1.0])


def test_tree_pipeline_hands_its_estimator_the_recorded_amounts(training_df, training_churn):
    """What the tree preprocessing does: the amounts arrive as they were recorded."""
    pipeline = create_tree_pipeline().fit(training_df, training_churn)

    amounts = preprocessed(pipeline, training_df)[:, : len(NUMERICAL_FEATURE_COLUMNS)]

    assert amounts == pytest.approx(training_df[NUMERICAL_FEATURE_COLUMNS].to_numpy(dtype=float))


# --- Which estimator each pipeline got, and how it is configured ---


def test_logistic_pipeline_ends_in_a_logistic_regression():
    """The logistic pipeline's last step is the estimator it is named for."""
    classifier = classifier_of(create_logistic_pipeline())

    assert isinstance(classifier, LogisticRegression)
    assert not isinstance(classifier, DecisionTreeClassifier)


def test_tree_pipeline_ends_in_a_decision_tree():
    """The tree pipeline's last step is the estimator it is named for."""
    classifier = classifier_of(create_tree_pipeline())

    assert isinstance(classifier, DecisionTreeClassifier)
    assert not isinstance(classifier, LogisticRegression)


def test_logistic_regression_is_given_room_to_converge():
    """The iteration limit is raised to 1000, well past the default of 100.

    The optimiser then stops because it converged rather than because it ran
    out of iterations. It is a limit on what the fit is allowed to finish, not
    a tuned parameter.
    """
    assert classifier_of(create_logistic_pipeline()).max_iter == 1000


def test_decision_tree_is_seeded_for_reproducibility():
    """The seed is fixed at 42, so the same records grow the same tree.

    A tree resolves equally good candidate splits by a random draw, which is
    what makes an unseeded fit vary between runs.
    """
    assert classifier_of(create_tree_pipeline()).random_state == 42


def test_decision_tree_growth_is_left_unrestricted():
    """No limit is placed on depth or on the size of a split or a leaf.

    The unrestricted tree is the baseline that any later limit on its growth
    has to be argued against, so these defaults are pinned rather than assumed:
    setting one of them here would be tuning done before there is a measurement
    to justify it.
    """
    classifier = classifier_of(create_tree_pipeline())

    assert classifier.max_depth is None
    assert classifier.min_samples_split == 2
    assert classifier.min_samples_leaf == 1


def test_neither_baseline_reweights_the_two_outcomes():
    """Neither estimator compensates for the imbalance between churn and no churn.

    Churn is the minority outcome, and how to account for that is a modelling
    decision to be taken against a measured baseline. Pinning it here keeps
    that decision from being made silently inside a factory.
    """
    assert classifier_of(create_logistic_pipeline()).class_weight is None
    assert classifier_of(create_tree_pipeline()).class_weight is None


# --- The state a new pipeline is in ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_returned_pipeline_is_not_yet_fitted(factory):
    """Construction learns nothing, so there is no fitted state to inherit."""
    with pytest.raises(NotFittedError):
        check_is_fitted(factory())


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_returned_pipeline_holds_no_fitted_preprocessing_state(factory):
    """Neither the transformer nor the statistics its steps would learn exist yet.

    A pipeline that arrived with fitted preprocessing would carry medians,
    scales, and categories from whatever data built it, which is the one thing
    fitting after the split is meant to rule out.
    """
    preprocessor = preprocessor_of(factory())

    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)
    assert not hasattr(preprocessor, "transformers_")
    for _, transformer, _ in preprocessor.transformers:
        for step in steps_of(transformer):
            assert not hasattr(step, "statistics_")
            assert not hasattr(step, "categories_")
            assert not hasattr(step, "mean_")


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_returned_estimator_is_not_yet_fitted(factory):
    """The estimator holds no learned parameters either."""
    with pytest.raises(NotFittedError):
        check_is_fitted(classifier_of(factory()))


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_predicting_before_fitting_raises(factory, held_out_df):
    """An unfitted pipeline refuses to predict rather than inventing a fit."""
    with pytest.raises(NotFittedError):
        factory().predict(held_out_df)


# --- Independence between pipelines ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_factory_creates_independent_instances(factory):
    """Two calls share no object, down to the steps inside the preprocessing."""
    first = factory()
    second = factory()

    assert first is not second
    assert first.steps is not second.steps
    assert preprocessor_of(first) is not preprocessor_of(second)
    assert classifier_of(first) is not classifier_of(second)
    assert (
        branch_of(preprocessor_of(first), NUMERICAL_BRANCH)[0]
        is not branch_of(preprocessor_of(second), NUMERICAL_BRANCH)[0]
    )
    assert encoder_of(first) is not encoder_of(second)


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_fitting_one_pipeline_leaves_another_unfitted(factory, training_df, training_churn):
    """Sharing no mutable state, restated as behaviour: one fit reaches one pipeline."""
    fitted = factory()
    untouched = factory()

    fitted.fit(training_df, training_churn)

    check_is_fitted(fitted)
    with pytest.raises(NotFittedError):
        check_is_fitted(untouched)
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor_of(untouched))


def test_the_two_factories_share_no_object():
    """The logistic and tree pipelines are separate down to their steps.

    They apply the same categorical treatment, but through their own
    transformers, so fitting one leaves the other unfitted.
    """
    logistic_pipeline = create_logistic_pipeline()
    tree_pipeline = create_tree_pipeline()

    assert logistic_pipeline is not tree_pipeline
    assert preprocessor_of(logistic_pipeline) is not preprocessor_of(tree_pipeline)
    assert classifier_of(logistic_pipeline) is not classifier_of(tree_pipeline)
    assert encoder_of(logistic_pipeline) is not encoder_of(tree_pipeline)


# --- Fitting, and predicting for data the fit never saw ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_one_fit_fits_both_steps(factory, training_df, training_churn):
    """A single ``fit`` on a raw frame leaves the preprocessing and the model fitted.

    That is the point of combining them: the statistics and the model
    parameters come out of the same call on the same records, with no
    transformed matrix passing through the caller's hands.
    """
    pipeline = factory()

    pipeline.fit(training_df, training_churn)

    check_is_fitted(preprocessor_of(pipeline))
    check_is_fitted(classifier_of(pipeline))
    assert set(classifier_of(pipeline).classes_) == CHURN_LABELS


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_fitted_pipeline_predicts_from_a_raw_feature_frame(
    factory, training_df, training_churn, held_out_df
):
    """``predict`` takes the frame as it comes, not a preprocessed matrix.

    The pipeline applies the transformations it learned during ``fit`` to the
    frame it is given, so the caller hands over the same kind of frame it
    trained on and gets one label per row back.
    """
    pipeline = factory().fit(training_df, training_churn)

    predictions = pipeline.predict(held_out_df)

    assert len(predictions) == len(held_out_df)
    assert set(predictions) <= CHURN_LABELS


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_prediction_preprocesses_missing_and_unknown_values_on_the_way_in(
    factory, training_df, training_churn, held_out_df
):
    """The imputation and the encoding are applied inside ``predict``.

    A frame holding a missing amount and a category the fit never saw would
    stop an estimator handed it directly. Here it does not, because the fitted
    preprocessing fills the amount with the median it learned and encodes the
    unknown category as all zeros before the estimator sees anything.
    """
    pipeline = factory().fit(training_df, training_churn)
    awkward_df = held_out_df.assign(
        TotalCharges=[np.nan, 5715.00], InternetService=["Satellite", "Fiber optic"]
    )

    predictions = pipeline.predict(awkward_df)

    assert len(predictions) == len(awkward_df)
    assert set(predictions) <= CHURN_LABELS


def test_pipeline_preprocessing_learns_its_medians_from_the_frame_it_was_fitted_on(
    training_df, training_churn, held_out_df
):
    """A missing amount is filled with the median of the training records.

    The tree pipeline is used because it leaves the amounts unscaled, so the
    value that was filled in is readable straight out of the preprocessed
    matrix. It comes from the six records the pipeline was fitted on, not from
    the frame being predicted for.
    """
    pipeline = create_tree_pipeline().fit(training_df, training_churn)
    df_with_missing_amounts = held_out_df.assign(
        tenure=[np.nan, 60], TotalCharges=[136.65, np.nan]
    )

    amounts = preprocessed(pipeline, df_with_missing_amounts)

    # The numerical branch comes first, in schema order: tenure, MonthlyCharges,
    # TotalCharges.
    assert amounts[0, 0] == pytest.approx(TRAINING_MEDIANS["tenure"])
    assert amounts[1, 2] == pytest.approx(TRAINING_MEDIANS["TotalCharges"])


# --- Guarantees about the caller's frame ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_fitting_and_predicting_leave_the_callers_data_alone(
    factory, training_df, training_churn, held_out_df
):
    """Both calls read the frames they are given and write nothing back.

    The missing amount in the predicted frame is filled inside the pipeline
    only; the caller's copy still records no amount.
    """
    df_with_missing_amounts = held_out_df.assign(tenure=[np.nan, 60])
    original_training_df = training_df.copy(deep=True)
    original_training_churn = training_churn.copy(deep=True)
    original_held_out_df = df_with_missing_amounts.copy(deep=True)

    pipeline = factory().fit(training_df, training_churn)
    pipeline.predict(df_with_missing_amounts)

    assert_frame_equal(training_df, original_training_df)
    assert_series_equal(training_churn, original_training_churn)
    assert_frame_equal(df_with_missing_amounts, original_held_out_df)
