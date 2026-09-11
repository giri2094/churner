"""Unit tests for the model-specific preprocessors.

These tests establish the public contract of ``create_logistic_preprocessor``
and ``create_tree_preprocessor``: the columns each branch is given, the steps
each branch applies, the state the returned transformer is in before it is
fitted, and what fitting on one frame and transforming another produces.

The column lists and the medians below are written out here rather than
imported from ``churner.schema.features`` or read off the fixture, so a test
fails if the schema or a learned statistic changes rather than agreeing with
itself. Every expectation is hand-calculable from the six-customer training
fixture defined in this module.

Nothing here claims that a ``ColumnTransformer`` prevents leakage. It fits on
whatever frame it is handed; the tests below only pin down the fit/transform
contract that lets a caller fit on training records alone.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from churner.preprocessing.preprocessors import (
    CATEGORICAL_BRANCH,
    NUMERICAL_BRANCH,
    create_logistic_preprocessor,
    create_tree_preprocessor,
)

# --- The schema the preprocessors are expected to work from ---
# Restated here on purpose: comparing the branches against these lists is what
# makes the assignment testable, which comparing them against the schema module
# they are built from would not.
EXPECTED_NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

EXPECTED_CATEGORICAL_FEATURES = [
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

FEATURE_COLUMNS = EXPECTED_NUMERICAL_FEATURES + EXPECTED_CATEGORICAL_FEATURES

FACTORIES = [create_logistic_preprocessor, create_tree_preprocessor]

# --- Controlled training population ---
# Six customers holding the shape ``prepare_modeling_data`` hands on: amounts
# as numbers, categories as recorded, no identifier and no target. Every
# amount is distinct and none is missing, so each column's median is fixed by
# the fixture alone:
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

TRAINING_MEDIANS = {"tenure": 11.00, "MonthlyCharges": 62.50, "TotalCharges": 700.00}

# --- Controlled population the fitted preprocessors are asked to transform ---
# Two customers who were not in the training fixture, holding only values the
# training fixture also holds, so a transformation of them meets no unknown
# category unless a test introduces one.
UNSEEN_CUSTOMERS = {
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
def unseen_df() -> pd.DataFrame:
    """Build the two customers the fitted preprocessors are asked to transform."""
    return pd.DataFrame(UNSEEN_CUSTOMERS)[FEATURE_COLUMNS]


def branch_of(preprocessor: ColumnTransformer, branch_name: str):
    """Return the ``(transformer, columns)`` a preprocessor gives to one branch."""
    branches = {
        name: (transformer, columns) for name, transformer, columns in preprocessor.transformers
    }
    assert branch_name in branches, f"No '{branch_name}' branch in {sorted(branches)}."
    return branches[branch_name]


def fitted_branch_of(preprocessor: ColumnTransformer, branch_name: str):
    """Return the fitted transformer a preprocessor holds for one branch.

    A ``ColumnTransformer`` fits clones of the transformers it was built with,
    so the fitted state lives on the copies it keeps rather than on the objects
    ``transformers`` reports.
    """
    return preprocessor.named_transformers_[branch_name]


def steps_of(transformer) -> list:
    """List the transformers a branch applies, whether or not it is a pipeline.

    A branch holding a single transformation needs no pipeline around it, so
    the tests describe what a branch does without depending on which of the two
    shapes it took.
    """
    if isinstance(transformer, Pipeline):
        return [step for _, step in transformer.steps]
    return [transformer]


def encoder_of(preprocessor: ColumnTransformer) -> OneHotEncoder:
    """Return the one-hot encoder from a preprocessor's categorical branch."""
    categorical_transformer, _ = branch_of(preprocessor, CATEGORICAL_BRANCH)
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


# --- What the factories return ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_factory_returns_a_column_transformer(factory):
    """Each factory hands back a ``ColumnTransformer`` and nothing else.

    The caller is given an object it can fit and transform with; how the
    branches inside it were assembled is this module's concern.
    """
    assert isinstance(factory(), ColumnTransformer)


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_returned_preprocessor_is_not_yet_fitted(factory):
    """Construction learns nothing, so there is no fitted state to inherit.

    A preprocessor that arrived already fitted would carry statistics from
    whatever data built it, which is the one thing fitting after the split is
    meant to rule out.
    """
    with pytest.raises(NotFittedError):
        check_is_fitted(factory())


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_factory_creates_independent_instances(factory):
    """Two calls share no object, so fitting one cannot fit the other."""
    first = factory()
    second = factory()

    assert first is not second
    assert branch_of(first, NUMERICAL_BRANCH)[0] is not branch_of(second, NUMERICAL_BRANCH)[0]
    assert encoder_of(first) is not encoder_of(second)


def test_the_two_factories_share_no_transformer():
    """The logistic and tree preprocessors are separate down to their steps.

    They apply the same categorical treatment, but through their own
    transformers, so fitting one leaves the other unfitted.
    """
    logistic_preprocessor = create_logistic_preprocessor()
    tree_preprocessor = create_tree_preprocessor()

    assert logistic_preprocessor is not tree_preprocessor
    assert encoder_of(logistic_preprocessor) is not encoder_of(tree_preprocessor)


# --- Which columns each branch is given ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_numerical_branch_is_given_the_numerical_features(factory):
    """The recorded amounts, in schema order, go to the numerical branch."""
    _, numerical_columns = branch_of(factory(), NUMERICAL_BRANCH)

    assert list(numerical_columns) == EXPECTED_NUMERICAL_FEATURES


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_categorical_branch_is_given_the_categorical_features(factory):
    """The recorded categories, in schema order, go to the categorical branch."""
    _, categorical_columns = branch_of(factory(), CATEGORICAL_BRANCH)

    assert list(categorical_columns) == EXPECTED_CATEGORICAL_FEATURES


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_no_feature_is_handled_by_both_branches(factory):
    """The branches partition the features: none is left out and none is doubled."""
    _, numerical_columns = branch_of(factory(), NUMERICAL_BRANCH)
    _, categorical_columns = branch_of(factory(), CATEGORICAL_BRANCH)

    assert set(numerical_columns).isdisjoint(categorical_columns)
    assert sorted(list(numerical_columns) + list(categorical_columns)) == sorted(FEATURE_COLUMNS)


# --- What each branch does ---


def test_logistic_numerical_branch_imputes_the_median_then_standardises():
    """Logistic Regression gets its amounts imputed and put on one scale.

    The order matters: standardising first would have to compute a mean and a
    standard deviation around values that are still missing.
    """
    numerical_transformer, _ = branch_of(create_logistic_preprocessor(), NUMERICAL_BRANCH)
    steps = steps_of(numerical_transformer)

    assert [type(step) for step in steps] == [SimpleImputer, StandardScaler]
    assert steps[0].strategy == "median"


def test_tree_numerical_branch_imputes_the_median_without_standardising():
    """A tree gets its amounts imputed and left on the scale they were recorded on."""
    numerical_transformer, _ = branch_of(create_tree_preprocessor(), NUMERICAL_BRANCH)
    steps = steps_of(numerical_transformer)

    assert [type(step) for step in steps] == [SimpleImputer]
    assert steps[0].strategy == "median"


def test_no_scaler_anywhere_in_the_tree_preprocessor():
    """Not in the numerical branch and not in the categorical one either."""
    tree_preprocessor = create_tree_preprocessor()

    for _, transformer, _ in tree_preprocessor.transformers:
        assert not any(isinstance(step, StandardScaler) for step in steps_of(transformer))


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_categorical_branch_imputes_the_most_frequent_value_then_encodes(factory):
    """Both models get the same categorical treatment, in the same order.

    Encoding first would turn each missing value into a category of its own,
    leaving nothing for the imputer to fill.
    """
    categorical_transformer, _ = branch_of(factory(), CATEGORICAL_BRANCH)
    steps = steps_of(categorical_transformer)

    assert [type(step) for step in steps] == [SimpleImputer, OneHotEncoder]
    assert steps[0].strategy == "most_frequent"


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_one_hot_encoder_drops_a_reference_category_and_ignores_unknown_ones(factory):
    """Dropping the first category and ignoring unknown values are both settings.

    The first leaves one category as the reference the others are measured
    against, which is what keeps a column's indicators from being perfectly
    collinear with a model's intercept. The second gives ``transform`` a defined
    behaviour when it meets a value that was absent at fit time.
    """
    encoder = encoder_of(factory())

    assert encoder.drop == "first"
    assert encoder.handle_unknown == "ignore"


# --- Fitting, and transforming data the fit never saw ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_fitting_on_training_data_transforms_other_data_without_refitting(
    factory, training_df, unseen_df
):
    """A preprocessor fitted once transforms further records as they arrive.

    This is the fit/transform contract a caller relies on to keep test records
    out of what the preprocessor learned: the fit happens on the training frame
    and the statistics it produced are not revisited when another frame is
    transformed. The transformer does not enforce that on its own; fitting it on
    training records alone is what does.
    """
    preprocessor = factory()

    preprocessor.fit(training_df)
    imputer = steps_of(fitted_branch_of(preprocessor, NUMERICAL_BRANCH))[0]
    medians_after_fit = imputer.statistics_.copy()

    transformed = as_dense(preprocessor.transform(unseen_df))

    assert transformed.shape[0] == len(unseen_df)
    assert imputer.statistics_.tolist() == medians_after_fit.tolist()


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_fitted_medians_come_from_the_training_records(factory, training_df):
    """The imputer learns each column's median from the frame it was fitted on."""
    preprocessor = factory().fit(training_df)

    imputer = steps_of(fitted_branch_of(preprocessor, NUMERICAL_BRANCH))[0]

    assert imputer.statistics_.tolist() == pytest.approx(
        [TRAINING_MEDIANS[column] for column in EXPECTED_NUMERICAL_FEATURES]
    )


def test_missing_amounts_are_filled_with_the_training_median(training_df, unseen_df):
    """A missing amount becomes the median of the records the fit was given.

    The tree preprocessor is used because it leaves the amounts unscaled, so
    the value that was filled in is readable straight out of the output.
    """
    preprocessor = create_tree_preprocessor().fit(training_df)
    df_with_missing_amounts = unseen_df.assign(tenure=[np.nan, 60], TotalCharges=[136.65, np.nan])

    transformed = as_dense(preprocessor.transform(df_with_missing_amounts))

    # The numerical branch comes first, in schema order: tenure, MonthlyCharges,
    # TotalCharges.
    assert transformed[0, 0] == pytest.approx(TRAINING_MEDIANS["tenure"])
    assert transformed[1, 2] == pytest.approx(TRAINING_MEDIANS["TotalCharges"])
    assert not np.isnan(transformed).any()


def test_missing_categories_are_filled_rather_than_encoded_as_their_own(training_df, unseen_df):
    """A missing category is filled in, so it adds no indicator of its own."""
    preprocessor = create_tree_preprocessor().fit(training_df)
    column_count_before = len(preprocessor.get_feature_names_out())
    df_with_missing_categories = unseen_df.assign(Contract=[None, "One year"])

    transformed = as_dense(preprocessor.transform(df_with_missing_categories))

    assert transformed.shape[1] == column_count_before
    assert not np.isnan(transformed).any()


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_a_category_absent_from_the_fit_is_transformed_without_raising(
    factory, training_df, unseen_df
):
    """An unseen category is encoded as all zeros instead of stopping the transform.

    That is the same encoding the dropped reference category receives, so the
    two are not distinguishable in the output; what the setting buys is a
    defined behaviour rather than a failed transformation.
    """
    preprocessor = factory().fit(training_df)
    df_with_unknown_category = unseen_df.assign(InternetService=["Satellite", "Fiber optic"])

    transformed = as_dense(preprocessor.transform(df_with_unknown_category))

    indicator_columns = [
        position
        for position, name in enumerate(preprocessor.get_feature_names_out())
        if name.startswith(f"{CATEGORICAL_BRANCH}__InternetService_")
    ]
    assert indicator_columns
    assert transformed[0, indicator_columns].tolist() == [0.0] * len(indicator_columns)


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_transformed_output_is_a_numeric_feature_matrix(factory, training_df):
    """What comes out is a finite numeric matrix a model can be fitted on.

    Its width is whatever the fitted encoder produced, which depends on how
    many categories the training records held; the row count is the one thing
    fixed regardless.
    """
    preprocessor = factory()

    transformed = as_dense(preprocessor.fit_transform(training_df))

    assert np.issubdtype(transformed.dtype, np.floating)
    assert transformed.shape == (len(training_df), len(preprocessor.get_feature_names_out()))
    assert np.isfinite(transformed).all()


def test_standardised_amounts_are_centred_on_the_training_records(training_df):
    """The logistic preprocessor reports its amounts in standard deviations.

    Fitted and transformed on the same records, each standardised column has a
    mean of zero and a standard deviation of one, which is what puts the three
    amounts on comparable terms in a weighted sum.
    """
    preprocessor = create_logistic_preprocessor()

    transformed = as_dense(preprocessor.fit_transform(training_df))
    standardised_amounts = transformed[:, : len(EXPECTED_NUMERICAL_FEATURES)]

    assert standardised_amounts.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert standardised_amounts.std(axis=0) == pytest.approx([1.0, 1.0, 1.0])


# --- Guarantees about the caller's frame ---


@pytest.mark.parametrize("factory", FACTORIES, ids=["logistic", "tree"])
def test_input_dataframe_is_not_modified(factory, training_df, unseen_df):
    """Fitting and transforming read the frames they are given and write nothing back.

    The missing amount in the transformed frame is filled in the output only;
    the caller's copy still records no amount.
    """
    df_with_missing_amounts = unseen_df.assign(tenure=[np.nan, 60])
    original_training_df = training_df.copy(deep=True)
    original_unseen_df = df_with_missing_amounts.copy(deep=True)

    preprocessor = factory().fit(training_df)
    preprocessor.transform(df_with_missing_amounts)

    assert_frame_equal(training_df, original_training_df)
    assert_frame_equal(df_with_missing_amounts, original_unseen_df)
