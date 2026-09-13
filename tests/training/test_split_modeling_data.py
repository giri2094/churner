"""Unit tests for the modelling-data train/test split.

These tests establish the public contract of ``split_modeling_data``: the
sizes of the two partitions, the objects it hands back, that the same
records produce the same split on every call, that the target's class
proportions survive the split, and that the caller's frames are left
untouched.

The split configuration is written out here rather than imported from the
module under test, so a test fails if the evaluation boundary changes
rather than agreeing with itself. Every expectation is hand-calculable from
the twenty-customer fixture defined in this module.

Nothing here trains a model, constructs a pipeline, or measures one.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.model_selection import train_test_split

from churner.training.split_modeling_data import split_modeling_data

# Restated here on purpose: comparing the split against these values is what
# makes the evaluation boundary testable, which comparing it against the
# constants it was built from would not.
EXPECTED_TEST_SIZE = 0.2
EXPECTED_RANDOM_STATE = 42

# --- Controlled fixture population ---
# Twenty customers holding the shape ``prepare_modeling_data`` hands on, with
# the target kept beside them as a separate series. Four of the twenty
# churned, so a 20% test split lands on whole numbers and the stratified
# class counts are checkable by hand:
#
#   partition   n    No   Yes   Yes rate
#   all        20    16     4      0.20
#   train      16    13     3      0.1875
#   test        4     3     1      0.25
#
# The train and test Yes rates sit 1.25 and 5 percentage points from the
# population rate; that gap is the rounding a 20% draw from four churned
# records cannot avoid. Tenure is unique per row so a record that crossed
# partitions, or a predictor that came unstuck from its label, would be
# visible.
CUSTOMER_COUNT = 20
EXPECTED_TRAIN_COUNT = 16
EXPECTED_TEST_COUNT = 4

RETAINED_COUNT = 16
CHURNED_COUNT = 4

FEATURE_COLUMNS = ["tenure", "MonthlyCharges"]

CUSTOMERS = {
    "tenure": list(range(CUSTOMER_COUNT)),
    "MonthlyCharges": [20.05 + (10.0 * i) for i in range(CUSTOMER_COUNT)],
}

CHURN_LABELS = (["No"] * RETAINED_COUNT) + (["Yes"] * CHURNED_COUNT)

# The largest class-rate gap the 20-row split produces is five percentage
# points on the test set (0.25 against 0.20). The bound is that observed
# gap plus a little room, not a licence for an unstratified split.
MAX_CLASS_RATE_GAP = 0.06


@pytest.fixture
def features_df() -> pd.DataFrame:
    """Build the controlled twenty-customer predictor frame described above."""
    return pd.DataFrame(CUSTOMERS)[FEATURE_COLUMNS]


@pytest.fixture
def target(features_df: pd.DataFrame) -> pd.Series:
    """Build the churn labels of the fixture, aligned to the predictor index."""
    return pd.Series(CHURN_LABELS, index=features_df.index, name="Churn")


def class_rate(labels: pd.Series, churn_value: str = "Yes") -> float:
    """Return the share of one label in a target series."""
    return (labels == churn_value).mean()


# --- Sizes ---


def test_train_and_test_have_the_expected_sizes(features_df, target):
    """80% of the rows train, 20% test, as whole counts on this fixture.

    Twenty rows and a 20% test size leave sixteen and four, so a split that
    rounded differently, or that ignored ``test_size`` altogether, cannot
    hide behind a fractional remainder.
    """
    X_train, X_test, y_train, y_test = split_modeling_data(features_df, target)

    assert len(X_train) == EXPECTED_TRAIN_COUNT
    assert len(X_test) == EXPECTED_TEST_COUNT
    assert len(y_train) == EXPECTED_TRAIN_COUNT
    assert len(y_test) == EXPECTED_TEST_COUNT
    assert len(X_train) + len(X_test) == CUSTOMER_COUNT


# --- Structure ---


def test_split_returns_four_objects_in_sklearn_order(features_df, target):
    """The return is the four partitions, in the order sklearn itself uses."""
    result = split_modeling_data(features_df, target)

    assert len(result) == 4
    X_train, X_test, y_train, y_test = result
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_predictors_keep_their_columns(features_df, target):
    """Both feature frames keep the columns they were given, in order."""
    X_train, X_test, _, _ = split_modeling_data(features_df, target)

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS


def test_target_keeps_its_name(features_df, target):
    """The label series is still called ``Churn`` after the split."""
    _, _, y_train, y_test = split_modeling_data(features_df, target)

    assert y_train.name == "Churn"
    assert y_test.name == "Churn"


def test_predictors_and_target_stay_aligned(features_df, target):
    """Each row of predictors is paired with the outcome of the same customer.

    The split shuffles row order, so the index is what keeps a customer's
    predictors attached to their own label.
    """
    X_train, X_test, y_train, y_test = split_modeling_data(features_df, target)

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert_series_equal(y_train, target.loc[X_train.index])
    assert_series_equal(y_test, target.loc[X_test.index])


def test_train_and_test_form_a_partition(features_df, target):
    """Every original row lands in exactly one of the two partitions."""
    X_train, X_test, y_train, y_test = split_modeling_data(features_df, target)

    train_index = set(X_train.index)
    test_index = set(X_test.index)

    assert train_index.isdisjoint(test_index)
    assert train_index.union(test_index) == set(features_df.index)
    assert set(y_train.index) == train_index
    assert set(y_test.index) == test_index


# --- Reproducibility ---


def test_split_is_reproducible(features_df, target):
    """Two calls on the same records produce the same partition.

    A seeded split is what lets a later measurement be compared to an
    earlier one; an unseeded shuffle would make that comparison meaningless.
    """
    first = split_modeling_data(features_df, target)
    second = split_modeling_data(features_df, target)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])
    assert_series_equal(first[2], second[2])
    assert_series_equal(first[3], second[3])


def test_split_matches_the_configured_sklearn_split(features_df, target):
    """The partition is the one sklearn produces at test_size=0.2 and seed 42.

    Comparing against an independent ``train_test_split`` call is what pins
    the three configured arguments; reproducibility alone would still pass
    under a different seed.
    """
    expected = train_test_split(
        features_df,
        target,
        test_size=EXPECTED_TEST_SIZE,
        random_state=EXPECTED_RANDOM_STATE,
        stratify=target,
    )
    actual = split_modeling_data(features_df, target)

    assert_frame_equal(actual[0], expected[0])
    assert_frame_equal(actual[1], expected[1])
    assert_series_equal(actual[2], expected[2])
    assert_series_equal(actual[3], expected[3])


# --- Stratification ---


def test_stratification_preserves_class_proportions(features_df, target):
    """Each partition's churn rate stays close to the population rate.

    An unstratified split of twenty records can easily put all four churned
    customers on one side. Stratifying keeps both classes in both partitions
    and keeps the rates within the rounding the fixture cannot avoid.
    """
    _, _, y_train, y_test = split_modeling_data(features_df, target)
    population_rate = class_rate(target)

    assert class_rate(y_train) == pytest.approx(population_rate, abs=MAX_CLASS_RATE_GAP)
    assert class_rate(y_test) == pytest.approx(population_rate, abs=MAX_CLASS_RATE_GAP)
    assert set(y_train.unique()) == {"No", "Yes"}
    assert set(y_test.unique()) == {"No", "Yes"}


# --- Guarantees about the caller's frames ---


def test_input_frames_are_not_modified(features_df, target):
    """The split reads the frames it is given and writes nothing back."""
    original_features = features_df.copy(deep=True)
    original_target = target.copy(deep=True)

    split_modeling_data(features_df, target)

    assert_frame_equal(features_df, original_features)
    assert_series_equal(target, original_target)


def test_results_are_new_objects(features_df, target):
    """Editing a partition cannot reach back into the caller's frames."""
    original_features = features_df.copy(deep=True)
    original_target = target.copy(deep=True)

    X_train, X_test, y_train, y_test = split_modeling_data(features_df, target)
    X_train.iloc[0, 0] = 999
    X_test.iloc[0, 0] = 999
    y_train.iloc[0] = "MUTATED"
    y_test.iloc[0] = "MUTATED"

    assert_frame_equal(features_df, original_features)
    assert_series_equal(target, original_target)
