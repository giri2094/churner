"""Unit tests for the targeted 0-6 month tenure churn analysis.

These tests establish the public contract of ``analyze_early_tenure_churn``:
which customers the slice holds, the counts and rates derived from them, how
the slice is compared against the overall baseline, and the input it refuses.

The boundary tests matter most. The slice is defined inclusively, so a tenure
of exactly 6 months belongs to it and a tenure of 7 months does not; an
off-by-one there would change every figure the analysis reports without
producing an obviously wrong-looking result.

Every expectation below is calculated by hand from the small fixtures defined
in this module, so a failure points at the implementation rather than at a
figure copied from the real dataset.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from churner.analysis.analyze_early_tenure_churn import analyze_early_tenure_churn

# --- Controlled fixture population ---
# Hand-calculated expectations, used by the assertions below:
#
#   slice (tenure 0-6):  5 customers, 3 churned, 2 retained -> rate 0.60
#   whole dataset:      10 customers, 4 churned            -> rate 0.40
#   rate difference:    0.60 - 0.40 = 0.20
#   relative rate:      0.60 / 0.40 = 1.50
#   population share:   5 / 10 = 0.50
#   churn contribution: 3 /  4 = 0.75
#
# The slice deliberately holds both boundary values with both outcomes, and the
# customers just outside it include a churned one at 7 months, so an inclusive
# upper boundary of 7 would change the counts rather than pass unnoticed.
EARLY_TENURE_ROWS = [
    (0, "Yes"),
    (3, "Yes"),
    (6, "Yes"),
    (0, "No"),
    (6, "No"),
    # Outside the slice. The churned customer at 7 months is the one the upper
    # boundary has to exclude.
    (7, "Yes"),
    (7, "No"),
    (12, "No"),
    (72, "No"),
    # Negative tenure is unusual rather than missing. It is kept in the dataset
    # and simply does not satisfy the slice condition.
    (-1, "No"),
]


@pytest.fixture
def early_tenure_df() -> pd.DataFrame:
    """Build the controlled population described above."""
    return pd.DataFrame(EARLY_TENURE_ROWS, columns=["tenure", "Churn"])


def analyze_single_row(df: pd.DataFrame) -> pd.Series:
    """Run the analysis and return its single result row."""
    result = analyze_early_tenure_churn(df)
    assert len(result) == 1
    return result.iloc[0]


# --- Slice boundaries ---


@pytest.mark.parametrize("tenure_months", [0, 1, 5, 6])
def test_tenure_within_the_boundaries_is_included(tenure_months):
    """The slice runs from 0 through 6 months inclusive.

    A tenure of 0 is a real observation rather than an absent one, and 6 is
    inside the region by definition, so both ends belong to the slice.
    """
    df = pd.DataFrame({"tenure": [tenure_months, 40], "Churn": ["Yes", "No"]})

    result = analyze_single_row(df)

    assert result["slice_customer_count"] == 1
    assert result["slice_churned_count"] == 1


@pytest.mark.parametrize("tenure_months", [7, 8, 72, -1])
def test_tenure_outside_the_boundaries_is_excluded(tenure_months):
    """Anything beyond either boundary stays out of the slice.

    Seven months is the case the inclusive upper boundary turns on. A negative
    tenure is excluded for the same reason as any other value outside the
    region, not because it looks wrong: it is left in the dataset untouched.
    """
    df = pd.DataFrame({"tenure": [tenure_months, 40], "Churn": ["Yes", "No"]})

    result = analyze_single_row(df)

    assert result["slice_customer_count"] == 0
    assert result["slice_churned_count"] == 0
    assert result["overall_customer_count"] == 2


def test_missing_tenure_does_not_enter_the_slice():
    """A customer with no recorded tenure cannot be shown to be in the region.

    They stay counted in the overall population, since the baseline is measured
    over the whole dataset, but they are not assumed into the slice.
    """
    df = pd.DataFrame({"tenure": [2.0, None], "Churn": ["Yes", "Yes"]})

    result = analyze_single_row(df)

    assert result["slice_customer_count"] == 1
    assert result["slice_churned_count"] == 1
    assert result["overall_customer_count"] == 2
    assert result["overall_churned_count"] == 2


def test_reported_boundaries_state_the_selected_region(early_tenure_df):
    """The row carries the boundaries it was measured with.

    They are the region this module selected, not anything read from the data,
    so the result stays interpretable on its own.
    """
    result = analyze_single_row(early_tenure_df)

    assert result["slice_min_tenure"] == 0
    assert result["slice_max_tenure"] == 6


# --- Counts and rates ---


def test_result_matches_manual_calculation(early_tenure_df):
    """Every reported figure is the hand-calculated one.

    Comparing the whole frame also pins down the single-row shape, the result
    schema, and the column order.
    """
    result = analyze_early_tenure_churn(early_tenure_df)

    expected = pd.DataFrame(
        {
            "slice_min_tenure": [0],
            "slice_max_tenure": [6],
            "slice_customer_count": [5],
            "slice_churned_count": [3],
            "slice_retained_count": [2],
            "slice_churn_rate": [0.6],
            "slice_population_share": [0.5],
            "slice_churn_contribution": [0.75],
            "overall_customer_count": [10],
            "overall_churned_count": [4],
            "overall_churn_rate": [0.4],
            "rate_difference": [0.2],
            "relative_churn_rate": [1.5],
        }
    )

    assert_frame_equal(result, expected)


def test_slice_population_reconciles(early_tenure_df):
    """The slice's customers are exactly its churned plus its retained."""
    result = analyze_single_row(early_tenure_df)

    assert (
        result["slice_customer_count"]
        == result["slice_churned_count"] + result["slice_retained_count"]
    )


def test_churn_rate_and_churn_contribution_measure_different_things():
    """A region can churn heavily while accounting for little of all churn.

    Here the slice churns at 100% yet holds only 1 of the 5 churned customers,
    so the two quantities cannot be read as interchangeable.
    """
    df = pd.DataFrame(
        {
            "tenure": [1, 30, 40, 50, 60, 70],
            "Churn": ["Yes", "Yes", "Yes", "Yes", "Yes", "No"],
        }
    )

    result = analyze_single_row(df)

    assert result["slice_churn_rate"] == pytest.approx(1.0)
    assert result["slice_churn_contribution"] == pytest.approx(0.2)
    assert result["slice_population_share"] == pytest.approx(1 / 6)


def test_slice_below_the_baseline_reports_a_negative_difference():
    """A slice that churns less than the dataset reads below 1.0 and below 0.

    The direction of both comparisons matters as much as their magnitude, so a
    slice under the baseline is checked as well as one over it.
    """
    df = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 30, 40, 50, 60],
            "Churn": ["Yes", "No", "No", "No", "Yes", "Yes", "Yes", "No"],
        }
    )

    result = analyze_single_row(df)

    assert result["slice_churn_rate"] == pytest.approx(0.25)
    assert result["overall_churn_rate"] == pytest.approx(0.5)
    assert result["rate_difference"] == pytest.approx(-0.25)
    assert result["relative_churn_rate"] == pytest.approx(0.5)


# --- Populations with nothing to measure ---


def test_empty_slice_has_no_churn_rate_but_measurable_shares():
    """A slice holding nobody has no rate, though its shares are measured.

    Its churn rate is undefined for want of a denominator, while its share of
    the population and of all churn are genuinely zero: the dataset does have
    customers and churn, and none of them are here.
    """
    df = pd.DataFrame({"tenure": [30, 40, 50], "Churn": ["Yes", "No", "No"]})

    result = analyze_single_row(df)

    assert result["slice_customer_count"] == 0
    assert result["slice_churned_count"] == 0
    assert result["slice_retained_count"] == 0
    assert pd.isna(result["slice_churn_rate"])
    assert pd.isna(result["rate_difference"])
    assert pd.isna(result["relative_churn_rate"])
    assert result["slice_population_share"] == pytest.approx(0.0)
    assert result["slice_churn_contribution"] == pytest.approx(0.0)
    assert result["overall_churn_rate"] == pytest.approx(1 / 3)


def test_dataset_without_churn_leaves_the_comparisons_undefined():
    """With no churn anywhere, the baseline gives nothing to divide by.

    The slice churn rate is a measured 0.0, but the contribution and the
    relative rate are undefined rather than zero.
    """
    df = pd.DataFrame({"tenure": [1, 2, 30], "Churn": ["No", "No", "No"]})

    result = analyze_single_row(df)

    assert result["slice_churn_rate"] == pytest.approx(0.0)
    assert result["overall_churn_rate"] == pytest.approx(0.0)
    assert result["rate_difference"] == pytest.approx(0.0)
    assert pd.isna(result["slice_churn_contribution"])
    assert pd.isna(result["relative_churn_rate"])


def test_empty_dataframe_reports_zero_counts_and_undefined_rates():
    """A frame with the required columns but no rows is measured, not refused."""
    df = pd.DataFrame({"tenure": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")})

    result = analyze_single_row(df)

    count_columns = [
        "slice_customer_count",
        "slice_churned_count",
        "slice_retained_count",
        "overall_customer_count",
        "overall_churned_count",
    ]
    assert list(result[count_columns]) == [0, 0, 0, 0, 0]

    rate_columns = [
        "slice_churn_rate",
        "slice_population_share",
        "slice_churn_contribution",
        "overall_churn_rate",
        "rate_difference",
        "relative_churn_rate",
    ]
    assert result[rate_columns].isna().all()


# --- Input validation ---


@pytest.mark.parametrize("absent_column", ["tenure", "Churn"])
def test_missing_required_column_raises_value_error(early_tenure_df, absent_column):
    """Either required column being absent is refused rather than worked around."""
    df = early_tenure_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        analyze_early_tenure_churn(df)


@pytest.mark.parametrize("non_numeric_tenure", [["3", "40"], ["three", "forty"]])
def test_non_numeric_tenure_raises_value_error(non_numeric_tenure):
    """Text tenure is refused, since the boundaries are numeric comparisons."""
    df = pd.DataFrame({"tenure": non_numeric_tenure, "Churn": ["Yes", "No"]})

    with pytest.raises(ValueError, match="tenure"):
        analyze_early_tenure_churn(df)


@pytest.mark.parametrize("unexpected_value", ["Unknown", "yes", 1])
def test_unexpected_churn_value_raises_value_error(unexpected_value):
    """A target value outside the documented domain is refused, as elsewhere.

    Counting it as retained would inflate the slice's retained count and
    understate its churn rate, which is the quantity under investigation.
    """
    df = pd.DataFrame({"tenure": [1, 2, 3], "Churn": ["Yes", "No", unexpected_value]})

    with pytest.raises(ValueError, match="Churn"):
        analyze_early_tenure_churn(df)


def test_missing_churn_inside_the_slice_raises_value_error():
    """An unclassifiable customer inside the slice stops the measurement.

    They would otherwise sit in the denominator of the slice churn rate while
    counting as neither churned nor retained, leaving the slice counts unable
    to reconcile and the rate quietly understated.
    """
    df = pd.DataFrame({"tenure": [1, 2, 30], "Churn": ["Yes", None, "No"]})

    with pytest.raises(ValueError, match="no recorded 'Churn' value"):
        analyze_early_tenure_churn(df)


def test_missing_churn_outside_the_slice_is_tolerated():
    """Outside the slice, an unclassifiable customer only affects the baseline.

    They stay in the overall population, as the established baseline counts
    every customer, and they are not counted as churned.
    """
    df = pd.DataFrame({"tenure": [1, 2, 30], "Churn": ["Yes", "No", None]})

    result = analyze_single_row(df)

    assert result["slice_customer_count"] == 2
    assert result["slice_churn_rate"] == pytest.approx(0.5)
    assert result["overall_customer_count"] == 3
    assert result["overall_churned_count"] == 1
    assert result["overall_churn_rate"] == pytest.approx(1 / 3)


# --- Observational guarantees ---


def test_input_dataframe_is_not_modified(early_tenure_df):
    """The analysis reads the frame it is given and writes nothing back.

    In particular the customers outside the slice are still there afterwards,
    including the negative tenure, which is excluded from the slice rather than
    removed from the data.
    """
    original_df = early_tenure_df.copy(deep=True)

    analyze_early_tenure_churn(early_tenure_df)

    assert_frame_equal(early_tenure_df, original_df)
