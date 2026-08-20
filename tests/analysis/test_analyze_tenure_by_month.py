"""Unit tests for the per-tenure-month churn analysis.

These tests establish the public contract of ``analyze_tenure_by_month``: which
tenure months get a row, the counts and rate each row reports, the order the
rows come in, and the input the analysis refuses.

The tests that matter most are the ones about which rows exist at all. The
analysis reports observed tenure months, so a month nobody was recorded at must
not appear; an implementation that reindexed over a continuous range would still
produce plausible-looking output while describing months the data never held.

Every expectation below is calculated by hand from the small fixtures defined in
this module, so a failure points at the implementation rather than at a figure
copied from the real dataset.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from churner.analysis.analyze_tenure_by_month import analyze_tenure_by_month

RESULT_COLUMNS = [
    "tenure_month",
    "customer_count",
    "churned_count",
    "retained_count",
    "churn_rate",
]

# --- Controlled fixture population ---
# Hand-calculated expectations, used by the assertions below:
#
#   month 0: 1 customer,  0 churned, 1 retained -> rate 0
#   month 1: 2 customers, 1 churned, 1 retained -> rate 0.5
#   month 2: 3 customers, 2 churned, 1 retained -> rate 2/3
#
# The three months differ in customer count as well as in rate, so a row cannot
# match by coincidence, and month 2 gives a rate that no rounding to two
# decimals would reproduce exactly.
MONTHLY_ROWS = [
    (0, "No"),
    (1, "Yes"),
    (1, "No"),
    (2, "Yes"),
    (2, "Yes"),
    (2, "No"),
]


@pytest.fixture
def monthly_df() -> pd.DataFrame:
    """Build the controlled population described above."""
    return pd.DataFrame(MONTHLY_ROWS, columns=["tenure", "Churn"])


# --- Aggregation ---


def test_result_matches_manual_calculation(monthly_df):
    """Every reported figure is the hand-calculated one.

    Comparing the whole frame also pins down the row count, the result schema,
    the column order, and the ascending row order in a single assertion.
    """
    result = analyze_tenure_by_month(monthly_df)

    expected = pd.DataFrame(
        {
            "tenure_month": [0, 1, 2],
            "customer_count": [1, 2, 3],
            "churned_count": [0, 1, 2],
            "retained_count": [1, 1, 1],
            "churn_rate": [0.0, 0.5, 2 / 3],
        }
    )

    assert_frame_equal(result, expected)


def test_customers_at_the_same_tenure_share_one_row(monthly_df):
    """Each tenure month is reported once, however many customers hold it."""
    result = analyze_tenure_by_month(monthly_df)

    assert result["tenure_month"].tolist() == [0, 1, 2]
    assert not result["tenure_month"].duplicated().any()


def test_churn_rate_is_a_proportion_not_a_percentage():
    """Rates run from 0 to 1, so a fully churned month reads 1.0 and not 100.

    Both extremes are checked: a month where everyone stayed is a measured 0.0
    rather than a missing value.
    """
    df = pd.DataFrame({"tenure": [1, 1, 5, 5], "Churn": ["Yes", "Yes", "No", "No"]})

    result = analyze_tenure_by_month(df).set_index("tenure_month")

    assert result.loc[1, "churn_rate"] == pytest.approx(1.0)
    assert result.loc[5, "churn_rate"] == pytest.approx(0.0)


def test_churn_rate_follows_from_the_counts(monthly_df):
    """The rate is each month's churned share of its own customers.

    Deriving it from the reported counts rather than from a literal keeps the
    two from disagreeing without the test noticing.
    """
    result = analyze_tenure_by_month(monthly_df)

    expected_rate = result["churned_count"] / result["customer_count"]

    assert result["churn_rate"].tolist() == pytest.approx(expected_rate.tolist())
    assert result["churn_rate"].between(0, 1).all()


def test_every_row_reconciles(monthly_df):
    """A month's customers are exactly its churned plus its retained."""
    result = analyze_tenure_by_month(monthly_df)

    assert result["customer_count"].tolist() == (
        result["churned_count"] + result["retained_count"]
    ).tolist()


def test_result_carries_no_columns_beyond_the_contract(monthly_df):
    """The frame holds these five columns and nothing else.

    Comparisons against a baseline, population shares, and tenure bands belong
    to other analyses; this one describes each observed month on its own terms.
    """
    result = analyze_tenure_by_month(monthly_df)

    assert list(result.columns) == RESULT_COLUMNS


# --- Which tenure months get a row ---


def test_tenure_zero_gets_its_own_row():
    """A recorded tenure of zero is an observation, not an absent one."""
    df = pd.DataFrame({"tenure": [0, 0, 4], "Churn": ["Yes", "No", "No"]})

    result = analyze_tenure_by_month(df).set_index("tenure_month")

    assert result.loc[0, "customer_count"] == 2
    assert result.loc[0, "churned_count"] == 1
    assert result.loc[0, "churn_rate"] == pytest.approx(0.5)


def test_highest_observed_tenure_gets_its_own_row():
    """The top of the observed range is reported like any other month.

    The 72-month maximum is a property of this dataset rather than a rule, so
    nothing about it is treated as a boundary.
    """
    df = pd.DataFrame({"tenure": [1, 72], "Churn": ["Yes", "No"]})

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [1, 72]
    assert result.iloc[-1]["customer_count"] == 1
    assert result.iloc[-1]["retained_count"] == 1


def test_unobserved_tenure_months_are_not_fabricated():
    """Only months present in the data appear, with no gaps filled in.

    A row for a month nobody was recorded at would describe the dataset's
    coverage rather than churn, and would carry a rate over no customers.
    """
    df = pd.DataFrame({"tenure": [0, 2, 5, 7], "Churn": ["No", "Yes", "No", "Yes"]})

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [0, 2, 5, 7]
    assert len(result) == 4


def test_rows_are_ordered_by_ascending_tenure_month():
    """The months read upward regardless of the input's row order."""
    df = pd.DataFrame(
        {
            "tenure": [30, 2, 71, 0, 2, 9],
            "Churn": ["Yes", "No", "No", "Yes", "Yes", "No"],
        }
    )

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [0, 2, 9, 30, 71]
    assert result.index.tolist() == list(range(len(result)))


def test_unusual_tenure_is_reported_rather_than_removed():
    """A negative tenure gets a row of its own, at the bottom of the order.

    It is unusual rather than missing, and an exploratory summary that dropped
    it would hide exactly what the summary exists to reveal.
    """
    df = pd.DataFrame({"tenure": [-1, 3], "Churn": ["Yes", "No"]})

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [-1, 3]


# --- Missing values ---


def test_missing_tenure_enters_no_month():
    """A customer with no recorded tenure belongs to no numerical month.

    Placing them anywhere would attribute churn to a month they were never
    observed at, so they are left out of the aggregation entirely.
    """
    df = pd.DataFrame({"tenure": [2.0, 2.0, None], "Churn": ["Yes", "No", "Yes"]})

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [2.0]
    assert result.iloc[0]["customer_count"] == 2
    assert result.iloc[0]["churned_count"] == 1
    assert result.iloc[0]["churn_rate"] == pytest.approx(0.5)


def test_missing_churn_at_a_recorded_tenure_raises_value_error():
    """An unclassifiable customer inside a month stops the measurement.

    They would otherwise sit in the denominator of that month's churn rate
    while counting as neither churned nor retained, leaving the row unable to
    reconcile and the rate quietly understated. This matches how the
    early-tenure analysis treats an unclassifiable customer in its slice.
    """
    df = pd.DataFrame({"tenure": [1, 2, 3], "Churn": ["Yes", None, "No"]})

    with pytest.raises(ValueError, match="no recorded 'Churn' value"):
        analyze_tenure_by_month(df)


def test_missing_churn_without_a_tenure_is_tolerated():
    """A customer in no month cannot unbalance one.

    Their tenure is missing, so they are already outside every row and their
    unknown churn status has nothing to disturb.
    """
    df = pd.DataFrame({"tenure": [1.0, None], "Churn": ["Yes", None]})

    result = analyze_tenure_by_month(df)

    assert result["tenure_month"].tolist() == [1.0]
    assert result.iloc[0]["customer_count"] == 1


# --- Populations with nothing to aggregate ---


def test_empty_dataframe_returns_the_empty_schema():
    """A frame with the required columns but no rows is measured, not refused."""
    df = pd.DataFrame({"tenure": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")})

    result = analyze_tenure_by_month(df)

    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == 0


def test_all_tenure_missing_returns_the_empty_schema():
    """With no tenure observed anywhere, there is no month to report.

    The result is still a well-formed frame, so a caller needs no special case
    for it.
    """
    df = pd.DataFrame({"tenure": pd.Series([None, None], dtype="float64"), "Churn": ["Yes", "No"]})

    result = analyze_tenure_by_month(df)

    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == 0


def test_empty_result_keeps_the_populated_column_types():
    """The empty frame's counts stay integers and its rate stays a float.

    An empty result inferred as ``object`` would break arithmetic downstream
    for callers that treat both cases alike.
    """
    empty_result = analyze_tenure_by_month(
        pd.DataFrame({"tenure": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")})
    )
    populated_result = analyze_tenure_by_month(
        pd.DataFrame({"tenure": [1.0], "Churn": ["Yes"]})
    )

    count_columns = ["customer_count", "churned_count", "retained_count"]
    assert_frame_equal(
        empty_result[count_columns].dtypes.to_frame(),
        populated_result[count_columns].dtypes.to_frame(),
    )
    assert empty_result["churn_rate"].dtype == populated_result["churn_rate"].dtype


# --- Input validation ---


@pytest.mark.parametrize("absent_column", ["tenure", "Churn"])
def test_missing_required_column_raises_value_error(monthly_df, absent_column):
    """Either required column being absent is refused rather than worked around."""
    df = monthly_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        analyze_tenure_by_month(df)


@pytest.mark.parametrize("non_numeric_tenure", [["3", "40"], ["three", "forty"]])
def test_non_numeric_tenure_raises_value_error(non_numeric_tenure):
    """Text tenure is refused, as it is throughout the tenure analyses.

    Deciding what an unparseable entry means is a data-cleaning decision, not
    one this module makes.
    """
    df = pd.DataFrame({"tenure": non_numeric_tenure, "Churn": ["Yes", "No"]})

    with pytest.raises(ValueError, match="tenure"):
        analyze_tenure_by_month(df)


@pytest.mark.parametrize("unexpected_value", ["Unknown", "yes", 1])
def test_unexpected_churn_value_raises_value_error(unexpected_value):
    """A target value outside the documented domain is refused, as elsewhere.

    Counting it as retained would inflate that month's retained count and
    understate its churn rate, which is the quantity under investigation.
    """
    df = pd.DataFrame({"tenure": [1, 2, 3], "Churn": ["Yes", "No", unexpected_value]})

    with pytest.raises(ValueError, match="Churn"):
        analyze_tenure_by_month(df)


# --- Observational guarantees ---


def test_input_dataframe_is_not_modified(monthly_df):
    """The analysis reads the frame it is given and writes nothing back."""
    original_df = monthly_df.copy(deep=True)

    analyze_tenure_by_month(monthly_df)

    assert_frame_equal(monthly_df, original_df)


def test_input_dataframe_with_missing_values_is_not_modified():
    """Nothing is filled in or dropped in place, missing tenure included."""
    df = pd.DataFrame({"tenure": [1.0, None, 3.0], "Churn": ["Yes", "No", "No"]})
    original_df = df.copy(deep=True)

    analyze_tenure_by_month(df)

    assert_frame_equal(df, original_df)


def test_result_is_a_new_frame(monthly_df):
    """The result is a frame of its own, so editing it cannot touch the input."""
    result = analyze_tenure_by_month(monthly_df)

    assert isinstance(result, pd.DataFrame)
    assert result is not monthly_df
