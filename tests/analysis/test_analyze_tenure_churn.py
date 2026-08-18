"""Unit tests for the tenure vs. churn analysis.

These tests establish the public contract of ``analyze_tenure_churn``: the
populations it reports, the statistics it derives from each one, how it
accounts for missing tenure, what it does when a population is empty, and the
input it refuses.

Every expectation below is calculated by hand from the small fixtures defined
in this module, so a failure points at the implementation rather than at a
figure copied from the real dataset. Only observable behaviour is tested; the
helpers the function uses internally are not called directly.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from churner.analysis.analyze_tenure_churn import analyze_tenure_churn

# --- Controlled fixture population ---
# Tenure values chosen so every statistic is exact in binary floating point and
# checkable by hand:
#
#   Churn  tenure observations  missing  mean  median  Q1  Q3  IQR  min  max
#   No     0, 2, 4, 6, 8              1   4.0     4.0   2   6    4    0    8
#   Yes    1, 3, 5                    1   3.0     3.0   2   4    2    1    5
#
# The "No" population deliberately includes a recorded tenure of 0 alongside a
# missing tenure, so the two cannot be confused for one another. Quartiles use
# linear interpolation between observations, which is why the "Yes" quartiles
# fall between recorded values.
TENURE_CHURN_ROWS = [
    (0.0, "No"),
    (2.0, "No"),
    (4.0, "No"),
    (6.0, "No"),
    (8.0, "No"),
    (None, "No"),
    (1.0, "Yes"),
    (3.0, "Yes"),
    (5.0, "Yes"),
    (None, "Yes"),
]


@pytest.fixture
def tenure_churn_df() -> pd.DataFrame:
    """Build the controlled population described above.

    Both churn statuses hold several tenure observations plus one missing
    value, so a single fixture exercises the statistics and the missing-value
    accounting at the same time.
    """
    return pd.DataFrame(TENURE_CHURN_ROWS, columns=["tenure", "Churn"])


# --- Normal tenure analysis ---


def test_summary_matches_manual_calculation(tenure_churn_df):
    """Both populations carry hand-checkable counts and statistics.

    Comparing whole frames also pins down the result schema, the column order,
    and the reporting order of the two populations, which runs retained first.
    """
    result = analyze_tenure_churn(tenure_churn_df)

    expected = pd.DataFrame(
        {
            "churn_status": ["No", "Yes"],
            "customer_count": [6, 4],
            "missing_tenure_count": [1, 1],
            "valid_tenure_count": [5, 3],
            "mean_tenure": [4.0, 3.0],
            "median_tenure": [4.0, 3.0],
            "q1_tenure": [2.0, 2.0],
            "q3_tenure": [6.0, 4.0],
            "iqr_tenure": [4.0, 2.0],
            "min_tenure": [0.0, 1.0],
            "max_tenure": [8.0, 5.0],
        }
    )

    assert_frame_equal(result, expected)


def test_integer_tenure_column_is_summarized(tenure_churn_df):
    """A whole-month integer column, as the raw dataset supplies, is accepted.

    The statistics are reported as floats regardless, so a summary computed
    from integer months has the same shape as one computed from floats.
    """
    df = pd.DataFrame({"tenure": [1, 3, 10, 20], "Churn": ["No", "No", "Yes", "Yes"]})

    result = analyze_tenure_churn(df).set_index("churn_status")

    assert result.loc["No", "mean_tenure"] == pytest.approx(2.0)
    assert result.loc["Yes", "mean_tenure"] == pytest.approx(15.0)
    assert result.loc["Yes", "iqr_tenure"] == pytest.approx(5.0)


# --- Tenure observations that are easily mishandled ---


def test_missing_tenure_is_counted_and_excluded_from_statistics():
    """A missing tenure is counted, and never contributes as a zero.

    Filling the gap with 0 would drag the mean below every observed value and
    report a minimum nobody was recorded with, so both are checked here.
    """
    df = pd.DataFrame({"tenure": [10.0, 20.0, None], "Churn": ["Yes", "Yes", "Yes"]})

    churned = analyze_tenure_churn(df).set_index("churn_status").loc["Yes"]

    assert churned["customer_count"] == 3
    assert churned["missing_tenure_count"] == 1
    assert churned["valid_tenure_count"] == 2
    assert churned["mean_tenure"] == pytest.approx(15.0)
    assert churned["min_tenure"] == pytest.approx(10.0)


def test_zero_tenure_is_a_valid_observation():
    """A recorded tenure of 0 is described, not treated as absent.

    Zero months is a real observation in this dataset, and the distinction
    between it and a missing value is what the count columns exist to keep.
    """
    df = pd.DataFrame({"tenure": [0.0, 0.0, 4.0], "Churn": ["No", "Yes", "Yes"]})

    result = analyze_tenure_churn(df).set_index("churn_status")

    retained = result.loc["No"]
    assert retained["valid_tenure_count"] == 1
    assert retained["missing_tenure_count"] == 0
    assert retained["mean_tenure"] == pytest.approx(0.0)
    assert retained["median_tenure"] == pytest.approx(0.0)
    assert retained["min_tenure"] == pytest.approx(0.0)
    assert retained["max_tenure"] == pytest.approx(0.0)

    churned = result.loc["Yes"]
    assert churned["valid_tenure_count"] == 2
    assert churned["min_tenure"] == pytest.approx(0.0)
    assert churned["mean_tenure"] == pytest.approx(2.0)


def test_population_counts_reconcile(tenure_churn_df):
    """Each population accounts for its customers exactly once.

    Every customer either contributes a tenure observation or is counted as
    missing one, so the two counts must add back up to the population size.
    """
    result = analyze_tenure_churn(tenure_churn_df)

    reconciled = result["valid_tenure_count"] + result["missing_tenure_count"]
    assert list(reconciled) == list(result["customer_count"])
    assert int(result["customer_count"].sum()) == len(tenure_churn_df)


def test_customer_with_missing_churn_joins_neither_population():
    """A customer with no target value is counted in neither population.

    Their tenure is real, but which distribution it belongs to is unknown, so
    it is left out of both rather than guessed into one.
    """
    df = pd.DataFrame({"tenure": [1.0, 2.0, 99.0], "Churn": ["No", "Yes", None]})

    result = analyze_tenure_churn(df)

    assert int(result["customer_count"].sum()) == 2
    assert result["max_tenure"].max() == pytest.approx(2.0)


# --- Populations with nothing to describe ---


def test_churn_group_without_customers_is_reported_with_undefined_statistics():
    """A population nobody belongs to keeps its row, with no statistics.

    The row is what makes the comparison stable in shape, and the ``NaN``
    statistics are what stop "no observation" from reading as "a tenure of
    zero months".
    """
    df = pd.DataFrame({"tenure": [1.0, 2.0], "Churn": ["No", "No"]})

    churned = analyze_tenure_churn(df).set_index("churn_status").loc["Yes"]

    assert churned["customer_count"] == 0
    assert churned["missing_tenure_count"] == 0
    assert churned["valid_tenure_count"] == 0
    assert churned[["mean_tenure", "median_tenure", "q1_tenure", "q3_tenure"]].isna().all()
    assert churned[["iqr_tenure", "min_tenure", "max_tenure"]].isna().all()


def test_empty_dataframe_returns_both_populations_with_zero_counts():
    """A frame with the required columns but no rows is summarised, not refused.

    There is nothing to describe, but the two populations still exist as
    reporting categories, so the caller gets the usual shape back.
    """
    df = pd.DataFrame({"tenure": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")})

    result = analyze_tenure_churn(df)

    assert list(result["churn_status"]) == ["No", "Yes"]
    count_columns = ["customer_count", "missing_tenure_count", "valid_tenure_count"]
    assert (result[count_columns] == 0).all().all()

    statistic_columns = result.columns.drop(["churn_status", *count_columns])
    assert result[statistic_columns].isna().all().all()


# --- Input validation ---


@pytest.mark.parametrize("absent_column", ["tenure", "Churn"])
def test_missing_required_column_raises_value_error(tenure_churn_df, absent_column):
    """Either required column being absent is refused rather than worked around."""
    df = tenure_churn_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        analyze_tenure_churn(df)


@pytest.mark.parametrize("non_numeric_tenure", [["12", "8"], ["twelve", "eight"]])
def test_non_numeric_tenure_raises_value_error(non_numeric_tenure):
    """Text tenure is refused instead of being parsed or coerced.

    Digits stored as text are refused alongside plain words: deciding how to
    convert a column, and what an unparseable entry means, belongs to a
    deliberate cleaning step rather than to a descriptive summary.
    """
    df = pd.DataFrame({"tenure": non_numeric_tenure, "Churn": ["Yes", "No"]})

    with pytest.raises(ValueError, match="tenure"):
        analyze_tenure_churn(df)


@pytest.mark.parametrize("unexpected_value", ["Unknown", "yes", 1])
def test_unexpected_churn_value_raises_value_error(unexpected_value):
    """A target value outside the documented domain is refused.

    Silently reading it as "not churned" would move an unclassifiable customer
    into the retained population and quietly distort the comparison, so the
    offending value is reported instead.
    """
    df = pd.DataFrame({"tenure": [1.0, 2.0, 3.0], "Churn": ["Yes", "No", unexpected_value]})

    with pytest.raises(ValueError, match="Churn"):
        analyze_tenure_churn(df)


# --- Observational guarantees ---


def test_input_dataframe_is_not_modified(tenure_churn_df):
    """The analysis reads the frame it is given and writes nothing back.

    Missing tenure values in the fixture are the risk here: they must stay
    missing rather than being filled, and no helper column may be left behind.
    """
    original_df = tenure_churn_df.copy(deep=True)

    analyze_tenure_churn(tenure_churn_df)

    assert_frame_equal(tenure_churn_df, original_df)
