"""Unit tests for the contract-level churn analysis.

These tests establish the public contract of ``analyze_contract_churn``: the
groups it reports, the counts and rates it derives, how it surfaces contract
values that depart from the expected categories, and the input it refuses.

Every expectation below is calculated by hand from the small fixture defined in
this module, so a failure points at the implementation rather than at a figure
copied from the real dataset. The internal helper the function uses is not
tested directly; only observable behaviour is.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from churner.analysis.analyze_contract_churn import analyze_contract_churn

# Overall churn baseline supplied to the function under test. It is deliberately
# not any group's own churn rate, so a difference of zero cannot pass by
# coincidence, and it stands in for the baseline a caller would pass in rather
# than being recomputed from the fixture.
OVERALL_CHURN_RATE = 30.0

# --- Controlled fixture population ---
# Hand-calculated expectations, used by the assertions below:
#
#   Contract        customers  churned  churn rate  vs. 30.0 baseline
#   Month-to-month          4        2      50.00%           +20.00 pp
#   One year                5        1      20.00%           -10.00 pp
#   Two year                2        0       0.00%           -30.00 pp
#   "Unknown"               2        1      50.00%           +20.00 pp
#   missing                 2        1      50.00%           +20.00 pp
#   Total                  15        5
CONTRACT_CHURN_ROWS = [
    ("Month-to-month", "Yes"),
    ("Month-to-month", "Yes"),
    ("Month-to-month", "No"),
    ("Month-to-month", "No"),
    ("One year", "Yes"),
    ("One year", "No"),
    ("One year", "No"),
    ("One year", "No"),
    ("One year", "No"),
    # Two year holds customers but no churn, so its rate is a measured 0.00%.
    # This is the contrast to a category with no customers at all, whose rate
    # is not measurable; see the zero-observation test below.
    ("Two year", "No"),
    ("Two year", "No"),
    # Outside the expected categories. The value is used only because it is not
    # one of the documented terms; no business meaning is attributed to it, and
    # in particular it is not a stand-in for a missing value.
    ("Unknown", "Yes"),
    ("Unknown", "No"),
    # Genuinely absent contract values, kept distinct from "Unknown" above.
    (None, "Yes"),
    (None, "No"),
]


@pytest.fixture
def contract_churn_df() -> pd.DataFrame:
    """Build the controlled population described above.

    The frame covers all three expected categories with both churned and
    retained customers, one unexpected value, and one missing value, so a
    single fixture exercises every kind of group the function reports.
    """
    return pd.DataFrame(CONTRACT_CHURN_ROWS, columns=["Contract", "Churn"])


# --- Normal contract analysis ---


def test_expected_contract_groups_match_manual_calculation(contract_churn_df):
    """The three expected categories carry hand-checkable counts and rates.

    Comparing whole frames also pins down the result schema and the order of
    the expected categories, which runs by commitment length rather than by
    frequency or alphabetically.
    """
    result = analyze_contract_churn(contract_churn_df, OVERALL_CHURN_RATE)

    expected_groups = result.loc[result["value_status"] == "expected"].reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "contract": ["Month-to-month", "One year", "Two year"],
            "value_status": ["expected", "expected", "expected"],
            "customer_count": [4, 5, 2],
            "churned_count": [2, 1, 0],
            "churn_rate_percent": [50.0, 20.0, 0.0],
            "difference_percentage_points": [20.0, -10.0, -30.0],
        }
    )

    assert_frame_equal(expected_groups, expected)


# --- Contract values that depart from the expected categories ---


def test_missing_contract_value_is_reported_as_its_own_group(contract_churn_df):
    """Missing contract values form a separate group instead of vanishing.

    The group is identified by its ``value_status`` rather than by a label
    placed in the ``contract`` column, which holds no observed value here. Its
    rate is measured because the group does contain customers.
    """
    result = analyze_contract_churn(contract_churn_df, OVERALL_CHURN_RATE)

    missing_groups = result.loc[result["value_status"] == "missing"]
    assert len(missing_groups) == 1

    missing_group = missing_groups.iloc[0]
    assert pd.isna(missing_group["contract"])
    assert missing_group["customer_count"] == 2
    assert missing_group["churned_count"] == 1
    assert missing_group["churn_rate_percent"] == pytest.approx(50.0)
    assert missing_group["difference_percentage_points"] == pytest.approx(20.0)


def test_unexpected_contract_value_is_preserved_as_its_own_group(contract_churn_df):
    """An undocumented value is reported verbatim under its own status.

    Preserving the observed value is what lets the affected records be traced
    later; the test also confirms the value is neither folded into an expected
    category nor merged with the missing group.
    """
    result = analyze_contract_churn(contract_churn_df, OVERALL_CHURN_RATE)

    unexpected_groups = result.loc[result["value_status"] == "unexpected"]
    assert list(unexpected_groups["contract"]) == ["Unknown"]

    unexpected_group = unexpected_groups.iloc[0]
    assert unexpected_group["customer_count"] == 2
    assert unexpected_group["churned_count"] == 1
    assert unexpected_group["churn_rate_percent"] == pytest.approx(50.0)
    assert unexpected_group["difference_percentage_points"] == pytest.approx(20.0)

    # The expected categories still account for 4 + 5 + 2 customers, so none of
    # the "Unknown" customers was reassigned to one of them.
    expected_groups = result.loc[result["value_status"] == "expected"]
    assert "Unknown" not in set(expected_groups["contract"])
    assert int(expected_groups["customer_count"].sum()) == 11


def test_expected_category_with_no_customers_has_no_measurable_rate():
    """An expected category nobody holds is reported, with an unmeasured rate.

    The category keeps its row so the comparison has a stable shape, but its
    rate is ``NaN`` rather than 0%, because no denominator exists to divide by.
    """
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "Month-to-month", "One year"],
            "Churn": ["Yes", "No", "No"],
        }
    )

    result = analyze_contract_churn(df, OVERALL_CHURN_RATE)

    two_year_groups = result.loc[result["contract"] == "Two year"]
    assert len(two_year_groups) == 1

    two_year_group = two_year_groups.iloc[0]
    assert two_year_group["value_status"] == "expected"
    assert two_year_group["customer_count"] == 0
    assert two_year_group["churned_count"] == 0
    assert pd.isna(two_year_group["churn_rate_percent"])
    assert pd.isna(two_year_group["difference_percentage_points"])


# --- Observational guarantees ---


def test_input_dataframe_is_not_modified(contract_churn_df):
    """The analysis reads the frame it is given and writes nothing back."""
    original_df = contract_churn_df.copy(deep=True)

    analyze_contract_churn(contract_churn_df, OVERALL_CHURN_RATE)

    assert_frame_equal(contract_churn_df, original_df)


def test_group_counts_reconcile_with_input_population(contract_churn_df):
    """The reported groups partition the input records exhaustively.

    The fixture contains expected, unexpected, and missing contract values, so
    the totals can only reconcile if no record was dropped from any of them.
    """
    result = analyze_contract_churn(contract_churn_df, OVERALL_CHURN_RATE)

    assert int(result["customer_count"].sum()) == len(contract_churn_df)
    assert int(result["churned_count"].sum()) == int((contract_churn_df["Churn"] == "Yes").sum())


# --- Input validation ---


@pytest.mark.parametrize("absent_column", ["Contract", "Churn"])
def test_missing_required_column_raises_value_error(contract_churn_df, absent_column):
    """Either required column being absent is refused rather than worked around."""
    df = contract_churn_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        analyze_contract_churn(df, OVERALL_CHURN_RATE)


@pytest.mark.parametrize("invalid_baseline", [-0.1, 100.1])
def test_baseline_outside_percentage_range_raises_value_error(contract_churn_df, invalid_baseline):
    """A baseline that cannot be a percentage is refused.

    Guards against a proportion such as 0.2654 being passed where a percentage
    is expected, which would otherwise skew every reported difference.
    """
    with pytest.raises(ValueError, match="between 0 and 100"):
        analyze_contract_churn(contract_churn_df, invalid_baseline)
