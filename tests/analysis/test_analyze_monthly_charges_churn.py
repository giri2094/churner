"""Unit tests for the MonthlyCharges vs. churn analysis.

These tests establish the public contract of both functions in the module: the
statistics the distribution comparison reports for each churn population, the
quantile groups the quartile analysis divides the observed charges into, the
counts and rate each group reports, and the input both functions refuse.

The tests that matter most are the ones about where the group boundaries come
from and which customers end up inside them. The boundaries are quantiles of
the charges actually recorded, so a fixture whose quartiles are known by hand
would still pass against an implementation that had invented dollar ranges of
its own, unless the boundaries themselves are asserted. The grouping tests
cover the cases where a naive division loses or duplicates customers: charges
repeated across a boundary, charges sitting exactly on one, and a population
whose charges never vary at all.

Every expectation below is calculated by hand from the small fixtures defined
in this module, so a failure points at the implementation rather than at a
figure copied from the real dataset.
"""

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_integer_dtype
from pandas.testing import assert_frame_equal

from churner.analysis.analyze_monthly_charges_churn import (
    analyze_monthly_charges_churn,
    describe_monthly_charges_by_churn,
)

RESULT_COLUMNS = [
    "charge_group",
    "lower_charge",
    "upper_charge",
    "customer_count",
    "churned_count",
    "retained_count",
    "churn_rate",
]

DISTRIBUTION_COLUMNS = [
    "churn_status",
    "customer_count",
    "mean_monthly_charges",
    "median_monthly_charges",
    "std_monthly_charges",
    "min_monthly_charges",
    "q1_monthly_charges",
    "q3_monthly_charges",
    "max_monthly_charges",
]

PUBLIC_ANALYSES = [analyze_monthly_charges_churn, describe_monthly_charges_by_churn]

# --- Controlled fixture population ---
# Twelve customers charged 10 through 120 in steps of 10. The quartiles of that
# distribution, interpolated as pandas does, are 37.5, 65.0, and 92.5, which
# with the observed minimum and maximum give the boundaries used below:
#
#   Q1: 10.0 to 37.5   charges 10, 20, 30    0 churned -> rate 0
#   Q2: 37.5 to 65.0   charges 40, 50, 60    1 churned -> rate 1/3
#   Q3: 65.0 to 92.5   charges 70, 80, 90    2 churned -> rate 2/3
#   Q4: 92.5 to 120.0  charges 100, 110, 120 3 churned -> rate 1
#
# The four groups hold the same number of customers but differ in churned
# count, so no row can match another's figures by coincidence, and two of the
# rates are thirds that no rounding to two decimals would reproduce exactly.
CHARGE_ROWS = [
    (10.0, "No"),
    (20.0, "No"),
    (30.0, "No"),
    (40.0, "Yes"),
    (50.0, "No"),
    (60.0, "No"),
    (70.0, "Yes"),
    (80.0, "Yes"),
    (90.0, "No"),
    (100.0, "Yes"),
    (110.0, "Yes"),
    (120.0, "Yes"),
]

# --- Controlled fixture for the distribution comparison ---
# Two populations small enough to describe by hand:
#
#   retained: 10, 20         mean 15, median 15, std sqrt(50), Q1 12.5, Q3 17.5
#   churned:  30, 40, 50     mean 40, median 40, std 10,       Q1 35,   Q3 45
#
# They differ in size as well as in centre and spread, so a statistic taken
# from the wrong population cannot pass unnoticed.
DISTRIBUTION_ROWS = [
    (10.0, "No"),
    (20.0, "No"),
    (30.0, "Yes"),
    (40.0, "Yes"),
    (50.0, "Yes"),
]


@pytest.fixture
def charges_df() -> pd.DataFrame:
    """Build the controlled twelve-customer population described above."""
    return pd.DataFrame(CHARGE_ROWS, columns=["MonthlyCharges", "Churn"])


@pytest.fixture
def distribution_df() -> pd.DataFrame:
    """Build the controlled two-population fixture described above."""
    return pd.DataFrame(DISTRIBUTION_ROWS, columns=["MonthlyCharges", "Churn"])


# --- Quantile group aggregation ---


def test_result_matches_manual_calculation(charges_df):
    """Every reported figure is the hand-calculated one.

    Comparing the whole frame also pins down the row count, the result schema,
    the column order, and the ascending group order in a single assertion.
    """
    result = analyze_monthly_charges_churn(charges_df)

    expected = pd.DataFrame(
        {
            "charge_group": ["Q1", "Q2", "Q3", "Q4"],
            "lower_charge": [10.0, 37.5, 65.0, 92.5],
            "upper_charge": [37.5, 65.0, 92.5, 120.0],
            "customer_count": [3, 3, 3, 3],
            "churned_count": [0, 1, 2, 3],
            "retained_count": [3, 2, 1, 0],
            "churn_rate": [0.0, 1 / 3, 2 / 3, 1.0],
        }
    )

    assert_frame_equal(result, expected)


def test_boundaries_are_read_from_the_observed_distribution(charges_df):
    """The groups span the observed charges, divided at their own quartiles.

    An implementation that had defined dollar ranges of its own would still
    produce four plausible-looking rows, so the boundaries are asserted against
    the quantiles of the fixture rather than only the counts inside them.
    """
    result = analyze_monthly_charges_churn(charges_df)

    charges = charges_df["MonthlyCharges"]
    assert result["lower_charge"].tolist() == [
        charges.min(),
        charges.quantile(0.25),
        charges.quantile(0.50),
        charges.quantile(0.75),
    ]
    assert result["upper_charge"].tolist() == [
        charges.quantile(0.25),
        charges.quantile(0.50),
        charges.quantile(0.75),
        charges.max(),
    ]


def test_groups_run_from_the_lowest_quarter_to_the_highest(charges_df):
    """Rows read Q1 upward, each group starting where the previous one ended."""
    result = analyze_monthly_charges_churn(charges_df)

    assert result["charge_group"].tolist() == ["Q1", "Q2", "Q3", "Q4"]
    assert result["lower_charge"].is_monotonic_increasing
    assert result["upper_charge"].tolist()[:-1] == result["lower_charge"].tolist()[1:]
    assert result.index.tolist() == list(range(len(result)))


def test_row_order_of_the_input_does_not_change_the_result(charges_df):
    """The same customers give the same groups however the frame is ordered."""
    shuffled_df = charges_df.sample(frac=1, random_state=0)

    assert_frame_equal(
        analyze_monthly_charges_churn(shuffled_df),
        analyze_monthly_charges_churn(charges_df),
    )


def test_churn_rate_is_a_proportion_not_a_percentage(charges_df):
    """Rates run from 0 to 1, so a fully churned group reads 1.0 and not 100.

    Both extremes are checked: the group where everyone stayed is a measured
    0.0 rather than a missing value.
    """
    result = analyze_monthly_charges_churn(charges_df)

    assert result.iloc[0]["churn_rate"] == pytest.approx(0.0)
    assert result.iloc[-1]["churn_rate"] == pytest.approx(1.0)
    assert result["churn_rate"].between(0, 1).all()


def test_churn_rate_follows_from_the_counts(charges_df):
    """The rate is each group's churned share of its own customers.

    Deriving it from the reported counts rather than from a literal keeps the
    two from disagreeing without the test noticing.
    """
    result = analyze_monthly_charges_churn(charges_df)

    expected_rate = result["churned_count"] / result["customer_count"]

    assert result["churn_rate"].tolist() == pytest.approx(expected_rate.tolist())


def test_every_group_reconciles(charges_df):
    """A group's customers are exactly its churned plus its retained."""
    result = analyze_monthly_charges_churn(charges_df)

    assert result["customer_count"].tolist() == (
        result["churned_count"] + result["retained_count"]
    ).tolist()


def test_groups_account_for_every_record(charges_df):
    """No customer is lost between the frame and the groups built from it."""
    result = analyze_monthly_charges_churn(charges_df)

    assert result["customer_count"].sum() == len(charges_df)
    assert result["churned_count"].sum() == (charges_df["Churn"] == "Yes").sum()
    assert result["retained_count"].sum() == (charges_df["Churn"] == "No").sum()


def test_result_carries_no_columns_beyond_the_contract(charges_df):
    """The frame holds these seven columns and nothing else.

    Comparisons against a baseline, population shares, and risk labels belong
    to other analyses or to none; this one describes the observed groups.
    """
    result = analyze_monthly_charges_churn(charges_df)

    assert list(result.columns) == RESULT_COLUMNS


def test_result_columns_carry_usable_types(charges_df):
    """Counts are integers, boundaries and rates are floats, labels are text.

    A count inferred as a float or a rate inferred as ``object`` would break
    arithmetic downstream, and a label that is not a string would not be
    readable as one.
    """
    result = analyze_monthly_charges_churn(charges_df)

    assert all(isinstance(group_label, str) for group_label in result["charge_group"])
    for count_column in ["customer_count", "churned_count", "retained_count"]:
        assert is_integer_dtype(result[count_column])
    for float_column in ["lower_charge", "upper_charge", "churn_rate"]:
        assert is_float_dtype(result[float_column])


# --- Repeated and boundary charges ---


def test_charge_on_a_boundary_joins_the_group_below_it():
    """Groups are closed on the right, and the lowest holds its own boundary.

    The five charges below put an observed value on every quartile, which is
    where an off-by-one in the grouping would either lose a customer or count
    one twice. The 10 sits on the lower boundary of Q1 and belongs to it; the
    20, 30, and 40 each sit on a boundary and belong to the group beneath.
    """
    df = pd.DataFrame(
        {
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Churn": ["No", "Yes", "No", "Yes", "Yes"],
        }
    )

    result = analyze_monthly_charges_churn(df)

    assert result["lower_charge"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert result["upper_charge"].tolist() == [20.0, 30.0, 40.0, 50.0]
    assert result["customer_count"].tolist() == [2, 1, 1, 1]
    assert result["customer_count"].sum() == len(df)


def test_customers_charged_the_same_amount_share_one_group():
    """A repeated charge is never split across two groups.

    Five of these seven customers are charged the same amount, which the
    quartiles place on a boundary. They belong to one group, which makes the
    groups unequal in size: the alternative, splitting identical charges to
    even the groups out, would report two groups whose boundaries overlap.
    """
    df = pd.DataFrame(
        {
            "MonthlyCharges": [50.0] * 5 + [10.0, 90.0],
            "Churn": ["Yes", "No", "No", "No", "Yes", "Yes", "No"],
        }
    )

    result = analyze_monthly_charges_churn(df)

    assert result["charge_group"].tolist() == ["Q1", "Q2"]
    assert result["lower_charge"].tolist() == [10.0, 50.0]
    assert result["upper_charge"].tolist() == [50.0, 90.0]
    assert result["customer_count"].tolist() == [6, 1]
    assert result["churned_count"].tolist() == [3, 0]
    assert result["retained_count"].tolist() == [3, 1]
    assert result["churn_rate"].tolist() == pytest.approx([0.5, 0.0])
    assert result["customer_count"].sum() == len(df)


def test_constant_charges_form_a_single_group():
    """A distribution that never varies is divided by nothing.

    Its quartiles all sit at the one charge observed, so there is no interior
    boundary to split anyone at. One group spanning that charge describes the
    population; four groups, three of them empty, would describe nothing and
    would carry rates over no customers.
    """
    df = pd.DataFrame({"MonthlyCharges": [50.0] * 4, "Churn": ["Yes", "No", "No", "Yes"]})

    result = analyze_monthly_charges_churn(df)

    assert result["charge_group"].tolist() == ["Q1"]
    assert result["lower_charge"].tolist() == [50.0]
    assert result["upper_charge"].tolist() == [50.0]
    assert result["customer_count"].tolist() == [4]
    assert result["churned_count"].tolist() == [2]
    assert result["retained_count"].tolist() == [2]
    assert result["churn_rate"].tolist() == pytest.approx([0.5])


def test_no_group_is_reported_without_customers():
    """Only groups customers were placed in are reported.

    Two customers cannot fill four quarters: the middle boundaries fall between
    the only two charges observed, so the groups they would define hold nobody
    and are left out rather than reported with a rate over an empty population.
    """
    df = pd.DataFrame({"MonthlyCharges": [1.0, 2.0], "Churn": ["Yes", "No"]})

    result = analyze_monthly_charges_churn(df)

    assert (result["customer_count"] > 0).all()
    assert result["customer_count"].sum() == len(df)
    assert result["charge_group"].tolist() == ["Q1", "Q2"]


# --- The distribution comparison ---


def test_distribution_matches_manual_calculation(distribution_df):
    """Every statistic is the hand-calculated one, for both populations.

    Comparing the whole frame also pins down the result schema, the column
    order, and the reporting order of the two populations.
    """
    result = describe_monthly_charges_by_churn(distribution_df)

    expected = pd.DataFrame(
        {
            "churn_status": ["No", "Yes"],
            "customer_count": [2, 3],
            "mean_monthly_charges": [15.0, 40.0],
            "median_monthly_charges": [15.0, 40.0],
            "std_monthly_charges": [50**0.5, 10.0],
            "min_monthly_charges": [10.0, 30.0],
            "q1_monthly_charges": [12.5, 35.0],
            "q3_monthly_charges": [17.5, 45.0],
            "max_monthly_charges": [20.0, 50.0],
        }
    )

    assert_frame_equal(result, expected)


def test_distribution_counts_account_for_every_record(distribution_df):
    """The two populations together hold every customer in the frame."""
    result = describe_monthly_charges_by_churn(distribution_df)

    assert result["customer_count"].sum() == len(distribution_df)


def test_distribution_reports_a_population_that_holds_no_customer():
    """A status nobody holds is a row of zero customers, not an absent row.

    Its statistics are undefined rather than zero, which is a different claim
    from a population measured at a charge of nothing.
    """
    df = pd.DataFrame({"MonthlyCharges": [10.0, 20.0], "Churn": ["Yes", "Yes"]})

    result = describe_monthly_charges_by_churn(df).set_index("churn_status")

    assert result.loc["No", "customer_count"] == 0
    assert result.loc["No", ["mean_monthly_charges", "min_monthly_charges"]].isna().all()
    assert result.loc["Yes", "customer_count"] == 2


def test_distribution_of_a_single_customer_has_no_spread():
    """One observation has a centre and a range but no standard deviation."""
    df = pd.DataFrame({"MonthlyCharges": [42.0, 10.0], "Churn": ["Yes", "No"]})

    result = describe_monthly_charges_by_churn(df).set_index("churn_status")

    assert result.loc["Yes", "mean_monthly_charges"] == pytest.approx(42.0)
    assert result.loc["Yes", "min_monthly_charges"] == pytest.approx(42.0)
    assert result.loc["Yes", "max_monthly_charges"] == pytest.approx(42.0)
    assert pd.isna(result.loc["Yes", "std_monthly_charges"])


def test_distribution_carries_no_columns_beyond_the_contract(distribution_df):
    """The comparison holds these nine columns and nothing else."""
    result = describe_monthly_charges_by_churn(distribution_df)

    assert list(result.columns) == DISTRIBUTION_COLUMNS


# --- Populations with nothing to group ---


def test_empty_dataframe_returns_the_empty_schema():
    """A frame with the required columns but no rows is measured, not refused."""
    df = pd.DataFrame(
        {"MonthlyCharges": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")}
    )

    result = analyze_monthly_charges_churn(df)

    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == 0


def test_empty_result_keeps_the_populated_column_types(charges_df):
    """The empty frame's counts stay integers and its rate stays a float.

    An empty result inferred as ``object`` would break arithmetic downstream
    for callers that treat both cases alike.
    """
    empty_result = analyze_monthly_charges_churn(
        pd.DataFrame(
            {"MonthlyCharges": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")}
        )
    )
    populated_result = analyze_monthly_charges_churn(charges_df)

    assert_frame_equal(
        empty_result.dtypes.to_frame(),
        populated_result.dtypes.to_frame(),
    )


def test_empty_dataframe_is_described_as_two_empty_populations():
    """The comparison keeps its two rows, with nothing measured in either."""
    df = pd.DataFrame(
        {"MonthlyCharges": pd.Series(dtype="float64"), "Churn": pd.Series(dtype="object")}
    )

    result = describe_monthly_charges_by_churn(df)

    assert result["churn_status"].tolist() == ["No", "Yes"]
    assert result["customer_count"].tolist() == [0, 0]
    assert result[["mean_monthly_charges", "std_monthly_charges"]].isna().all().all()


# --- Input validation ---


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
@pytest.mark.parametrize("absent_column", ["MonthlyCharges", "Churn"])
def test_missing_required_column_raises_value_error(charges_df, analysis, absent_column):
    """Either required column being absent is refused rather than worked around."""
    df = charges_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        analysis(df)


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
@pytest.mark.parametrize("non_numeric_charges", [["18.25", "70.35"], ["cheap", "expensive"]])
def test_non_numeric_charges_raise_value_error(analysis, non_numeric_charges):
    """Text charges are refused, as text tenure is throughout the tenure analyses.

    Deciding what an unparseable entry means is a data-cleaning decision, not
    one this module makes.
    """
    df = pd.DataFrame({"MonthlyCharges": non_numeric_charges, "Churn": ["Yes", "No"]})

    with pytest.raises(ValueError, match="MonthlyCharges"):
        analysis(df)


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
@pytest.mark.parametrize("unexpected_value", ["Unknown", "yes", 1])
def test_unexpected_churn_value_raises_value_error(analysis, unexpected_value):
    """A target value outside the documented domain is refused, as elsewhere.

    Counting it as retained would inflate a group's retained count and
    understate its churn rate, which is the quantity under investigation.
    """
    df = pd.DataFrame(
        {"MonthlyCharges": [20.0, 50.0, 80.0], "Churn": ["Yes", "No", unexpected_value]}
    )

    with pytest.raises(ValueError, match="Churn"):
        analysis(df)


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
def test_missing_charges_raise_value_error(analysis):
    """A customer with no recorded charge cannot be placed in a group.

    Leaving them out would report groups that no longer account for the frame
    they were built from, and would shift the quartile boundaries themselves,
    so the shortfall is reported instead of absorbed.
    """
    df = pd.DataFrame({"MonthlyCharges": [20.0, None, 80.0], "Churn": ["Yes", "No", "No"]})

    with pytest.raises(ValueError, match="MonthlyCharges"):
        analysis(df)


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
def test_missing_churn_raises_value_error(analysis):
    """An unclassifiable customer stops the measurement.

    They would otherwise sit in the denominator of a group's churn rate while
    counting as neither churned nor retained, leaving the row unable to
    reconcile and the rate quietly understated.
    """
    df = pd.DataFrame({"MonthlyCharges": [20.0, 50.0, 80.0], "Churn": ["Yes", None, "No"]})

    with pytest.raises(ValueError, match="Churn"):
        analysis(df)


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
def test_nothing_is_repaired_before_the_refusal(analysis):
    """A frame that is refused is left exactly as it was passed in."""
    df = pd.DataFrame({"MonthlyCharges": [20.0, None, 80.0], "Churn": ["Yes", "No", None]})
    original_df = df.copy(deep=True)

    with pytest.raises(ValueError):
        analysis(df)

    assert_frame_equal(df, original_df)


# --- Observational guarantees ---


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
def test_input_dataframe_is_not_modified(charges_df, analysis):
    """The analysis reads the frame it is given and writes nothing back.

    The group each customer falls in is an intermediate of the calculation, and
    a column holding it would outlive the call if it were written here.
    """
    original_df = charges_df.copy(deep=True)

    analysis(charges_df)

    assert_frame_equal(charges_df, original_df)
    assert list(charges_df.columns) == ["MonthlyCharges", "Churn"]


@pytest.mark.parametrize("analysis", PUBLIC_ANALYSES)
def test_result_is_a_new_frame(charges_df, analysis):
    """The result is a frame of its own, so editing it cannot touch the input."""
    result = analysis(charges_df)

    assert isinstance(result, pd.DataFrame)
    assert result is not charges_df
