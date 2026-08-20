"""Per-month churn analysis of the Telco customer churn dataset.

This module investigates a single question: how does the observed churn rate
vary across individual tenure months? It answers with one row per tenure month
actually recorded in the data, holding that month's customer count, its churned
and retained counts, and the churn rate derived from them.

The resolution is the point. The tenure investigation compared two populations
by their tenure distributions, and the early-tenure analysis measured one chosen
region against the baseline; both summarise across many months at once. Here
each observed month is reported on its own, so the shape of the variation stays
visible instead of being averaged away.

Only months that appear in the data are reported. A month nobody was recorded at
has no customers to count and no rate to measure, so no row is fabricated for
it, and no months are grouped together: this module creates no tenure bands, no
risk categories, and no threshold.

The frame passed in is only read from: nothing is imputed, encoded, cleaned,
removed, or overwritten. A customer whose tenure is missing belongs to no
numerical month and so enters no row.

Rates here are proportions between 0 and 1 rather than percentages.

The result describes churn as observed at each tenure month. It does not
establish that tenure causes churn, and it tests nothing for significance.
"""

import pandas as pd

# The tenure analyses read the same two columns under the same contract: the
# same required columns, the same numeric requirement on tenure, and the same
# documented target values. Importing that contract keeps the modules from
# drifting apart, which duplicating the checks here would invite.
from churner.analysis.analyze_tenure_churn import (
    CHURNED_VALUE,
    RETAINED_VALUE,
    TARGET_COLUMN,
    TENURE_COLUMN,
    validate_inputs,
)

# --- Output columns ---
# ``tenure_month`` is named apart from the input ``tenure`` column because it
# reports an observed value that a whole row describes, rather than one
# customer's recorded months.
TENURE_MONTH_COLUMN = "tenure_month"
CUSTOMER_COUNT_COLUMN = "customer_count"
CHURNED_COUNT_COLUMN = "churned_count"
RETAINED_COUNT_COLUMN = "retained_count"
CHURN_RATE_COLUMN = "churn_rate"

RESULT_COLUMNS = (
    TENURE_MONTH_COLUMN,
    CUSTOMER_COUNT_COLUMN,
    CHURNED_COUNT_COLUMN,
    RETAINED_COUNT_COLUMN,
    CHURN_RATE_COLUMN,
)


def empty_result() -> pd.DataFrame:
    """Build the result for a population holding no tenure observation.

    The schema is declared rather than inferred, so a frame with nothing to
    aggregate still carries the same columns and the same dtypes as a populated
    one and can be consumed without a special case. ``tenure_month`` is typed
    as a float because no observation is present to take a type from, and a
    float admits every tenure value this analysis might later report.

    Returns
    -------
    pd.DataFrame
        An empty frame with the five result columns.
    """
    return pd.DataFrame(
        {
            TENURE_MONTH_COLUMN: pd.Series(dtype="float64"),
            CUSTOMER_COUNT_COLUMN: pd.Series(dtype="int64"),
            CHURNED_COUNT_COLUMN: pd.Series(dtype="int64"),
            RETAINED_COUNT_COLUMN: pd.Series(dtype="int64"),
            CHURN_RATE_COLUMN: pd.Series(dtype="float64"),
        }
    )


def analyze_tenure_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Measure the observed churn rate at each recorded tenure month.

    Every tenure value present in the data gets exactly one row, ordered by
    tenure month ascending. Absent months are not invented: a month no customer
    was recorded at would otherwise appear as a population of zero with an
    undefined rate, which describes the dataset's coverage rather than churn. A
    recorded tenure of zero is an observation like any other and gets its own
    row, as does the highest month observed, and unusual values such as a
    negative tenure are reported rather than removed, since dropping
    observations for looking wrong would hide what an exploratory summary
    exists to reveal.

    A missing tenure cannot be placed at a numerical month, so those customers
    enter no row. Their churn status is therefore not consulted either.

    Each row reconciles, ``customer_count`` equalling ``churned_count`` plus
    ``retained_count``, and the churn rate is that month's share of churned
    customers as a proportion between 0 and 1. Every row describes at least one
    observed customer, so no rate is ever divided by an empty population.

    The variation across months is an association observed in the data. It does
    not establish that tenure causes churn, it identifies no threshold, and it
    is not tested for significance: a month holding few customers can show an
    extreme rate on very little evidence, which the customer counts reported
    alongside the rates are there to expose.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``tenure`` and ``Churn`` columns.
        Only read from; never modified.

    Returns
    -------
    pd.DataFrame
        One row per observed tenure month, ascending, with the columns
        ``tenure_month``, ``customer_count``, ``churned_count``,
        ``retained_count``, and ``churn_rate``. Empty, with the same columns,
        when no tenure observation is present.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds a value outside its
        documented values, if ``tenure`` is not numeric, or if a customer with
        a recorded tenure has no recorded churn status, which would leave that
        month's counts unable to reconcile.
    """
    validate_inputs(df)

    # Selecting with a mask returns a new frame, so nothing below is written
    # back to the caller's.
    observed = df.loc[df[TENURE_COLUMN].notna(), [TENURE_COLUMN, TARGET_COLUMN]]

    # Each month's own customers are the population whose churn rate is being
    # measured, so every one of them has to be churned or retained. A customer
    # with no recorded churn status would sit in the denominator of that rate
    # while belonging to neither count, quietly depressing it, so the shortfall
    # is reported instead of being absorbed.
    unclassified_count = int(observed[TARGET_COLUMN].isna().sum())
    if unclassified_count != 0:
        raise ValueError(
            f"{unclassified_count} customer(s) with a recorded '{TENURE_COLUMN}' have no "
            f"recorded '{TARGET_COLUMN}' value. Those records need investigating before a "
            "churn rate for their tenure month can be reported."
        )

    if observed.empty:
        return empty_result()

    # Grouping by the renamed series labels the result column directly, and
    # sorting the groups puts the months in ascending order.
    tenure_months = observed[TENURE_COLUMN].rename(TENURE_MONTH_COLUMN)
    churned = observed[TARGET_COLUMN] == CHURNED_VALUE
    retained = observed[TARGET_COLUMN] == RETAINED_VALUE

    monthly_counts = pd.DataFrame(
        {
            CUSTOMER_COUNT_COLUMN: churned.groupby(tenure_months, sort=True).size(),
            CHURNED_COUNT_COLUMN: churned.groupby(tenure_months, sort=True).sum(),
            RETAINED_COUNT_COLUMN: retained.groupby(tenure_months, sort=True).sum(),
        }
    ).reset_index()

    return monthly_counts.assign(
        **{
            CHURN_RATE_COLUMN: monthly_counts[CHURNED_COUNT_COLUMN]
            / monthly_counts[CUSTOMER_COUNT_COLUMN]
        }
    )[list(RESULT_COLUMNS)]
