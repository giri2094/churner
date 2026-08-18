"""Targeted churn analysis of the 0-6 month tenure region.

This module measures one selected region of the tenure range against the
project's overall churn baseline: of the customers recorded with 0 to 6 months
of tenure, what share churned, and how does that share compare with the dataset
as a whole?

The region was chosen after the fact. The tenure investigation described the
two churn populations, the histogram and box plot showed churned customers
concentrated toward low tenure, and only then was this slice defined to measure
what those charts suggested. A slice selected because it looked interesting
will tend to look interesting when measured, so the result here is a lead worth
following rather than a validated finding. The 0-6 month boundary is an
exploratory choice made in this module, not a business threshold, not a
property of the data, and not a tenure band: no such feature is created, no
other region is categorised, and the display bins of the histogram carry no
analytical meaning.

The frame passed in is only read from: nothing is imputed, encoded, cleaned,
removed, or overwritten. A customer whose tenure is missing cannot be shown to
satisfy the boundary condition and so does not enter the slice.

Rates here are proportions between 0 and 1 rather than percentages, and the
overall churn rate is computed over the whole dataset so that it matches the
baseline established by the retention vs. churn investigation.

The result describes an association observed in one selected region. It does
not establish that early tenure causes churn, nor that any threshold exists.
"""

import pandas as pd

# The two analyses read the same two columns under the same contract: the same
# required columns, the same numeric requirement on tenure, and the same
# documented target values. Importing that contract keeps the two modules from
# drifting apart, which duplicating the checks here would invite.
from churner.analysis.analyze_tenure_churn import (
    CHURNED_VALUE,
    RETAINED_VALUE,
    TARGET_COLUMN,
    TENURE_COLUMN,
    validate_inputs,
)

# --- The selected region ---
# Inclusive on both sides, so a tenure of exactly 6 months belongs to the slice
# and a tenure of 7 months does not. Tenure is recorded in whole months, so the
# slice holds the values 0, 1, 2, 3, 4, 5, and 6. No upper bound on tenure is
# assumed anywhere: the 72-month maximum is a property of this dataset, not a
# rule.
SLICE_MIN_TENURE = 0
SLICE_MAX_TENURE = 6

# --- Output columns ---
# The boundaries come first, since they define what the row describes and were
# chosen rather than measured.
SLICE_MIN_TENURE_COLUMN = "slice_min_tenure"
SLICE_MAX_TENURE_COLUMN = "slice_max_tenure"
SLICE_CUSTOMER_COUNT_COLUMN = "slice_customer_count"
SLICE_CHURNED_COUNT_COLUMN = "slice_churned_count"
SLICE_RETAINED_COUNT_COLUMN = "slice_retained_count"
SLICE_CHURN_RATE_COLUMN = "slice_churn_rate"
SLICE_POPULATION_SHARE_COLUMN = "slice_population_share"
SLICE_CHURN_CONTRIBUTION_COLUMN = "slice_churn_contribution"
OVERALL_CUSTOMER_COUNT_COLUMN = "overall_customer_count"
OVERALL_CHURNED_COUNT_COLUMN = "overall_churned_count"
OVERALL_CHURN_RATE_COLUMN = "overall_churn_rate"
RATE_DIFFERENCE_COLUMN = "rate_difference"
RELATIVE_CHURN_RATE_COLUMN = "relative_churn_rate"

RESULT_COLUMNS = (
    SLICE_MIN_TENURE_COLUMN,
    SLICE_MAX_TENURE_COLUMN,
    SLICE_CUSTOMER_COUNT_COLUMN,
    SLICE_CHURNED_COUNT_COLUMN,
    SLICE_RETAINED_COUNT_COLUMN,
    SLICE_CHURN_RATE_COLUMN,
    SLICE_POPULATION_SHARE_COLUMN,
    SLICE_CHURN_CONTRIBUTION_COLUMN,
    OVERALL_CUSTOMER_COUNT_COLUMN,
    OVERALL_CHURNED_COUNT_COLUMN,
    OVERALL_CHURN_RATE_COLUMN,
    RATE_DIFFERENCE_COLUMN,
    RELATIVE_CHURN_RATE_COLUMN,
)


def ratio_or_nan(numerator: float, denominator: float) -> float:
    """Divide, reporting a ratio with no denominator as ``NaN``.

    A share of nothing is undefined, which is a different statement from a
    measured share of zero. Returning ``NaN`` keeps the two apart instead of
    letting an empty population read as a population with no churn.
    """
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def analyze_early_tenure_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Measure churn in the 0-6 month tenure region against the whole dataset.

    The slice holds every customer whose recorded tenure falls between
    ``SLICE_MIN_TENURE`` and ``SLICE_MAX_TENURE`` inclusive. A missing tenure
    cannot be shown to satisfy that condition and is left out, as is any tenure
    outside the boundaries, including a negative one; nothing is filled in or
    reassigned to make a customer fit.

    Three different quantities describe the slice and are easy to confuse. The
    churn rate is the share of the slice's own customers that churned. The
    population share is how much of the dataset the slice holds. The churn
    contribution is how much of all observed churn the slice accounts for. A
    small region can carry a high churn rate while contributing little churn
    overall, so the three are reported side by side.

    The comparison against the baseline is given in two forms: the rate
    difference subtracts the overall rate from the slice rate, and the relative
    churn rate divides one by the other, where 1.0 means the slice matches the
    baseline.

    Both quantities describe an association in a region that was selected after
    the distribution had been inspected. They do not establish that low tenure
    causes churn, and they do not identify a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``tenure`` and ``Churn`` columns.
        Only read from; never modified.

    Returns
    -------
    pd.DataFrame
        A single row holding the slice boundaries, the slice and overall
        counts, and the rates derived from those counts. Rates are proportions
        between 0 and 1, and are ``NaN`` where the denominator is zero.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds a value outside its
        documented values, if ``tenure`` is not numeric, or if the slice holds
        a customer whose churn status is unknown, which would leave the slice
        counts unable to reconcile.
    """
    validate_inputs(df)

    tenure_values = df[TENURE_COLUMN]
    churn_values = df[TARGET_COLUMN]

    # Comparisons against a missing tenure are false, so such customers fall
    # outside the slice without being singled out.
    in_slice = tenure_values.between(SLICE_MIN_TENURE, SLICE_MAX_TENURE, inclusive="both")

    slice_customer_count = int(in_slice.sum())
    slice_churned_count = int((in_slice & (churn_values == CHURNED_VALUE)).sum())
    slice_retained_count = int((in_slice & (churn_values == RETAINED_VALUE)).sum())

    # The slice is the population whose churn rate is being measured, so every
    # customer in it has to be one or the other. A customer with no recorded
    # churn status would sit in the denominator of that rate while belonging to
    # neither count, quietly depressing it, so the shortfall is reported
    # instead of being absorbed.
    unclassified_count = slice_customer_count - slice_churned_count - slice_retained_count
    if unclassified_count != 0:
        raise ValueError(
            f"The {SLICE_MIN_TENURE}-{SLICE_MAX_TENURE} month slice holds "
            f"{slice_customer_count} customers, of which {slice_churned_count} churned and "
            f"{slice_retained_count} were retained, leaving {unclassified_count} with no "
            f"recorded '{TARGET_COLUMN}' value. Those records need investigating before a "
            "churn rate for this region can be reported."
        )

    overall_customer_count = int(len(df))
    overall_churned_count = int((churn_values == CHURNED_VALUE).sum())

    slice_churn_rate = ratio_or_nan(slice_churned_count, slice_customer_count)
    overall_churn_rate = ratio_or_nan(overall_churned_count, overall_customer_count)

    return pd.DataFrame(
        [
            {
                SLICE_MIN_TENURE_COLUMN: SLICE_MIN_TENURE,
                SLICE_MAX_TENURE_COLUMN: SLICE_MAX_TENURE,
                SLICE_CUSTOMER_COUNT_COLUMN: slice_customer_count,
                SLICE_CHURNED_COUNT_COLUMN: slice_churned_count,
                SLICE_RETAINED_COUNT_COLUMN: slice_retained_count,
                SLICE_CHURN_RATE_COLUMN: slice_churn_rate,
                SLICE_POPULATION_SHARE_COLUMN: ratio_or_nan(
                    slice_customer_count, overall_customer_count
                ),
                SLICE_CHURN_CONTRIBUTION_COLUMN: ratio_or_nan(
                    slice_churned_count, overall_churned_count
                ),
                OVERALL_CUSTOMER_COUNT_COLUMN: overall_customer_count,
                OVERALL_CHURNED_COUNT_COLUMN: overall_churned_count,
                OVERALL_CHURN_RATE_COLUMN: overall_churn_rate,
                RATE_DIFFERENCE_COLUMN: slice_churn_rate - overall_churn_rate,
                RELATIVE_CHURN_RATE_COLUMN: ratio_or_nan(slice_churn_rate, overall_churn_rate),
            }
        ],
        columns=list(RESULT_COLUMNS),
    )
