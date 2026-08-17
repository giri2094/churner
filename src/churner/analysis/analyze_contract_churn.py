"""Contract-level churn analysis of the Telco customer churn dataset.

This module investigates a single question: how does the observed churn rate
differ across customer contract types? Each contract group's churn rate is
compared against the overall churn baseline established by the retention vs.
churn investigation, which the caller supplies rather than this module
recomputing it.

The analysis is observational. The frame passed in is only read from: nothing
is imputed, encoded, scaled, removed, cleaned, or overwritten. Contract values
outside the expected categories are reported as rows of their own and kept
distinct from missing values, so neither is silently dropped, replaced, or
counted as an expected category. Charts and reports are the caller's concern.

The result describes observed differences between contract groups. It does not
establish why those differences exist.
"""

import pandas as pd

# --- Input columns ---
# The data dictionary documents ``Contract`` as the customer's contract term
# and ``Churn`` as the label column, where "Yes" means the customer left
# during the last period.
CONTRACT_COLUMN = "Contract"
TARGET_COLUMN = "Churn"
CHURNED_VALUE = "Yes"

# Contract categories documented for the dataset, ordered by commitment
# length. This is an expectation to compare the data against, not a filter:
# any other value is reported rather than dropped or reassigned.
EXPECTED_CONTRACT_VALUES = ("Month-to-month", "One year", "Two year")

# --- Value status ---
# Separates the expected categories from values that depart from them. A value
# that is absent and a value that is present but undocumented are different
# observations, so they are never merged into one status: each may call for
# different follow-up once the affected records have been investigated.
EXPECTED_STATUS = "expected"
UNEXPECTED_STATUS = "unexpected"
MISSING_STATUS = "missing"

# --- Output columns ---
# Churn rates are expressed in percent, as in the baseline investigation, so
# the gap between two of them is a number of percentage points.
CONTRACT_RESULT_COLUMN = "contract"
VALUE_STATUS_COLUMN = "value_status"
CUSTOMER_COUNT_COLUMN = "customer_count"
CHURNED_COUNT_COLUMN = "churned_count"
CHURN_RATE_COLUMN = "churn_rate_percent"
DIFFERENCE_COLUMN = "difference_percentage_points"


def build_contract_groups(contract_values: pd.Series) -> list[tuple[str | None, str, pd.Series]]:
    """Pair each contract group with its value status and its row mask.

    The expected categories always come first, in commitment-length order, so
    the contract comparison keeps a stable shape and a category that no
    customer holds shows up as a zero count rather than as an absent row. Any
    undocumented value follows, reported under the value it actually holds so
    the affected records can be traced, and missing values come last as a
    group of their own. The groups are therefore exhaustive: the masks
    partition the frame, so the counts taken from them add up to its length.

    Parameters
    ----------
    contract_values : pd.Series
        The ``Contract`` column, read as-is.

    Returns
    -------
    list[tuple[str | None, str, pd.Series]]
        One ``(contract_value, value_status, mask)`` triple per group, in
        reporting order. ``contract_value`` is ``None`` for the missing group,
        which has no observed value to report.
    """
    missing = contract_values.isna()
    expected = contract_values.isin(EXPECTED_CONTRACT_VALUES)

    groups = [
        (expected_value, EXPECTED_STATUS, contract_values == expected_value)
        for expected_value in EXPECTED_CONTRACT_VALUES
    ]

    # Ordered from most to least frequent, so the widest departure from the
    # expected categories is reported first.
    unexpected_counts = contract_values[~expected & ~missing].value_counts()
    groups.extend(
        (str(observed_value), UNEXPECTED_STATUS, contract_values == observed_value)
        for observed_value in unexpected_counts.index
    )

    if missing.any():
        groups.append((None, MISSING_STATUS, missing))

    return groups


def analyze_contract_churn(df: pd.DataFrame, overall_churn_rate: float) -> pd.DataFrame:
    """Compare the observed churn rate of each contract type to the baseline.

    Counts are taken directly from the frame and the rates are derived from
    those counts, so the two cannot disagree. A group's churn rate is the share
    of its customers whose ``Churn`` value is ``"Yes"``, in percent; it is
    ``NaN`` for a group holding no customers, since no rate is measurable
    there.

    The ``value_status`` column carries the difference between the expected
    contract categories and any departure from them, so a caller can restrict
    the comparison to ``"expected"`` rows while the unexpected and missing
    counts stay visible in the same result. This function reports those
    departures; it makes no decision about how they should be handled, which
    requires investigating the affected records.

    The differences reported here are associations observed in the data. They
    do not establish that the contract type causes churn. Every ``Churn`` value
    other than ``"Yes"`` is counted as retained; this function does not check
    that column against its own expected values.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``Contract`` and ``Churn``
        columns. Only read from; never modified.
    overall_churn_rate : float
        The previously established overall churn baseline, in percent. Used
        solely as the reference the per-group rates are compared against; the
        baseline is not recomputed here.

    Returns
    -------
    pd.DataFrame
        One row per contract group, with the columns ``contract``,
        ``value_status``, ``customer_count``, ``churned_count``,
        ``churn_rate_percent``, and ``difference_percentage_points``. The rows
        account for every record in ``df``.

    Raises
    ------
    ValueError
        If a required column is absent, or if ``overall_churn_rate`` does not
        lie between 0 and 100 and so cannot be a percentage.
    """
    missing_columns = [
        column_name
        for column_name in (CONTRACT_COLUMN, TARGET_COLUMN)
        if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}. "
            f"Columns present: {list(df.columns)}"
        )

    if not 0 <= overall_churn_rate <= 100:
        raise ValueError(
            "overall_churn_rate is expected as a percentage between 0 and 100; "
            f"got {overall_churn_rate}."
        )

    contract_values = df[CONTRACT_COLUMN]
    churned = df[TARGET_COLUMN] == CHURNED_VALUE

    group_counts = pd.DataFrame(
        [
            {
                CONTRACT_RESULT_COLUMN: contract_value,
                VALUE_STATUS_COLUMN: value_status,
                CUSTOMER_COUNT_COLUMN: int(in_group.sum()),
                CHURNED_COUNT_COLUMN: int((in_group & churned).sum()),
            }
            for contract_value, value_status, in_group in build_contract_groups(contract_values)
        ]
    )

    churn_rate = group_counts[CHURNED_COUNT_COLUMN] / group_counts[CUSTOMER_COUNT_COLUMN] * 100

    return group_counts.assign(
        **{
            CHURN_RATE_COLUMN: churn_rate,
            DIFFERENCE_COLUMN: churn_rate - overall_churn_rate,
        }
    )
