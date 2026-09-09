"""MonthlyCharges vs. churn analysis of the Telco customer churn dataset.

This module investigates a single question: how does observed churn vary across
the ``MonthlyCharges`` distribution? It answers in two ways. The distribution
comparison describes ``MonthlyCharges`` separately for the retained and the
churned population, so the two can be read against each other on centre,
spread, and range. The quartile analysis divides the observed charges into four
descriptive quantile groups and reports the churn rate measured inside each.

The two views answer the same question from opposite directions. The first
describes a continuous variable inside each churn population, as the tenure
analysis does. The second measures churn inside groups of customers, as the
contract analysis does, except that the groups are derived from the observed
distribution instead of being read from a categorical column.

The quantile groups are a device for looking at the distribution in ordered
quarters. They are not business categories, not risk categories, not
thresholds, and not a feature: no dollar boundary is defined here, and the
boundaries reported alongside each group are the quantiles this dataset
happened to produce rather than ranges that would hold for another one.

The frame passed in is only read from: nothing is imputed, encoded, cleaned,
removed, or overwritten. Unlike the tenure analyses, which count a missing
tenure and carry on, this module refuses a missing ``MonthlyCharges`` or a
missing ``Churn`` value outright. Both the quantile boundaries and the group
populations are computed over every record, so a customer who could not be
placed in a group, or placed but not classified, would leave the reported
groups unable to account for the frame they came from.

Rates here are proportions between 0 and 1 rather than percentages.

The result describes churn as observed across portions of the recorded charge
distribution. It does not establish that ``MonthlyCharges`` causes churn, it
identifies no threshold, it selects no feature, and it is not tested for
significance. Whether ``MonthlyCharges`` carries predictive value is a question
for the model that is eventually fitted, not for this summary.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# The target contract is the one the tenure analyses already work under: the
# same target column and the same documented values. Importing it keeps the
# modules from drifting apart, which restating the values here would invite.
from churner.analysis.analyze_tenure_churn import (
    CHURNED_VALUE,
    EXPECTED_CHURN_VALUES,
    RETAINED_VALUE,
    TARGET_COLUMN,
)

# --- Input columns ---
# The data dictionary documents ``MonthlyCharges`` as the amount charged to the
# customer each month, a continuous currency value.
MONTHLY_CHARGES_COLUMN = "MonthlyCharges"

REQUIRED_COLUMNS = (MONTHLY_CHARGES_COLUMN, TARGET_COLUMN)

# --- The descriptive quantile groups ---
# Four groups, so each covers a quarter of the observed distribution. The
# boundaries are quantiles of the charges actually recorded, so they are read
# from the data rather than chosen here.
GROUP_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
GROUP_LABEL_PREFIX = "Q"

# --- Distribution comparison columns ---
CHURN_STATUS_COLUMN = "churn_status"
CUSTOMER_COUNT_COLUMN = "customer_count"
MEAN_CHARGES_COLUMN = "mean_monthly_charges"
MEDIAN_CHARGES_COLUMN = "median_monthly_charges"
STD_CHARGES_COLUMN = "std_monthly_charges"
MIN_CHARGES_COLUMN = "min_monthly_charges"
Q1_CHARGES_COLUMN = "q1_monthly_charges"
Q3_CHARGES_COLUMN = "q3_monthly_charges"
MAX_CHARGES_COLUMN = "max_monthly_charges"

# The descriptive statistics, kept apart from the count because they are
# undefined rather than zero when a population holds no observation.
STATISTIC_COLUMNS = (
    MEAN_CHARGES_COLUMN,
    MEDIAN_CHARGES_COLUMN,
    STD_CHARGES_COLUMN,
    MIN_CHARGES_COLUMN,
    Q1_CHARGES_COLUMN,
    Q3_CHARGES_COLUMN,
    MAX_CHARGES_COLUMN,
)

DISTRIBUTION_COLUMNS = (
    CHURN_STATUS_COLUMN,
    CUSTOMER_COUNT_COLUMN,
    *STATISTIC_COLUMNS,
)

# --- Quantile group columns ---
# The boundaries come directly after the group label, since they say which part
# of the distribution the row describes.
CHARGE_GROUP_COLUMN = "charge_group"
LOWER_CHARGE_COLUMN = "lower_charge"
UPPER_CHARGE_COLUMN = "upper_charge"
CHURNED_COUNT_COLUMN = "churned_count"
RETAINED_COUNT_COLUMN = "retained_count"
CHURN_RATE_COLUMN = "churn_rate"

RESULT_COLUMNS = (
    CHARGE_GROUP_COLUMN,
    LOWER_CHARGE_COLUMN,
    UPPER_CHARGE_COLUMN,
    CUSTOMER_COUNT_COLUMN,
    CHURNED_COUNT_COLUMN,
    RETAINED_COUNT_COLUMN,
    CHURN_RATE_COLUMN,
)


def validate_inputs(df: pd.DataFrame) -> None:
    """Refuse any frame this analysis cannot describe honestly.

    Four conditions are checked. The required columns must be present. The
    target must hold only its documented values, because it decides whether a
    customer counts as churned or retained and an unrecognised value can be
    called neither. ``MonthlyCharges`` must be numeric, because the summary is
    arithmetic on it and its groups are quantiles of it; a column of text is
    refused rather than converted here, since deciding what an unparseable
    entry means is a data-cleaning decision and not this module's to make. A
    frame with no rows carries no charge observation to judge, so its dtype is
    left alone.

    Missing values in either column are refused as well, which is stricter than
    the tenure analyses, where a missing tenure is counted and carried. The
    reason is what the quantile groups claim: they are computed over every
    record and are reported as accounting for the frame in full. A customer
    with no recorded charge belongs to no group, and one with no recorded churn
    status sits in a group's denominator while counting as neither churned nor
    retained. Either would leave the reported groups quietly failing to
    reconcile with the data they were built from, so the shortfall is reported
    instead of absorbed.

    Parameters
    ----------
    df : pd.DataFrame
        The frame about to be analysed. Only read from.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds a value outside
        ``EXPECTED_CHURN_VALUES``, if ``MonthlyCharges`` is not numeric, or if
        either required column holds a missing value.
    """
    missing_columns = [
        column_name for column_name in REQUIRED_COLUMNS if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}. "
            f"Columns present: {list(df.columns)}"
        )

    # A missing target is excluded from this check so that it is reported below
    # as the missing value it is, rather than as an unrecognised one.
    churn_values = df[TARGET_COLUMN]
    recognized = churn_values.isin(EXPECTED_CHURN_VALUES) | churn_values.isna()
    unexpected_values = churn_values[~recognized]
    if not unexpected_values.empty:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' holds value(s) outside "
            f"{list(EXPECTED_CHURN_VALUES)}: {unexpected_values.unique().tolist()}. "
            "The target decides whether a customer counts as churned or retained, so an "
            "unexpected value cannot be counted as either."
        )

    charge_values = df[MONTHLY_CHARGES_COLUMN]
    if len(df) > 0 and (not is_numeric_dtype(charge_values) or is_bool_dtype(charge_values)):
        raise ValueError(
            f"Column '{MONTHLY_CHARGES_COLUMN}' must be numeric to be described "
            f"statistically and divided into quantile groups; its dtype is "
            f"'{charge_values.dtype}'."
        )

    for column_name in REQUIRED_COLUMNS:
        missing_count = int(df[column_name].isna().sum())
        if missing_count != 0:
            raise ValueError(
                f"{missing_count} record(s) hold no '{column_name}' value. Every record has to "
                "be placed in a charge group and counted as churned or retained for the groups "
                "to account for the data they were built from, so those records need "
                "investigating before this analysis can run."
            )


def describe_charges(charges: pd.Series) -> dict[str, float]:
    """Summarise one population's charge observations.

    Every statistic is reported as a float, whatever numeric type the column
    holds, so the summary keeps one consistent shape across populations and
    datasets. A population with no observation has no centre, spread, or range
    to measure, so each statistic is ``NaN``: undefined, which is not the same
    claim as a measured zero. The standard deviation is the sample one, and is
    likewise ``NaN`` for a population of one, which has no spread to measure.

    Parameters
    ----------
    charges : pd.Series
        ``MonthlyCharges`` observations of a single churn population.

    Returns
    -------
    dict[str, float]
        The seven descriptive statistics, keyed by result column name.
    """
    if charges.empty:
        return {statistic: float("nan") for statistic in STATISTIC_COLUMNS}

    return {
        MEAN_CHARGES_COLUMN: float(charges.mean()),
        MEDIAN_CHARGES_COLUMN: float(charges.median()),
        STD_CHARGES_COLUMN: float(charges.std()),
        MIN_CHARGES_COLUMN: float(charges.min()),
        Q1_CHARGES_COLUMN: float(charges.quantile(0.25)),
        Q3_CHARGES_COLUMN: float(charges.quantile(0.75)),
        MAX_CHARGES_COLUMN: float(charges.max()),
    }


def describe_monthly_charges_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the charge distribution of churned and of retained customers.

    Both expected churn statuses are always reported, in the order ``"No"``,
    ``"Yes"``, so the comparison keeps a stable shape and a population that
    holds no customer shows up as a row of zero customers rather than as an
    absent row. Each population's statistics are computed from its own
    observations alone.

    The two distributions may differ on any of centre, spread, or range, and
    the statistics are reported side by side so that a difference in one is not
    read as a difference in all. Any difference is an association observed in
    the data: it does not establish that the amount charged causes churn, nor
    that churn explains the amount recorded.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``MonthlyCharges`` and ``Churn``
        columns. Only read from; never modified.

    Returns
    -------
    pd.DataFrame
        One row per expected churn status, with the columns ``churn_status``,
        ``customer_count``, ``mean_monthly_charges``,
        ``median_monthly_charges``, ``std_monthly_charges``,
        ``min_monthly_charges``, ``q1_monthly_charges``, ``q3_monthly_charges``,
        and ``max_monthly_charges``. Statistics are ``NaN`` where the
        population holds nothing to measure them from.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds an unexpected value,
        if ``MonthlyCharges`` is not numeric, or if either required column
        holds a missing value.
    """
    validate_inputs(df)

    churn_values = df[TARGET_COLUMN]
    charge_values = df[MONTHLY_CHARGES_COLUMN]

    population_summaries = []
    for churn_status in EXPECTED_CHURN_VALUES:
        # Selecting with a mask returns a new series, so the population's
        # charges are never written back to the caller's frame.
        population_charges = charge_values[churn_values == churn_status]

        population_summaries.append(
            {
                CHURN_STATUS_COLUMN: churn_status,
                CUSTOMER_COUNT_COLUMN: int(len(population_charges)),
                **describe_charges(population_charges),
            }
        )

    return pd.DataFrame(population_summaries, columns=list(DISTRIBUTION_COLUMNS))


def build_group_boundaries(charges: pd.Series) -> list[float]:
    """Derive the charge boundaries the quantile groups are divided at.

    The boundaries are the quantiles listed in ``GROUP_QUANTILES``, so they
    describe the distribution actually recorded rather than any range chosen
    here. Repeated charges can place two of those quantiles at the same value,
    which would define a group no charge could fall inside; the repeated
    boundary is dropped, leaving fewer and wider groups rather than empty ones.

    Parameters
    ----------
    charges : pd.Series
        The ``MonthlyCharges`` observations to be divided. Assumed non-empty
        and free of missing values, which ``validate_inputs`` establishes.

    Returns
    -------
    list[float]
        The boundaries, ascending, running from the lowest observed charge to
        the highest. A distribution holding one distinct charge has no interior
        boundary to be divided at, so that charge is returned as both the lower
        and the upper boundary of the single group it forms.
    """
    quantiles = charges.quantile(list(GROUP_QUANTILES)).drop_duplicates()
    boundaries = [float(boundary) for boundary in quantiles]

    if len(boundaries) == 1:
        return boundaries * 2

    return boundaries


def assign_charge_groups(charges: pd.Series, boundaries: list[float]) -> pd.Series:
    """Place each customer in the quantile group their charge falls in.

    Groups are closed on the right, and the lowest group also holds its own
    lower boundary, so every observed charge belongs to exactly one group and
    customers charged the same amount are never split across two. That comes at
    the cost of groups of unequal size: where one charge repeats across a
    boundary, the group holding it absorbs every customer charged it.

    Parameters
    ----------
    charges : pd.Series
        The ``MonthlyCharges`` observations to be divided. Assumed non-empty
        and free of missing values, which ``validate_inputs`` establishes.
    boundaries : list[float]
        The boundaries from ``build_group_boundaries``, for the same charges.

    Returns
    -------
    pd.Series
        The zero-based index of each customer's group, aligned to the index of
        ``charges``. Group ``i`` runs from ``boundaries[i]`` to
        ``boundaries[i + 1]``.
    """
    # A single distinct charge is divided by nothing, so everyone shares the
    # one group spanning it. ``pd.cut`` cannot express that, since its bin
    # edges have to increase.
    if boundaries[0] == boundaries[-1]:
        return pd.Series(0, index=charges.index)

    return pd.cut(charges, bins=boundaries, labels=False, include_lowest=True)


def empty_result() -> pd.DataFrame:
    """Build the result for a population holding no charge observation.

    The schema is declared rather than inferred, so a frame with nothing to
    group still carries the same columns and the same dtypes as a populated one
    and can be consumed without a special case.

    Returns
    -------
    pd.DataFrame
        An empty frame with the seven result columns.
    """
    return pd.DataFrame(
        {
            CHARGE_GROUP_COLUMN: pd.Series(dtype="str"),
            LOWER_CHARGE_COLUMN: pd.Series(dtype="float64"),
            UPPER_CHARGE_COLUMN: pd.Series(dtype="float64"),
            CUSTOMER_COUNT_COLUMN: pd.Series(dtype="int64"),
            CHURNED_COUNT_COLUMN: pd.Series(dtype="int64"),
            RETAINED_COUNT_COLUMN: pd.Series(dtype="int64"),
            CHURN_RATE_COLUMN: pd.Series(dtype="float64"),
        }
    )


def analyze_monthly_charges_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Measure the observed churn rate across quarters of the charge distribution.

    The recorded charges are divided at their own quartiles into the groups
    ``"Q1"`` through ``"Q4"``, ordered from the lowest quarter of observed
    charges to the highest, and each group's own churn rate is reported
    alongside the counts it was derived from. The boundaries are read from the
    data: no dollar range is defined here, and the ones reported describe this
    dataset rather than charges in general.

    Groups are closed on the right and the lowest group holds its own lower
    boundary, so every record enters exactly one group and the group
    populations add back up to the length of the frame. Customers charged the
    same amount always share a group, which leaves the groups unequal in size
    where a charge repeats across a boundary; the customer counts reported
    alongside the rates are what makes that visible.

    Fewer than four groups are reported when the observed distribution does not
    support four, as when repeated charges place two quartiles at the same
    value, or when every customer is charged the same amount and one group
    spans them all. The groups are numbered ``"Q1"`` upward as before,
    describing the quarters the data can actually be divided into, since a
    group holding no customer would carry a rate over an empty population.

    Each row reconciles, ``customer_count`` equalling ``churned_count`` plus
    ``retained_count``, and the churn rate is that group's share of churned
    customers as a proportion between 0 and 1. Every group holds at least one
    customer, so no rate is divided by an empty population.

    The variation across groups is an association observed in the data. It does
    not establish that the amount charged causes churn, it identifies no
    threshold, it makes no customer high risk, and it is not tested for
    significance. Whether ``MonthlyCharges`` is worth keeping as a predictor is
    settled by the model that is eventually fitted, not here.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``MonthlyCharges`` and ``Churn``
        columns. Only read from; never modified.

    Returns
    -------
    pd.DataFrame
        One row per quantile group, ascending, with the columns
        ``charge_group``, ``lower_charge``, ``upper_charge``,
        ``customer_count``, ``churned_count``, ``retained_count``, and
        ``churn_rate``. Empty, with the same columns, when the frame holds no
        record.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds an unexpected value,
        if ``MonthlyCharges`` is not numeric, or if either required column
        holds a missing value.
    """
    validate_inputs(df)

    if df.empty:
        return empty_result()

    charge_values = df[MONTHLY_CHARGES_COLUMN]
    churn_values = df[TARGET_COLUMN]

    boundaries = build_group_boundaries(charge_values)
    # A series of its own, rather than a column written into the caller's
    # frame, which the analysis has no business adding to.
    group_indexes = assign_charge_groups(charge_values, boundaries)

    churned = churn_values == CHURNED_VALUE
    retained = churn_values == RETAINED_VALUE

    # Grouping reports only the groups customers were actually placed in, so a
    # pair of boundaries no charge falls between is never reported as a group
    # of zero customers carrying an undefined rate.
    group_counts = pd.DataFrame(
        {
            CUSTOMER_COUNT_COLUMN: churned.groupby(group_indexes, sort=True).size(),
            CHURNED_COUNT_COLUMN: churned.groupby(group_indexes, sort=True).sum(),
            RETAINED_COUNT_COLUMN: retained.groupby(group_indexes, sort=True).sum(),
        }
    )

    observed_groups = [int(group_index) for group_index in group_counts.index]

    return group_counts.reset_index(drop=True).assign(
        **{
            # Numbered by position, so the labels read from Q1 upward without a
            # gap even where the distribution supported fewer than four groups.
            CHARGE_GROUP_COLUMN: [
                f"{GROUP_LABEL_PREFIX}{position}"
                for position in range(1, len(observed_groups) + 1)
            ],
            LOWER_CHARGE_COLUMN: [boundaries[group_index] for group_index in observed_groups],
            UPPER_CHARGE_COLUMN: [boundaries[group_index + 1] for group_index in observed_groups],
            CHURN_RATE_COLUMN: lambda counts: counts[CHURNED_COUNT_COLUMN]
            / counts[CUSTOMER_COUNT_COLUMN],
        }
    )[list(RESULT_COLUMNS)]
