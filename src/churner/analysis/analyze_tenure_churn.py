"""Tenure vs. churn analysis of the Telco customer churn dataset.

This module investigates a single question: do churned and retained customers
show meaningfully different tenure distributions? It answers with a numerical
descriptive summary of ``tenure`` for each of the two churn populations, so the
two distributions can be compared on centre, spread, and range.

Unlike the contract analysis, which asks how often churn occurs inside a
category, this one describes a continuous variable inside each churn
population. ``Churn`` therefore defines the populations being compared rather
than being counted, and an unexpected target value cannot be classified into
either population: it is refused instead of guessed at.

The frame passed in is only read from: nothing is imputed, encoded, cleaned,
removed, or overwritten. Missing tenure values are counted rather than filled,
and unusual observations such as a tenure of zero are kept, because an
exploratory summary should describe the data as recorded.

Charts, tenure bands, and significance tests are outside this module's scope.
The visualization layer works from the raw tenure observations rather than from
the summary returned here, which cannot reconstruct a distribution.

The result describes observed differences between the two populations. It does
not establish why those differences exist.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# --- Input columns ---
# The data dictionary documents ``tenure`` as the number of months the customer
# has stayed with the company, and ``Churn`` as the label column, where "Yes"
# means the customer left during the last period.
TENURE_COLUMN = "tenure"
TARGET_COLUMN = "Churn"

# The two populations being compared, in reporting order: the retained group
# first, so the churned group reads against it.
RETAINED_VALUE = "No"
CHURNED_VALUE = "Yes"
EXPECTED_CHURN_VALUES = (RETAINED_VALUE, CHURNED_VALUE)

# --- Output columns ---
CHURN_STATUS_COLUMN = "churn_status"
CUSTOMER_COUNT_COLUMN = "customer_count"
MISSING_TENURE_COUNT_COLUMN = "missing_tenure_count"
VALID_TENURE_COUNT_COLUMN = "valid_tenure_count"
MEAN_TENURE_COLUMN = "mean_tenure"
MEDIAN_TENURE_COLUMN = "median_tenure"
Q1_TENURE_COLUMN = "q1_tenure"
Q3_TENURE_COLUMN = "q3_tenure"
IQR_TENURE_COLUMN = "iqr_tenure"
MIN_TENURE_COLUMN = "min_tenure"
MAX_TENURE_COLUMN = "max_tenure"

# The descriptive statistics, kept apart from the counts because they are
# undefined rather than zero when a population holds no valid observation.
STATISTIC_COLUMNS = (
    MEAN_TENURE_COLUMN,
    MEDIAN_TENURE_COLUMN,
    Q1_TENURE_COLUMN,
    Q3_TENURE_COLUMN,
    IQR_TENURE_COLUMN,
    MIN_TENURE_COLUMN,
    MAX_TENURE_COLUMN,
)

RESULT_COLUMNS = (
    CHURN_STATUS_COLUMN,
    CUSTOMER_COUNT_COLUMN,
    MISSING_TENURE_COUNT_COLUMN,
    VALID_TENURE_COUNT_COLUMN,
    *STATISTIC_COLUMNS,
)


def describe_tenure(valid_tenure: pd.Series) -> dict[str, float]:
    """Summarise one population's tenure observations.

    Every statistic is reported as a float, whatever numeric type the column
    holds, so the summary keeps one consistent shape across populations and
    datasets. A population with no valid observation has no centre, spread, or
    range to measure, so each statistic is ``NaN``: undefined, which is not the
    same claim as a measured zero.

    Parameters
    ----------
    valid_tenure : pd.Series
        Tenure observations of a single churn population, already stripped of
        missing values.

    Returns
    -------
    dict[str, float]
        The seven descriptive statistics, keyed by result column name.
    """
    if valid_tenure.empty:
        return {statistic: float("nan") for statistic in STATISTIC_COLUMNS}

    first_quartile = valid_tenure.quantile(0.25)
    third_quartile = valid_tenure.quantile(0.75)

    return {
        MEAN_TENURE_COLUMN: float(valid_tenure.mean()),
        MEDIAN_TENURE_COLUMN: float(valid_tenure.median()),
        Q1_TENURE_COLUMN: float(first_quartile),
        Q3_TENURE_COLUMN: float(third_quartile),
        IQR_TENURE_COLUMN: float(third_quartile - first_quartile),
        MIN_TENURE_COLUMN: float(valid_tenure.min()),
        MAX_TENURE_COLUMN: float(valid_tenure.max()),
    }


def validate_inputs(df: pd.DataFrame) -> None:
    """Refuse any frame this analysis cannot summarise honestly.

    Three conditions are checked. The required columns must be present. The
    target must hold only its documented values, because it decides which
    population each customer belongs to and an unrecognised value cannot be
    assigned to either one. Tenure must be numeric, because the summary is
    arithmetic on it; a column of text is refused rather than converted here,
    since deciding what an unparseable entry means is a data-cleaning decision
    and not this module's to make. A frame with no rows carries no tenure
    observation to judge, so its dtype is left alone.

    Missing values are not an error in either column: a missing tenure is
    counted, and a missing target simply places that customer in neither
    population.

    Parameters
    ----------
    df : pd.DataFrame
        The frame about to be analysed. Only read from.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds a value outside
        ``EXPECTED_CHURN_VALUES``, or if ``tenure`` is not numeric.
    """
    missing_columns = [
        column_name
        for column_name in (TENURE_COLUMN, TARGET_COLUMN)
        if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}. "
            f"Columns present: {list(df.columns)}"
        )

    churn_values = df[TARGET_COLUMN]
    recognized = churn_values.isin(EXPECTED_CHURN_VALUES) | churn_values.isna()
    unexpected_values = churn_values[~recognized]
    if not unexpected_values.empty:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' holds value(s) outside "
            f"{list(EXPECTED_CHURN_VALUES)}: {unexpected_values.unique().tolist()}. "
            "The target defines the populations being compared, so an unexpected "
            "value cannot be classified as churned or retained."
        )

    tenure_values = df[TENURE_COLUMN]
    if len(df) > 0 and (not is_numeric_dtype(tenure_values) or is_bool_dtype(tenure_values)):
        raise ValueError(
            f"Column '{TENURE_COLUMN}' must be numeric to be described "
            f"statistically; its dtype is '{tenure_values.dtype}'."
        )


def analyze_tenure_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the tenure distribution of churned and of retained customers.

    Both expected churn statuses are always reported, in the order ``"No"``,
    ``"Yes"``, so the comparison keeps a stable shape and a population that
    holds no customer shows up as a row of zero counts rather than as an absent
    row.

    Each population's statistics are computed from its valid tenure
    observations alone. Missing tenure is counted separately and never stands
    in as a zero, which would pull the mean and the minimum toward a month
    count nobody was observed to have. A recorded tenure of zero is an
    observation like any other and is kept. Unusual values, negative ones
    included, are described rather than removed: this is an exploratory summary
    of the data as recorded, and dropping observations for looking wrong would
    hide exactly what the summary exists to reveal.

    The counts reconcile per population, ``customer_count`` equalling
    ``valid_tenure_count + missing_tenure_count``. Across the whole frame they
    need not: a customer whose target value is missing belongs to neither
    population and is counted in neither row.

    Any difference the summary shows is an association observed in the data. It
    does not establish that tenure causes churn, nor that churn causes the
    tenure recorded; both are consistent with these numbers.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``tenure`` and ``Churn`` columns.
        Only read from; never modified.

    Returns
    -------
    pd.DataFrame
        One row per expected churn status, with the columns ``churn_status``,
        ``customer_count``, ``missing_tenure_count``, ``valid_tenure_count``,
        ``mean_tenure``, ``median_tenure``, ``q1_tenure``, ``q3_tenure``,
        ``iqr_tenure``, ``min_tenure``, and ``max_tenure``. Statistics are
        ``NaN`` where the population holds no valid tenure observation.

    Raises
    ------
    ValueError
        If a required column is absent, if ``Churn`` holds an unexpected value,
        or if ``tenure`` is not numeric.
    """
    validate_inputs(df)

    churn_values = df[TARGET_COLUMN]
    tenure_values = df[TENURE_COLUMN]

    population_summaries = []
    for churn_status in EXPECTED_CHURN_VALUES:
        # Selecting with a mask returns a new series, so the population's
        # tenure values are never written back to the caller's frame.
        population_tenure = tenure_values[churn_values == churn_status]
        valid_tenure = population_tenure.dropna()

        population_summaries.append(
            {
                CHURN_STATUS_COLUMN: churn_status,
                CUSTOMER_COUNT_COLUMN: int(len(population_tenure)),
                MISSING_TENURE_COUNT_COLUMN: int(population_tenure.isna().sum()),
                VALID_TENURE_COUNT_COLUMN: int(len(valid_tenure)),
                **describe_tenure(valid_tenure),
            }
        )

    return pd.DataFrame(population_summaries, columns=list(RESULT_COLUMNS))
