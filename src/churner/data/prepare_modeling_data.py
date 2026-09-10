"""Deterministic preparation of the Telco customer churn dataset for modelling.

This module performs the first two steps that stand between the raw file and a
model: it reads ``TotalCharges`` as the amount it always was, and it separates
the target from the predictors.

Both steps are deterministic. Neither one learns anything from the data, so
neither can leak information from records the model has not been trained on,
which is why they belong here rather than after the train/test split. Nothing
is imputed, encoded, scaled, engineered, split, or dropped: those steps either
fit parameters on the data or change what the data says, and they are the
concern of the stage that follows this one.

``TotalCharges`` is documented as a continuous amount but stored as text,
because 11 records hold a blank string instead of a figure. Converting the
column turns that blank into a missing value, which is a change of
representation rather than of content: the same 11 records that carried no
amount before still carry none afterwards. What that absence means, and what
should stand in for it, is left to the imputation stage. A value that is
neither blank nor readable as an amount is refused instead, since it is not
documented, and coercing it would make it indistinguishable from the blanks
the data dictionary does document.

The separation excludes ``customerID`` from the predictors because it
identifies a customer rather than describing one, and ``Churn`` because it is
the outcome being predicted. Both are excluded from the feature frame only:
the caller's own frame keeps every column it came with, so the identifier
remains available for tracing records back to source.
"""

import pandas as pd

# --- Input columns ---
# The data dictionary documents ``customerID`` as the per-customer identifier,
# ``TotalCharges`` as the cumulative amount charged, stored as text in the raw
# file, and ``Churn`` as the label column.
ID_COLUMN = "customerID"
TOTAL_CHARGES_COLUMN = "TotalCharges"
TARGET_COLUMN = "Churn"

REQUIRED_COLUMNS = (ID_COLUMN, TOTAL_CHARGES_COLUMN, TARGET_COLUMN)

# --- Columns kept out of the predictors ---
# The identifier carries no information about a customer beyond which one they
# are, and the target is the outcome the model is asked to predict. Leaving
# either in the feature frame would let a model fit on it.
NON_FEATURE_COLUMNS = (ID_COLUMN, TARGET_COLUMN)

# --- Converted representation ---
# The blank entries are what forced the raw column to be read as text. The
# converted column holds the amounts as ``MonthlyCharges`` already does, so the
# two currency columns share one dtype and one missing-value marker.
BLANK_VALUE = ""
TOTAL_CHARGES_DTYPE = "float64"


def validate_inputs(df: pd.DataFrame) -> None:
    """Refuse any frame this preparation cannot be performed on.

    Only the presence of the required columns is checked. The target's values
    are not, and neither are the predictors': this step changes how one column
    is represented and which columns are handed to the model, and it draws no
    conclusion from what any of them hold.

    Parameters
    ----------
    df : pd.DataFrame
        The frame about to be prepared. Only read from.

    Raises
    ------
    ValueError
        If a required column is absent.
    """
    missing_columns = [
        column_name for column_name in REQUIRED_COLUMNS if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required column(s) not found: {missing_columns}. "
            f"Columns present: {list(df.columns)}"
        )


def to_numeric_total_charges(total_charges: pd.Series) -> pd.Series:
    """Read the recorded ``TotalCharges`` text as the amounts it stands for.

    An entry that is empty or holds nothing but whitespace records no amount,
    so it becomes a missing value: the blank is a representation of absence,
    and the conversion carries that absence across rather than resolving it.
    Every other entry is parsed as the amount it reads as, unrounded, so no
    recorded figure is altered by being converted.

    Surrounding whitespace is stripped before parsing, which is what lets a
    whitespace-only entry be recognised as blank at all. An entry that survives
    that and still cannot be read as an amount is refused, because the data
    dictionary documents no such value: coercing it would quietly file it with
    the blanks, where the imputation stage would later fill it in as though an
    amount had simply been absent.

    Parameters
    ----------
    total_charges : pd.Series
        The ``TotalCharges`` column as recorded, ordinarily text.

    Returns
    -------
    pd.Series
        The same values as ``float64`` amounts, aligned to the index of
        ``total_charges``, with ``NaN`` wherever no amount was recorded.

    Raises
    ------
    ValueError
        If an entry is neither blank nor readable as an amount.
    """
    recorded_amounts = total_charges.astype("string").str.strip().replace(BLANK_VALUE, pd.NA)

    try:
        return pd.to_numeric(recorded_amounts).astype(TOTAL_CHARGES_DTYPE)
    except (TypeError, ValueError) as conversion_failure:
        raise ValueError(
            f"Column '{TOTAL_CHARGES_COLUMN}' holds an entry that is neither blank nor "
            f"readable as an amount: {conversion_failure}. Such a value is not documented, "
            "so what it stands for has to be investigated rather than converted here."
        ) from conversion_failure


def prepare_modeling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into predictors and target, with charges read as numbers.

    The work is done on a copy, so the caller's frame keeps the text column and
    every column it came with, this one included. The predictors are everything
    left after the identifier and the target are set aside; the two frames share
    an index, so each row of predictors still lines up with its own outcome.

    Only the representation of ``TotalCharges`` changes. No value is imputed,
    encoded, scaled, or derived, no row is dropped, and no data is split into
    training and test sets. Missing amounts are handed on as missing, which is
    what lets the imputation that follows be fitted on training records alone.

    Parameters
    ----------
    df : pd.DataFrame
        Customer records holding at least the ``customerID``, ``TotalCharges``,
        and ``Churn`` columns. Only read from; never modified.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The predictor frame and the ``Churn`` target. The predictors hold every
        column of ``df`` except ``customerID`` and ``Churn``, with
        ``TotalCharges`` converted; the target holds the recorded churn labels
        unchanged.

    Raises
    ------
    ValueError
        If a required column is absent, or if a ``TotalCharges`` entry is
        neither blank nor readable as an amount.
    """
    validate_inputs(df)

    prepared_df = df.copy(deep=True)
    prepared_df[TOTAL_CHARGES_COLUMN] = to_numeric_total_charges(prepared_df[TOTAL_CHARGES_COLUMN])

    features = prepared_df.drop(columns=list(NON_FEATURE_COLUMNS))
    target = prepared_df[TARGET_COLUMN]

    return features, target
