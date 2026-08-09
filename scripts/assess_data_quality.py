"""Non-destructive data-quality inspection of the Telco customer churn dataset.

This script loads the raw dataset through the reusable
``churner.data.load_dataset.load_dataset`` function and reports observations
about its size, duplication, missing values, categorical value distributions,
and explicit semantic consistency rules. It never modifies, cleans, or
transforms the data, and it draws no conclusion about whether the dataset is
acceptable; it reports evidence only, for manual review.
"""

# --- Standard library imports ---
import sys
from pathlib import Path

# --- Third-party imports ---
import pandas as pd

# --- Make the reusable "churner" package importable ---
# The script lives in ``<project_root>/scripts``, so the project root is one
# level up. The source code lives under ``<project_root>/src``. Adding that
# directory to ``sys.path`` lets us import the existing module without having
# the package installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))

# --- Import the existing, reusable data-loading function ---
# We deliberately reuse this function instead of calling ``pandas.read_csv``
# directly, so the assessment runs against the real loading logic.
from churner.data.load_dataset import load_dataset

# --- Resolve the dataset path relative to the project root ---
# Building the path from ``PROJECT_ROOT`` avoids hardcoding an absolute path
# and keeps the script portable across machines.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Identifier column expected to be unique per customer.
ID_COLUMN = "customerID"

# Text columns kept out of the categorical profile. ``customerID`` is a
# per-customer identifier, and ``TotalCharges`` is a continuous amount that
# pandas reads as text, so neither describes a category worth enumerating.
NON_CATEGORICAL_TEXT_COLUMNS = (ID_COLUMN, "TotalCharges")

# --- Semantic rule: customers without internet service ---
# Categorical profiling showed ``InternetService == "No"`` in 1,526 rows and
# ``"No internet service"`` in exactly 1,526 rows of each dependent column
# below, which is the evidence behind this expected correspondence.
INTERNET_SERVICE_COLUMN = "InternetService"
NO_INTERNET_SERVICE = "No"
NO_INTERNET_DEPENDENT_VALUE = "No internet service"
INTERNET_DEPENDENT_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

# --- Semantic rule: customers without phone service ---
# Categorical profiling showed ``PhoneService == "No"`` in 682 rows and
# ``"No phone service"`` in exactly 682 rows of ``MultipleLines``, which is the
# evidence behind this expected correspondence.
PHONE_SERVICE_COLUMN = "PhoneService"
NO_PHONE_SERVICE = "No"
MULTIPLE_LINES_COLUMN = "MultipleLines"
NO_PHONE_DEPENDENT_VALUE = "No phone service"


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Count rows that are exact duplicates of an earlier row."""
    return int(df.duplicated().sum())


def count_duplicate_ids(df: pd.DataFrame, id_column: str) -> int:
    """Count values in ``id_column`` that repeat a value seen earlier."""
    return int(df[id_column].duplicated().sum())


def count_missing_values(df: pd.DataFrame) -> pd.Series:
    """Count missing values per column, as detected by pandas.

    Only pandas' own notion of missingness is used here; placeholder values
    such as empty strings are not interpreted.
    """
    return df.isna().sum()


def select_categorical_columns(df: pd.DataFrame) -> list[str]:
    """List the columns to profile as categories.

    Numeric columns are skipped because they hold measures rather than
    categories, as are the columns named in ``NON_CATEGORICAL_TEXT_COLUMNS``.
    """
    return [
        str(name)
        for name, dtype in df.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
        and name not in NON_CATEGORICAL_TEXT_COLUMNS
    ]


def print_categorical_profile(df: pd.DataFrame, row_count: int) -> None:
    """Print every value of each categorical column with its count and share.

    Values are listed from most to least frequent. Percentages are relative to
    the full dataset, and missing values are counted rather than skipped.
    """
    for column_name in select_categorical_columns(df):
        value_counts = df[column_name].value_counts(dropna=False)
        value_width = max([len("Value")] + [len(str(v)) for v in value_counts.index]) + 2

        print(f"Column: {column_name}")
        print(f"{'Value':<{value_width}}{'Count':>10}{'Percent':>10}")
        for value, count in value_counts.items():
            value_percent = count / row_count * 100
            print(
                f"{str(value):<{value_width}}"
                f"{count:>10}"
                f"{value_percent:>9.2f}%"
            )
        print()


def find_no_internet_violations(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Find rows that depart from the no-internet-service correspondence.

    Looks only at rows where ``InternetService`` equals ``"No"`` and returns,
    for each dependent column that holds anything other than
    ``"No internet service"``, the observed values and how many rows carry
    each. Columns with no such rows are omitted. The frame is only read from.
    """
    no_internet_rows = df.loc[df[INTERNET_SERVICE_COLUMN] == NO_INTERNET_SERVICE]

    violations: dict[str, pd.Series] = {}
    for column_name in INTERNET_DEPENDENT_COLUMNS:
        column_values = no_internet_rows[column_name]
        unexpected = column_values[column_values != NO_INTERNET_DEPENDENT_VALUE]
        if not unexpected.empty:
            violations[column_name] = unexpected.value_counts(dropna=False)
    return violations


def print_no_internet_rule(df: pd.DataFrame) -> None:
    """Print evidence for the no-internet-service consistency rule.

    Reports any departures from the rule for manual investigation; it does not
    decide whether the affected records are erroneous.
    """
    no_internet_count = int((df[INTERNET_SERVICE_COLUMN] == NO_INTERNET_SERVICE).sum())
    violations = find_no_internet_violations(df)

    print("Rule:")
    print(
        f'  {INTERNET_SERVICE_COLUMN} = "{NO_INTERNET_SERVICE}" is expected to '
        f'correspond to "{NO_INTERNET_DEPENDENT_VALUE}"'
    )
    print("  in the dependent internet-service columns.")
    print(
        f'  Rows with {INTERNET_SERVICE_COLUMN} = "{NO_INTERNET_SERVICE}": '
        f"{no_internet_count}"
    )
    print()

    if not violations:
        print("No semantic consistency violations detected for this rule.")
        return

    print("Semantic consistency violations (require investigation):")
    for column_name, observed_counts in violations.items():
        print(f"{column_name}:")
        for observed_value, affected_rows in observed_counts.items():
            print(f"  Expected:      {NO_INTERNET_DEPENDENT_VALUE}")
            print(f"  Observed:      {observed_value}")
            print(f"  Affected rows: {affected_rows}")


def find_no_phone_violations(df: pd.DataFrame) -> pd.Series:
    """Find rows that depart from the no-phone-service correspondence.

    Looks only at rows where ``PhoneService`` equals ``"No"`` and returns the
    ``MultipleLines`` values other than ``"No phone service"`` together with
    how many rows carry each. The Series is empty when the rule holds, and the
    frame is only read from.
    """
    no_phone_rows = df.loc[df[PHONE_SERVICE_COLUMN] == NO_PHONE_SERVICE]

    column_values = no_phone_rows[MULTIPLE_LINES_COLUMN]
    unexpected = column_values[column_values != NO_PHONE_DEPENDENT_VALUE]
    return unexpected.value_counts(dropna=False)


def print_no_phone_rule(df: pd.DataFrame) -> None:
    """Print evidence for the no-phone-service consistency rule.

    Reports any departures from the rule for manual investigation; it does not
    decide whether the affected records are erroneous.
    """
    no_phone_count = int((df[PHONE_SERVICE_COLUMN] == NO_PHONE_SERVICE).sum())
    observed_counts = find_no_phone_violations(df)

    print("Rule:")
    print(
        f'  {PHONE_SERVICE_COLUMN} = "{NO_PHONE_SERVICE}" is expected to '
        f'correspond to "{NO_PHONE_DEPENDENT_VALUE}"'
    )
    print(f"  in {MULTIPLE_LINES_COLUMN}.")
    print(
        f'  Rows with {PHONE_SERVICE_COLUMN} = "{NO_PHONE_SERVICE}": '
        f"{no_phone_count}"
    )
    print()

    if observed_counts.empty:
        print("No semantic consistency violations detected for this rule.")
        return

    print("Semantic consistency violations (require investigation):")
    print(f"{MULTIPLE_LINES_COLUMN}:")
    for observed_value, affected_rows in observed_counts.items():
        print(f"  Expected:      {NO_PHONE_DEPENDENT_VALUE}")
        print(f"  Observed:      {observed_value}")
        print(f"  Affected rows: {affected_rows}")


def print_semantic_consistency_checks(df: pd.DataFrame) -> None:
    """Print each semantic consistency rule in its own block."""
    print_no_internet_rule(df)
    print()
    print_no_phone_rule(df)


def main() -> None:
    """Load the dataset and print quality evidence for manual review."""
    # Load the dataset using the reusable module. The DataFrame is only read
    # from; no operation below writes back to it.
    customer_churn_df = load_dataset(str(DATASET_PATH))

    row_count, column_count = customer_churn_df.shape
    duplicate_rows = count_duplicate_rows(customer_churn_df)
    duplicate_ids = count_duplicate_ids(customer_churn_df, ID_COLUMN)
    missing_counts = count_missing_values(customer_churn_df)

    print("Data quality assessment")
    print("-" * 40)
    print(f"{'Rows:':<32}{row_count}")
    print(f"{'Columns:':<32}{column_count}")
    print(f"{'Fully duplicated rows:':<32}{duplicate_rows}")
    print(f"{f'Duplicated {ID_COLUMN} values:':<32}{duplicate_ids}")

    # --- Per-column missing values ---
    # The name column is sized to the longest column name so the table stays
    # aligned if the dataset schema changes.
    name_width = max(len(str(name)) for name in missing_counts.index)
    print()
    print("Missing values by column")
    print("-" * 40)
    print(f"{'Column':<{name_width}}{'Missing':>10}{'Percent':>10}")
    for column_name, missing_count in missing_counts.items():
        missing_percent = missing_count / row_count * 100
        print(
            f"{str(column_name):<{name_width}}"
            f"{missing_count:>10}"
            f"{missing_percent:>9.2f}%"
        )

    # --- Categorical value distributions ---
    print()
    print("Categorical value profile")
    print("-" * 40)
    print_categorical_profile(customer_churn_df, row_count)

    # --- Semantic consistency checks ---
    print("Semantic consistency checks")
    print("-" * 40)
    print_semantic_consistency_checks(customer_churn_df)


if __name__ == "__main__":
    main()
