"""Non-destructive data-quality inspection of the Telco customer churn dataset.

This script loads the raw dataset through the reusable
``churner.data.load_dataset.load_dataset`` function and reports observations
about its size, duplication, missing values, categorical value distributions,
explicit semantic consistency rules, the numeric readability of
``TotalCharges``, and the numerical quality of ``tenure``, ``SeniorCitizen``,
and ``MonthlyCharges``. It never modifies, cleans, or transforms the data, and it
draws no conclusion about whether the dataset is acceptable; it reports
evidence only, for manual review.
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

# Amount that the raw file stores as text but that is semantically numeric.
TOTAL_CHARGES_COLUMN = "TotalCharges"

# Text columns kept out of the categorical profile. ``customerID`` is a
# per-customer identifier, and ``TotalCharges`` is a continuous amount that
# pandas reads as text, so neither describes a category worth enumerating.
NON_CATEGORICAL_TEXT_COLUMNS = (ID_COLUMN, TOTAL_CHARGES_COLUMN)

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

# --- Numeric readability of TotalCharges ---
# Columns shown for each affected record, so a reviewer has enough context to
# interpret the value by hand instead of relying on the count alone.
TOTAL_CHARGES_CONTEXT_COLUMNS = [
    ID_COLUMN,
    "tenure",
    "MonthlyCharges",
    TOTAL_CHARGES_COLUMN,
    "Churn",
]
TOTAL_CHARGES_SAMPLE_SIZE = 10

# --- Numerical quality: tenure ---
# The data dictionary describes tenure as an integer count of months starting
# at 0, so 0 is the documented lower bound. No upper bound is documented, and
# none is invented here.
TENURE_COLUMN = "tenure"
TENURE_MINIMUM = 0
TENURE_CONTEXT_COLUMNS = [
    ID_COLUMN,
    TENURE_COLUMN,
    "MonthlyCharges",
    TOTAL_CHARGES_COLUMN,
    "Contract",
    INTERNET_SERVICE_COLUMN,
    "Churn",
]

# --- Numerical quality: SeniorCitizen ---
# The data dictionary defines this as a category encoded as 0/1. The integer
# storage is an encoding detail, so the expectation is set membership rather
# than a numeric range.
SENIOR_CITIZEN_COLUMN = "SeniorCitizen"
SENIOR_CITIZEN_EXPECTED_VALUES = (0, 1)
SENIOR_CITIZEN_CONTEXT_COLUMNS = [
    ID_COLUMN,
    SENIOR_CITIZEN_COLUMN,
    TENURE_COLUMN,
    "MonthlyCharges",
    TOTAL_CHARGES_COLUMN,
    "Churn",
]

# --- Numerical quality: MonthlyCharges ---
# No business minimum or maximum is documented for this amount, so it is
# profiled only; no threshold is defined and no value is called invalid.
MONTHLY_CHARGES_COLUMN = "MonthlyCharges"

# Number of affected records shown for the numerical checks below.
NUMERICAL_SAMPLE_SIZE = 10


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


def find_blank_values(series: pd.Series) -> pd.Series:
    """Flag entries that are present but empty or whitespace-only."""
    return series.notna() & (series.astype(str).str.strip() == "")


def find_total_charges_issues(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Group ``TotalCharges`` rows by how they respond to numeric conversion.

    ``TotalCharges`` is treated as semantically numeric even though the raw
    file stores it as text. Conversion runs on a derived Series with
    ``errors="coerce"``, so the original column is never written to. Returns
    boolean masks keyed by ``missing``, ``blank``, ``non_numeric``, and
    ``convertible``; the first three are mutually exclusive.
    """
    total_charges = df[TOTAL_CHARGES_COLUMN]
    numeric_attempt = pd.to_numeric(total_charges, errors="coerce")

    missing = total_charges.isna()
    blank = find_blank_values(total_charges)

    return {
        "missing": missing,
        "blank": blank,
        "non_numeric": numeric_attempt.isna() & ~missing & ~blank,
        "convertible": numeric_attempt.notna(),
    }


def print_total_charges_quality_check(df: pd.DataFrame) -> None:
    """Report whether ``TotalCharges`` values are numerically interpretable.

    Counts each category of value and shows contextual evidence for the
    records that did not convert. It does not judge whether those records are
    invalid, and it does not substitute a value for them.
    """
    issues = find_total_charges_issues(df)
    affected = issues["missing"] | issues["blank"] | issues["non_numeric"]
    affected_count = int(affected.sum())

    print(f"{'Stored dtype:':<38}{df[TOTAL_CHARGES_COLUMN].dtype}")
    print(f"{'Missing (pandas NA):':<38}{int(issues['missing'].sum())}")
    print(f"{'Blank or whitespace-only:':<38}{int(issues['blank'].sum())}")
    print(f"{'Non-blank, not numeric-convertible:':<38}{int(issues['non_numeric'].sum())}")
    print(f"{'Numeric-convertible:':<38}{int(issues['convertible'].sum())}")

    if affected_count == 0:
        print()
        print(f"Every {TOTAL_CHARGES_COLUMN} value is numerically interpretable.")
        return

    # --- tenure context for blank values ---
    # Reported as an observed association only; a blank value is not treated
    # as invalid, and no tenure value is treated as explaining it.
    blank = issues["blank"]
    if blank.any():
        print()
        print(f"tenure values where {TOTAL_CHARGES_COLUMN} is blank or whitespace-only:")
        for tenure_value, count in df.loc[blank, "tenure"].value_counts().items():
            print(f"  tenure = {tenure_value}: {count}")

    print()
    print(
        f"Records with affected {TOTAL_CHARGES_COLUMN} values ({affected_count} total, "
        f"showing up to {TOTAL_CHARGES_SAMPLE_SIZE}):"
    )
    sample = df.loc[affected, TOTAL_CHARGES_CONTEXT_COLUMNS].head(TOTAL_CHARGES_SAMPLE_SIZE)
    print(sample.to_string(index=False))


def print_affected_records(df: pd.DataFrame, affected: pd.Series, context_columns: list[str]) -> None:
    """Print contextual columns for the rows flagged by ``affected``.

    Shows at most ``NUMERICAL_SAMPLE_SIZE`` rows so a reviewer can interpret
    the flagged values by hand. The frame is only read from.
    """
    affected_count = int(affected.sum())
    print()
    print(
        f"Affected records ({affected_count} total, "
        f"showing up to {NUMERICAL_SAMPLE_SIZE}):"
    )
    sample = df.loc[affected, context_columns].head(NUMERICAL_SAMPLE_SIZE)
    print(sample.to_string(index=False))


def find_tenure_below_minimum(df: pd.DataFrame) -> pd.Series:
    """Flag rows whose ``tenure`` falls below the documented minimum.

    ``tenure`` counts months and the data dictionary describes it as starting
    at 0, so values below ``TENURE_MINIMUM`` are unexpected. No maximum is
    assumed, and the column is only read from.
    """
    return df[TENURE_COLUMN] < TENURE_MINIMUM


def print_tenure_check(df: pd.DataFrame) -> None:
    """Report ``tenure`` values below the documented minimum.

    Reports the observed minimum, how many records fall below the expected
    one, and the values involved. It does not correct, clip, or remove them.
    """
    below_minimum = find_tenure_below_minimum(df)
    affected_count = int(below_minimum.sum())

    print(f"Constrained check: {TENURE_COLUMN}")
    print("Expectation:")
    print(
        f"  {TENURE_COLUMN} is an integer count of months with a documented "
        f"minimum of {TENURE_MINIMUM}."
    )
    print("  No maximum is documented, so none is assumed here.")
    print()
    print(f"{'  Stored dtype:':<38}{df[TENURE_COLUMN].dtype}")
    print(f"{'  Observed minimum:':<38}{df[TENURE_COLUMN].min()}")
    print(f"{f'  Records with {TENURE_COLUMN} < {TENURE_MINIMUM}:':<38}{affected_count}")

    if affected_count == 0:
        print()
        print(f"No {TENURE_COLUMN} values below {TENURE_MINIMUM} detected.")
        return

    print()
    print(f"Unexpected {TENURE_COLUMN} values (require investigation):")
    for observed_value, count in df.loc[below_minimum, TENURE_COLUMN].value_counts().items():
        print(f"  Observed:      {observed_value}")
        print(f"  Affected rows: {count}")

    print_affected_records(df, below_minimum, TENURE_CONTEXT_COLUMNS)


def find_unexpected_senior_citizen_values(df: pd.DataFrame) -> pd.Series:
    """Flag rows whose ``SeniorCitizen`` value is outside the encoded domain.

    The data dictionary defines the column as a category encoded as 0/1, so
    membership in ``SENIOR_CITIZEN_EXPECTED_VALUES`` is the expectation rather
    than a numeric range. Missing values fall outside the domain and are
    flagged too. The column is only read from.
    """
    return ~df[SENIOR_CITIZEN_COLUMN].isin(SENIOR_CITIZEN_EXPECTED_VALUES)


def print_senior_citizen_check(df: pd.DataFrame) -> None:
    """Report ``SeniorCitizen`` values outside the documented 0/1 encoding.

    Lists every observed value with its count and shows contextual evidence
    for any value outside the domain. It does not re-encode or drop them.
    """
    unexpected = find_unexpected_senior_citizen_values(df)
    affected_count = int(unexpected.sum())
    expected_values = ", ".join(str(value) for value in SENIOR_CITIZEN_EXPECTED_VALUES)

    print(f"Constrained check: {SENIOR_CITIZEN_COLUMN}")
    print("Expectation:")
    print(
        f"  {SENIOR_CITIZEN_COLUMN} is documented as a category encoded as "
        f"{expected_values},"
    )
    print("  so it is checked against that domain rather than a numeric range.")
    print()
    print(f"{'  Stored dtype:':<38}{df[SENIOR_CITIZEN_COLUMN].dtype}")
    print(f"{'  Expected domain:':<38}{{{expected_values}}}")
    print(f"{'  Records outside the domain:':<38}{affected_count}")

    print()
    print("Observed values:")
    observed_counts = df[SENIOR_CITIZEN_COLUMN].value_counts(dropna=False)
    for observed_value, count in observed_counts.items():
        print(f"  {observed_value}: {count}")

    if affected_count == 0:
        print()
        print(
            f"No {SENIOR_CITIZEN_COLUMN} values outside {{{expected_values}}} detected."
        )
        return

    print()
    print(f"Unexpected {SENIOR_CITIZEN_COLUMN} values (require investigation):")
    unexpected_counts = df.loc[unexpected, SENIOR_CITIZEN_COLUMN].value_counts(dropna=False)
    for observed_value, count in unexpected_counts.items():
        print(f"  Observed:      {observed_value}")
        print(f"  Affected rows: {count}")

    print_affected_records(df, unexpected, SENIOR_CITIZEN_CONTEXT_COLUMNS)


def print_monthly_charges_profile(df: pd.DataFrame) -> None:
    """Describe the observed shape of ``MonthlyCharges`` without judging it.

    No business minimum or maximum is documented for this amount, so there is
    nothing to compare the data against. The observed range is reported as
    context only; no value is classified as valid or invalid.
    """
    monthly_charges = df[MONTHLY_CHARGES_COLUMN]

    print(f"Observational profile: {MONTHLY_CHARGES_COLUMN}")
    print("Expectation:")
    print("  No business minimum or maximum is documented for this amount,")
    print("  so no validity range is defined and no value is flagged.")
    print()
    print(f"{'  Stored dtype:':<38}{monthly_charges.dtype}")
    print(f"{'  Missing (pandas NA):':<38}{int(monthly_charges.isna().sum())}")
    print(f"{'  Observed minimum:':<38}{monthly_charges.min()}")
    print(f"{'  Observed maximum:':<38}{monthly_charges.max()}")


def print_numerical_quality_checks(df: pd.DataFrame) -> None:
    """Print each numerical check in its own block.

    The constrained checks compare the data against an expectation documented
    in the data dictionary; the observational profile has no such expectation
    to compare against and therefore flags nothing.
    """
    print_tenure_check(df)
    print()
    print_senior_citizen_check(df)
    print()
    print_monthly_charges_profile(df)


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

    # --- TotalCharges numeric readability ---
    print()
    print(f"{TOTAL_CHARGES_COLUMN} numeric readability")
    print("-" * 40)
    print_total_charges_quality_check(customer_churn_df)

    # --- Numerical quality checks ---
    print()
    print("Numerical quality checks")
    print("-" * 40)
    print_numerical_quality_checks(customer_churn_df)


if __name__ == "__main__":
    main()
