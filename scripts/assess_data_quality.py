"""Non-destructive data-quality inspection of the Telco customer churn dataset.

This script loads the raw dataset through the reusable
``churner.data.load_dataset.load_dataset`` function and reports observations
about its size and duplication. It never modifies, cleans, or transforms the
data, and it draws no conclusion about whether the dataset is acceptable; it
reports evidence only, for manual review.
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


def main() -> None:
    """Load the dataset and print duplication evidence for manual review."""
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


if __name__ == "__main__":
    main()
