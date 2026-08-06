"""Temporary script to verify the reusable data-loading module.

This script only checks that ``churner.data.load_dataset.load_dataset`` can
read the Telco customer churn dataset correctly. It performs no preprocessing,
validation, or modification of the data.
"""

# --- Standard library imports ---
import sys
from pathlib import Path

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
# directly, so this script validates the real loading logic.
from churner.data.load_dataset import load_dataset

# --- Resolve the dataset path relative to the project root ---
# Building the path from ``PROJECT_ROOT`` avoids hardcoding an absolute path
# and keeps the script portable across machines.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main() -> None:
    """Load the dataset and print basic information for verification."""
    # Load the dataset using the reusable module under test.
    customer_churn_df = load_dataset(str(DATASET_PATH))

    # --- Show the first five rows ---
    print("First five rows")
    print("-" * 40)
    print(customer_churn_df.head())
    print()

    # --- Show the dataset shape (rows, columns) ---
    print("Dataset shape (rows, columns)")
    print("-" * 40)
    print(customer_churn_df.shape)
    print()

    # --- Show the column names ---
    print("Column names")
    print("-" * 40)
    print(list(customer_churn_df.columns))

    print()
    print("Dataset information")
    print("-" * 40)
    customer_churn_df.info()

    print()
    print("Descriptive statistics")
    print("-" * 40)
    print(customer_churn_df.describe(include="all"))

    print()
    print("Missing values")
    print("-" * 40)
    print(customer_churn_df.isnull().sum())

if __name__ == "__main__":
    main()
    
