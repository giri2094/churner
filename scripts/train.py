"""Executable entry point for the train-and-promote workflow.

This script runs the existing workflow once over the raw Telco customer
churn dataset. It loads the dataset through the reusable loading module,
states the policy the run is judged under, hands both to
``churner.workflow.train_and_promote.train_and_promote``, and reports what
that run decided.

It holds no modelling logic of its own. Preparing the frame, splitting it,
fitting the candidates, measuring them, applying the policy, and writing
the artifact all belong to the workflow and the packages beneath it; this
script resolves the two paths, supplies the policy, and prints the outcome.

An empty selection is reported as the complete run it is. When the policy
accepts nothing, the workflow writes no artifact and names none, so neither
does this report: a path printed for a run that saved nothing would claim
an artifact exists where none does.

Failures are left to surface. A missing dataset, an unreadable file, or an
error raised inside the workflow is allowed to propagate rather than being
caught and summarised, because a run that reports an outcome it did not
reach is worse than one that stops with a traceback.
"""

# --- Standard library imports ---
import sys
from pathlib import Path

# --- Make the reusable "churner" package importable ---
# The script lives in ``<project_root>/scripts``, so the project root is one
# level up. The source code lives under ``<project_root>/src``. Adding that
# directory to ``sys.path`` lets us import the existing modules without
# having the package installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))

# --- Import the existing, reusable workflow pieces ---
# The dataset is read through the same loading function the other scripts
# use, and the run itself is delegated to the existing workflow, so this
# entry point exercises the real logic rather than a copy of it.
from churner.data.load_dataset import load_dataset
from churner.evaluation.select import SelectionPolicy
from churner.workflow.train_and_promote import WorkflowResult, train_and_promote

# --- Resolve paths relative to the project root ---
# Building paths from ``PROJECT_ROOT`` avoids hardcoding an absolute path
# and keeps the script portable across machines. The artifact location
# matches the relative default the serving configuration falls back to, so
# a service started from the project root without ``CHURNER_MODEL_PATH``
# finds what this run wrote.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# --- The baseline the policy is constructed with ---
# ``SelectionPolicy`` requires a do-nothing baseline because it is a
# property of the data rather than of the policy, and the workflow measures
# the real figure on this run's test partition and replaces this one. Zero
# is the honest placeholder for a figure this script does not know: were it
# ever read as given, a baseline of zero would accept no candidate that
# costs anything rather than quietly promote one.
UNMEASURED_BASELINE_DO_NOTHING_COST = 0.0


def build_selection_policy() -> SelectionPolicy:
    """State the conditions this run's candidates have to meet.

    Every threshold, weight, and tolerance is left at the project default
    the policy already documents; only the baseline is supplied, since it
    has no default. Restating nothing else here keeps the assumptions in
    one place rather than in each run that applies them.
    """
    return SelectionPolicy(baseline_do_nothing_cost=UNMEASURED_BASELINE_DO_NOTHING_COST)


def print_decision_trace(result: WorkflowResult) -> None:
    """Print the stages the selection decision passed through.

    The trace is the workflow's audit record of which gates ran and how
    they ended, which is what accounts for an empty selection. It is
    printed as it was recorded; no entry is interpreted here.
    """
    print()
    print("Decision trace")
    print("-" * 40)
    for entry in result.decision_trace:
        print("  " + ", ".join(f"{key}={value}" for key, value in entry.items()))


def print_outcome(result: WorkflowResult) -> None:
    """Report what the run decided, and what it wrote.

    A run that promoted a model names it and the artifact it was written
    to. A run that promoted nothing says so and names no path, because the
    workflow wrote no file for it.
    """
    print("Train-and-promote outcome")
    print("-" * 40)
    print(f"{'Selection status:':<24}{result.selection_status}")

    if result.selected_model_id is None:
        print(
            "No candidate met the selection policy, so no model was promoted "
            "and no artifact was written."
        )
        print_decision_trace(result)
        return

    print(f"{'Selected model:':<24}{result.selected_model_id}")
    print(f"{'Artifact path:':<24}{result.artifact_path}")
    print_decision_trace(result)


def main() -> None:
    """Load the dataset, run the workflow once, and report the outcome."""
    # Load the dataset using the reusable module. The DataFrame is handed to
    # the workflow as loaded; nothing here writes back to it.
    customer_churn_df = load_dataset(str(DATASET_PATH))

    result = train_and_promote(
        customer_churn_df,
        ARTIFACT_PATH,
        build_selection_policy(),
    )

    print_outcome(result)


if __name__ == "__main__":
    main()
