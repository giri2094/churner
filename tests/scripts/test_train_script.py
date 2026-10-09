"""Unit tests for the train-and-promote entry point script.

These tests establish what the entry point is responsible for: it resolves
both paths against the project root, loads the dataset through the reusable
loading function, builds a policy from the project's own defaults, hands
the loaded frame, the artifact path, and that policy to the workflow, and
reports the outcome it is given.

Reporting is the part worth pinning down. A run that promoted a model has
to name the model, the status, and the path it was written to; a run that
promoted nothing has to say so without printing a path, because no file was
written for it and a path on screen would read as one that was.

Both the loader and the workflow are replaced with mocks, patched where the
script looks them up. No CSV is read, no pipeline is fitted, no policy is
applied, and no artifact is written: those modules have their own tests, and
a failure here points at this script wiring them together wrongly rather
than at any one of them misbehaving. The frame flows through as a sentinel,
so what reached the workflow is visible without a frame existing.

Nothing here tests the loading function, the selection algorithm, the
workflow's orchestration, or what the models predict.
"""

import sys
from pathlib import Path
from unittest.mock import DEFAULT, patch, sentinel

import pytest

from churner.evaluation.select import (
    COST_TOLERANCE,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    MINIMUM_RECALL,
    MINIMUM_ROC_AUC,
    POSITIVE_LABEL,
    SelectionPolicy,
)
from churner.workflow.train_and_promote import WorkflowResult

# --- Make the script importable ---
# The scripts are standalone entry points rather than part of the installed
# package, so the directory holding them is added to ``sys.path`` the same
# way ``conftest.py`` adds ``src``. The script is imported under test; its
# module-level code only resolves paths and imports, so importing it reads
# no file and runs no workflow.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train

# --- Stand-in outcomes ---
# The two results the workflow can report, shaped as the real ones are. The
# trace entries are small fixed dictionaries in the shape selection
# produces; the script prints them without interpreting them.
#
# The artifact path deliberately is not the one the script resolves, so a
# path that appears in the report can only have been read from the result
# rather than printed from the script's own constant. It is never opened.
SELECTED_MODEL_ID = "logistic_regression"
SELECTED_STATUS = "SELECTED"
SELECTED_ARTIFACT_PATH = "/tmp/churner/models/churn_model.joblib"
SELECTED_TRACE = [
    {"stage": "technical_gate", "status": "passed"},
    {"stage": "business_cost_gate", "status": "passed"},
    {"stage": "cost_tolerance", "status": "within"},
    {"stage": "selection", "criterion": "recall"},
]

NO_ACCEPTABLE_MODEL_STATUS = "NO_ACCEPTABLE_MODEL"
NO_SELECTION_TRACE = [
    {"stage": "technical_gate", "status": "no_acceptable_candidates"},
]

SELECTED_RESULT = WorkflowResult(
    selected_model_id=SELECTED_MODEL_ID,
    selection_status=SELECTED_STATUS,
    artifact_path=SELECTED_ARTIFACT_PATH,
    decision_trace=SELECTED_TRACE,
)
NO_SELECTION_RESULT = WorkflowResult(
    selected_model_id=None,
    selection_status=NO_ACCEPTABLE_MODEL_STATUS,
    artifact_path=None,
    decision_trace=NO_SELECTION_TRACE,
)

SCRIPT_MODULE = "train"


@pytest.fixture
def collaborators():
    """Replace the loader and the workflow with mocks.

    Loading reports a sentinel rather than a frame, so the object the
    workflow was handed can be identified as the loaded one without any
    CSV being read. The workflow defaults to reporting a promoted model; a
    test that needs the empty outcome overrides its return value.
    """
    with patch.multiple(
        SCRIPT_MODULE,
        load_dataset=DEFAULT,
        train_and_promote=DEFAULT,
    ) as mocks:
        mocks["load_dataset"].return_value = sentinel.dataframe
        mocks["train_and_promote"].return_value = SELECTED_RESULT

        yield mocks


def workflow_call_arguments(collaborators) -> tuple[object, Path, SelectionPolicy]:
    """Return the frame, path, and policy the workflow was called with."""
    dataframe, artifact_path, policy = collaborators["train_and_promote"].call_args.args
    return dataframe, artifact_path, policy


# --- Paths are resolved against the project root ---


def test_both_paths_are_resolved_against_the_project_root():
    """Neither path depends on the directory the script is run from.

    A relative path would resolve against the working directory, so the
    same run would read a different dataset, or write an artifact
    somewhere else, depending on where it was started.
    """
    assert train.PROJECT_ROOT == PROJECT_ROOT
    assert train.DATASET_PATH.is_absolute()
    assert train.ARTIFACT_PATH.is_absolute()
    assert train.DATASET_PATH.parent == PROJECT_ROOT / "data" / "raw"
    assert train.ARTIFACT_PATH.parent == PROJECT_ROOT / "models"


def test_the_artifact_is_named_rather_than_the_directory_it_sits_in():
    """``save_model`` is given a filename, so the path has to carry one."""
    assert train.ARTIFACT_PATH.suffix == ".joblib"


# --- The policy is the project's own ---


def test_the_policy_is_built_from_the_documented_defaults():
    """Every condition is left at the default the policy module states.

    The entry point is not the place to restate thresholds or cost
    weights: a figure repeated here could drift from the one the project
    documents while still looking deliberate.
    """
    policy = train.build_selection_policy()

    assert isinstance(policy, SelectionPolicy)
    assert policy.minimum_recall == MINIMUM_RECALL
    assert policy.minimum_roc_auc == MINIMUM_ROC_AUC
    assert policy.cost_tolerance == COST_TOLERANCE
    assert policy.false_positive_cost == FALSE_POSITIVE_COST
    assert policy.false_negative_cost == FALSE_NEGATIVE_COST
    assert policy.positive_label == POSITIVE_LABEL


def test_the_supplied_baseline_promotes_nothing_if_it_were_ever_read():
    """The placeholder baseline accepts no candidate that costs anything.

    The workflow measures the real figure on its own test partition and
    replaces this one, so the value here is never applied. Were that to
    change, a baseline of zero fails the business-cost gate rather than
    promoting a model against a baseline nobody measured.
    """
    assert train.build_selection_policy().baseline_do_nothing_cost == 0.0


# --- Scenario 1: a model is selected ---


def test_the_dataset_is_loaded_through_the_reusable_function(collaborators):
    """The configured dataset is read once, through ``load_dataset``.

    The path is handed over as a string, which is the form the loading
    function documents.
    """
    train.main()

    collaborators["load_dataset"].assert_called_once_with(str(train.DATASET_PATH))


def test_the_workflow_runs_once_on_the_loaded_frame(collaborators):
    """The workflow receives the loaded frame, the path, and the policy.

    The frame has to be the object loading returned rather than one this
    script built, and the path the one resolved against the project root.
    """
    train.main()

    collaborators["train_and_promote"].assert_called_once()
    dataframe, artifact_path, policy = workflow_call_arguments(collaborators)

    assert dataframe is sentinel.dataframe
    assert artifact_path == train.ARTIFACT_PATH
    assert policy == train.build_selection_policy()


def test_a_selected_run_reports_the_model_the_status_and_the_artifact(
    collaborators, capsys
):
    """A promoted model, its status, and where it was written are reported."""
    train.main()

    output = capsys.readouterr().out

    assert SELECTED_MODEL_ID in output
    assert SELECTED_STATUS in output
    assert SELECTED_ARTIFACT_PATH in output


def test_a_selected_run_reports_the_decision_trace(collaborators, capsys):
    """The stages the decision passed through are reported as recorded."""
    train.main()

    output = capsys.readouterr().out

    for entry in SELECTED_TRACE:
        for value in entry.values():
            assert str(value) in output


# --- Scenario 2: no acceptable model ---


def test_a_run_that_promoted_nothing_reports_that_outcome(collaborators, capsys):
    """An empty selection is reported as the complete run it is."""
    collaborators["train_and_promote"].return_value = NO_SELECTION_RESULT

    train.main()

    output = capsys.readouterr().out

    assert NO_ACCEPTABLE_MODEL_STATUS in output
    assert "no artifact was written" in output


def test_a_run_that_promoted_nothing_names_no_artifact(collaborators, capsys):
    """No path and no model identifier is printed for an empty selection.

    The workflow wrote no file, so a path on screen would describe an
    artifact that does not exist, and ``None`` printed as a path would read
    as one too.
    """
    collaborators["train_and_promote"].return_value = NO_SELECTION_RESULT

    train.main()

    output = capsys.readouterr().out

    assert str(train.ARTIFACT_PATH) not in output
    assert SELECTED_MODEL_ID not in output
    assert "None" not in output


def test_a_run_that_promoted_nothing_still_reports_the_trace(collaborators, capsys):
    """The gate that ended the decision is reported, since it explains it."""
    collaborators["train_and_promote"].return_value = NO_SELECTION_RESULT

    train.main()

    output = capsys.readouterr().out

    assert "technical_gate" in output
    assert "no_acceptable_candidates" in output


# --- Scenario 3: failures stay visible ---


def test_a_loading_failure_is_not_swallowed(collaborators):
    """A dataset that cannot be read stops the run.

    Reporting an outcome after loading failed would describe a run that
    never happened, so the error is left to surface.
    """
    collaborators["load_dataset"].side_effect = FileNotFoundError(train.DATASET_PATH)

    with pytest.raises(FileNotFoundError):
        train.main()

    collaborators["train_and_promote"].assert_not_called()


def test_a_workflow_failure_is_not_swallowed(collaborators, capsys):
    """An error raised inside the workflow stops the run unreported."""
    collaborators["train_and_promote"].side_effect = ValueError("training failed")

    with pytest.raises(ValueError):
        train.main()

    assert capsys.readouterr().out == ""
