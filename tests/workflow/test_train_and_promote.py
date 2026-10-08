"""Unit tests for the train-and-promote workflow and its result contract.

These tests establish two contracts. ``WorkflowResult`` can be
constructed for a run that promoted a model and for a run that correctly
promoted nothing, it holds exactly the values it was given, and it cannot
be rewritten after the fact. ``train_and_promote`` calls the data,
training, evaluation, comparison, selection, and packaging functions in
order, keeps the fitted pipelines and their evidence under the same
identifiers, measures the do-nothing baseline on the test partition and
gives selection a policy carrying it without writing to the caller's, and
saves the named pipeline only when one is named.

The orchestration tests replace every function the workflow delegates to
with a mock, patched where the workflow looks it up. No frame is prepared,
no pipeline is fitted, no metric is computed, no policy is applied, and no
file is written: those modules already have their own tests, and a failure
here points at the workflow wiring them together wrongly rather than at
any one of them misbehaving. The stand-ins are shaped like the real
results -- real ``EvaluationResult``, ``CostResult``, ``ModelComparison``,
``SelectionPolicy``, and ``SelectionResult`` objects -- so the workflow
exercises the same attribute reads it would on a real run.

Nothing here tests sklearn, preprocessing, or the arithmetic of any
metric or cost.
"""

from dataclasses import FrozenInstanceError
from unittest.mock import DEFAULT, Mock, patch, sentinel

import pytest

from churner.evaluation.compare import ModelComparison, ModelEvidence
from churner.evaluation.cost import CostResult
from churner.evaluation.evaluate import EvaluationResult
from churner.evaluation.select import SelectionPolicy, SelectionResult
from churner.workflow.train_and_promote import WorkflowResult, train_and_promote

# --- Result values ---
# A promoted run and an empty run, as the future workflow would report
# them. The trace entries are small fixed dictionaries in the same shape
# selection produces; their contents are not interpreted by these tests.
SELECTED_MODEL_ID = "logistic_regression"
SELECTED_STATUS = "SELECTED"
ARTIFACT_PATH = "models/churn_model.joblib"
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


def make_selected_result() -> WorkflowResult:
    """Build the result of a run that promoted a model."""
    return WorkflowResult(
        selected_model_id=SELECTED_MODEL_ID,
        selection_status=SELECTED_STATUS,
        artifact_path=ARTIFACT_PATH,
        decision_trace=SELECTED_TRACE,
    )


def make_no_selection_result() -> WorkflowResult:
    """Build the result of a run that promoted nothing."""
    return WorkflowResult(
        selected_model_id=None,
        selection_status=NO_ACCEPTABLE_MODEL_STATUS,
        artifact_path=None,
        decision_trace=NO_SELECTION_TRACE,
    )


# --- Construction ---


def test_a_selected_result_can_be_constructed():
    """A run that promoted a model is representable."""
    result = make_selected_result()

    assert isinstance(result, WorkflowResult)
    assert result.decision_trace


def test_a_no_selection_result_can_be_constructed():
    """A run that promoted nothing is representable.

    An empty outcome is a real outcome, so the record has to be able to
    say so with no model and no artifact.
    """
    result = make_no_selection_result()

    assert isinstance(result, WorkflowResult)


# --- Field values ---


def test_selected_result_holds_the_supplied_values():
    """Each field reads back as the value it was constructed with."""
    result = make_selected_result()

    assert result.selected_model_id == SELECTED_MODEL_ID
    assert result.selection_status == SELECTED_STATUS
    assert result.artifact_path == ARTIFACT_PATH
    assert result.decision_trace == SELECTED_TRACE


def test_no_selection_result_holds_the_supplied_values():
    """A no-selection result names no model and no artifact."""
    result = make_no_selection_result()

    assert result.selected_model_id is None
    assert result.selection_status == NO_ACCEPTABLE_MODEL_STATUS
    assert result.artifact_path is None
    assert result.decision_trace == NO_SELECTION_TRACE


# --- Immutability ---


def test_result_is_frozen():
    """A recorded outcome cannot be rewritten after the fact."""
    result = make_selected_result()

    with pytest.raises(FrozenInstanceError):
        result.selected_model_id = "decision_tree"


def test_every_field_is_frozen():
    """No field on the result, not only the identifier, accepts assignment."""
    result = make_no_selection_result()

    with pytest.raises(FrozenInstanceError):
        result.selection_status = SELECTED_STATUS

    with pytest.raises(FrozenInstanceError):
        result.artifact_path = ARTIFACT_PATH

    with pytest.raises(FrozenInstanceError):
        result.decision_trace = SELECTED_TRACE


# --- Orchestration stand-ins ---
# The workflow is patched where it looks its collaborators up, so none
# of the real modules is entered.
WORKFLOW_MODULE = "churner.workflow.train_and_promote"

LOGISTIC_REGRESSION_ID = "logistic_regression"
DECISION_TREE_ID = "decision_tree"

# Evidence as the real evaluation and cost functions would shape it. The
# figures are labels for telling one candidate's evidence from the
# other's, not a ranking; selection is mocked, so nothing reads them.
LOGISTIC_EVALUATION = EvaluationResult(
    accuracy=0.80, precision=0.55, recall=0.60, f1=0.57, roc_auc=0.82
)
TREE_EVALUATION = EvaluationResult(
    accuracy=0.77, precision=0.50, recall=0.62, f1=0.55, roc_auc=0.74
)
LOGISTIC_COST = CostResult(
    false_positive_count=120, false_negative_count=150, total_cost=870.0
)
TREE_COST = CostResult(
    false_positive_count=232, false_negative_count=142, total_cost=942.0
)
COMPARISON = ModelComparison(
    candidates={
        LOGISTIC_REGRESSION_ID: ModelEvidence(LOGISTIC_EVALUATION, LOGISTIC_COST),
        DECISION_TREE_ID: ModelEvidence(TREE_EVALUATION, TREE_COST),
    }
)

# The caller's policy, and the baseline the mocked calculation reports
# for this run's test partition. The two figures differ on purpose: the
# one selection is given has to be the measured one, and the caller's
# object has to still read 500.0 afterwards.
CALLER_BASELINE_COST = 500.0
POLICY = SelectionPolicy(baseline_do_nothing_cost=CALLER_BASELINE_COST)
BASELINE_DO_NOTHING_COST = 600.0

SELECTED_SELECTION = SelectionResult(
    selected_model_id=SELECTED_MODEL_ID,
    selection_status=SELECTED_STATUS,
    decision_trace=SELECTED_TRACE,
)
NO_SELECTION = SelectionResult(
    selected_model_id=None,
    selection_status=NO_ACCEPTABLE_MODEL_STATUS,
    decision_trace=NO_SELECTION_TRACE,
)


@pytest.fixture
def collaborators():
    """Replace every function the workflow delegates to with a mock.

    Data flows through as sentinels, so a test can see which partition
    reached which stage without any frame existing. The two fitted
    pipelines are mocks whose ``predict`` returns a distinct sentinel, and
    evaluation and cost are keyed on the pipeline and the predictions they
    are handed, so the evidence that comes back says which model it was
    measured on. The do-nothing baseline reports a figure of its own,
    distinct from the one the caller's policy carries. Selection defaults
    to naming the logistic model; a test that needs the empty outcome
    overrides ``select``.
    """
    with patch.multiple(
        WORKFLOW_MODULE,
        prepare_modeling_data=DEFAULT,
        split_modeling_data=DEFAULT,
        train_logistic=DEFAULT,
        train_tree=DEFAULT,
        evaluate_model=DEFAULT,
        calculate_business_cost=DEFAULT,
        calculate_do_nothing_cost=DEFAULT,
        compare_models=DEFAULT,
        select=DEFAULT,
        save_model=DEFAULT,
    ) as mocks:
        fitted_logistic = Mock(name="fitted_logistic_pipeline")
        fitted_logistic.predict.return_value = sentinel.logistic_predictions
        fitted_tree = Mock(name="fitted_tree_pipeline")
        fitted_tree.predict.return_value = sentinel.tree_predictions

        def evaluate_by_model(fitted_model, X_test, y_test):
            return {
                fitted_logistic: LOGISTIC_EVALUATION,
                fitted_tree: TREE_EVALUATION,
            }[fitted_model]

        def cost_by_predictions(
            y_true, y_pred, false_positive_cost, false_negative_cost, positive_label
        ):
            return {
                sentinel.logistic_predictions: LOGISTIC_COST,
                sentinel.tree_predictions: TREE_COST,
            }[y_pred]

        mocks["prepare_modeling_data"].return_value = (
            sentinel.features,
            sentinel.target,
        )
        mocks["split_modeling_data"].return_value = (
            sentinel.X_train,
            sentinel.X_test,
            sentinel.y_train,
            sentinel.y_test,
        )
        mocks["train_logistic"].return_value = fitted_logistic
        mocks["train_tree"].return_value = fitted_tree
        mocks["evaluate_model"].side_effect = evaluate_by_model
        mocks["calculate_business_cost"].side_effect = cost_by_predictions
        mocks["calculate_do_nothing_cost"].return_value = BASELINE_DO_NOTHING_COST
        mocks["compare_models"].return_value = COMPARISON
        mocks["select"].return_value = SELECTED_SELECTION

        yield mocks


def policy_given_to_select(collaborators) -> SelectionPolicy:
    """Return the policy ``select`` was called with on the run just made."""
    _, policy = collaborators["select"].call_args.args
    return policy


# --- Scenario 1: a model is selected ---


def test_selected_run_prepares_splits_and_trains_both_candidates(collaborators):
    """The frame is prepared, split once, and both baselines fit on the split.

    Both trainers receive the training partition and nothing else, so
    the test partition cannot enter either fit.
    """
    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    collaborators["prepare_modeling_data"].assert_called_once_with(sentinel.dataframe)
    collaborators["split_modeling_data"].assert_called_once_with(
        sentinel.features, sentinel.target
    )
    collaborators["train_logistic"].assert_called_once_with(
        sentinel.X_train, sentinel.y_train
    )
    collaborators["train_tree"].assert_called_once_with(
        sentinel.X_train, sentinel.y_train
    )


def test_selected_run_measures_each_candidate_on_the_test_partition(collaborators):
    """Each fitted pipeline is evaluated once and costed once, on test data.

    The cost is charged at the weights and positive label the supplied
    policy states, so the figure selection compares against its baseline
    was produced under the policy's own assumptions.
    """
    fitted_logistic = collaborators["train_logistic"].return_value
    fitted_tree = collaborators["train_tree"].return_value

    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    evaluate_model = collaborators["evaluate_model"]
    assert evaluate_model.call_count == 2
    evaluate_model.assert_any_call(fitted_logistic, sentinel.X_test, sentinel.y_test)
    evaluate_model.assert_any_call(fitted_tree, sentinel.X_test, sentinel.y_test)

    calculate_business_cost = collaborators["calculate_business_cost"]
    assert calculate_business_cost.call_count == 2
    for predictions in (sentinel.logistic_predictions, sentinel.tree_predictions):
        calculate_business_cost.assert_any_call(
            y_true=sentinel.y_test,
            y_pred=predictions,
            false_positive_cost=POLICY.false_positive_cost,
            false_negative_cost=POLICY.false_negative_cost,
            positive_label=POLICY.positive_label,
        )


def test_selected_run_compares_selects_and_saves_the_named_pipeline(collaborators):
    """The evidence is compared, the policy applied, and the winner saved.

    Selection names the logistic model, so the pipeline handed to
    ``save_model`` has to be the one ``train_logistic`` returned, written
    to the path the caller supplied. The tree is not saved.
    """
    fitted_logistic = collaborators["train_logistic"].return_value

    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    compare_models = collaborators["compare_models"]
    compare_models.assert_called_once()
    (evidence,) = compare_models.call_args.args
    assert set(evidence) == {LOGISTIC_REGRESSION_ID, DECISION_TREE_ID}

    collaborators["select"].assert_called_once()
    comparison_received, _ = collaborators["select"].call_args.args
    assert comparison_received is COMPARISON
    collaborators["save_model"].assert_called_once_with(fitted_logistic, ARTIFACT_PATH)


def test_selected_run_reports_the_selection(collaborators):
    """The result carries the selection through and names the artifact."""
    result = train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    assert result == WorkflowResult(
        selected_model_id=SELECTED_MODEL_ID,
        selection_status=SELECTED_STATUS,
        artifact_path=ARTIFACT_PATH,
        decision_trace=SELECTED_TRACE,
    )


# --- Scenario 2: no acceptable model ---


def test_no_selection_still_trains_and_measures_both_candidates(collaborators):
    """An empty outcome is reached by running the whole workflow.

    Both baselines are fitted and measured, the evidence compared, and the
    policy applied; only the last step changes its answer.
    """
    collaborators["select"].return_value = NO_SELECTION

    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    collaborators["train_logistic"].assert_called_once()
    collaborators["train_tree"].assert_called_once()
    assert collaborators["evaluate_model"].call_count == 2
    assert collaborators["calculate_business_cost"].call_count == 2
    collaborators["compare_models"].assert_called_once()
    collaborators["select"].assert_called_once()
    comparison_received, _ = collaborators["select"].call_args.args
    assert comparison_received is COMPARISON


def test_no_selection_saves_nothing_and_reports_no_artifact(collaborators):
    """When the policy accepts nothing, no artifact is written or named.

    The status and trace come from selection as they were; the identifier
    and the path are ``None`` rather than a least-bad model or an unused
    path.
    """
    collaborators["select"].return_value = NO_SELECTION

    result = train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    collaborators["save_model"].assert_not_called()
    assert result == WorkflowResult(
        selected_model_id=None,
        selection_status=NO_ACCEPTABLE_MODEL_STATUS,
        artifact_path=None,
        decision_trace=NO_SELECTION_TRACE,
    )


# --- Scenario 3: model and evidence identifiers stay aligned ---


def test_evidence_is_filed_under_the_fitted_model_identifiers(collaborators):
    """The evidence handed to comparison is keyed exactly as the pipelines are.

    Comparison and selection see evidence only, and the workflow goes back
    to the fitted pipelines with whatever identifier selection returns. If
    the two mappings used different keys, or the evidence for one model
    were filed under the other's name, the wrong pipeline would be saved.
    So the keys have to match the fitted-model identifiers exactly, and
    the evidence under each key has to be the evaluation and cost measured
    on that key's pipeline.
    """
    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    (evidence,) = collaborators["compare_models"].call_args.args

    assert list(evidence) == [LOGISTIC_REGRESSION_ID, DECISION_TREE_ID]
    assert evidence[LOGISTIC_REGRESSION_ID] == ModelEvidence(
        LOGISTIC_EVALUATION, LOGISTIC_COST
    )
    assert evidence[DECISION_TREE_ID] == ModelEvidence(TREE_EVALUATION, TREE_COST)


# --- Scenario 4: the baseline is measured on the test partition ---


def test_the_baseline_is_calculated_on_the_held_out_target(collaborators):
    """The do-nothing cost is measured on the same partition the models are.

    A baseline counted on any other rows would be compared against costs
    it does not describe, so the target handed over has to be ``y_test``,
    charged at the weights the caller's policy states.
    """
    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    collaborators["calculate_do_nothing_cost"].assert_called_once_with(
        y_true=sentinel.y_test,
        false_negative_cost=POLICY.false_negative_cost,
        positive_label=POLICY.positive_label,
    )


def test_selection_receives_the_measured_baseline_and_nothing_else_changed(
    collaborators,
):
    """``select`` is given a policy carrying the measured baseline.

    The caller's object cannot be the one passed on, since its baseline
    is the figure this run replaced. Every other condition and weight is
    the caller's, so only that one field may differ.
    """
    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    effective_policy = policy_given_to_select(collaborators)

    assert effective_policy is not POLICY
    assert effective_policy.baseline_do_nothing_cost == BASELINE_DO_NOTHING_COST
    assert effective_policy.minimum_recall == POLICY.minimum_recall
    assert effective_policy.minimum_roc_auc == POLICY.minimum_roc_auc
    assert effective_policy.cost_tolerance == POLICY.cost_tolerance
    assert effective_policy.false_positive_cost == POLICY.false_positive_cost
    assert effective_policy.false_negative_cost == POLICY.false_negative_cost
    assert effective_policy.positive_label == POLICY.positive_label


def test_the_callers_policy_is_not_written_to(collaborators):
    """The policy handed in still reads as it did after the run.

    A run tailors the baseline to its own test partition, so it has to
    build a policy for itself rather than overwrite configuration the
    caller may reuse for the next run.
    """
    train_and_promote(sentinel.dataframe, ARTIFACT_PATH, POLICY)

    assert POLICY.baseline_do_nothing_cost == CALLER_BASELINE_COST


# --- Scenario 5: the artifact path is represented correctly ---


def test_artifact_path_is_passed_through_and_reported_as_a_string(
    collaborators, tmp_path
):
    """A ``Path`` is handed to ``save_model`` as given and reported as ``str``.

    ``save_model`` accepts either form, so the workflow passes the
    caller's object through unchanged. ``WorkflowResult.artifact_path``
    is typed ``str | None``, so the same location is reported as its
    string form. No file is written: ``save_model`` is a mock.
    """
    fitted_logistic = collaborators["train_logistic"].return_value
    artifact_path = tmp_path / "churn_model.joblib"

    result = train_and_promote(sentinel.dataframe, artifact_path, POLICY)

    collaborators["save_model"].assert_called_once_with(fitted_logistic, artifact_path)
    assert result.artifact_path == str(artifact_path)
    assert not artifact_path.exists()
