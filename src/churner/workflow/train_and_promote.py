"""Orchestration of the train-and-promote workflow, and its result contract.

This module is the step that runs the whole workflow end to end: it
prepares the data, splits it, fits both baseline candidates, measures
each, compares the evidence, selects at most one under the caller's
policy, and persists the chosen pipeline as an artifact. Each of those
pieces lives in its own package -- ``data``, ``training``, ``evaluation``,
``packaging`` -- and this module's only job is to call them in order and
report what happened.

Two pieces of state are carried side by side on purpose. The fitted
pipelines are kept in one mapping and the evidence gathered about them in
another, both under the same identifiers. Comparison and selection see
only the evidence, so they cannot re-measure or refit anything; once a
model is named, the workflow goes back to the first mapping to find the
object to save. The identifier is the only thing that joins the two.

An empty selection is a complete run, not a failure. When the policy
accepts nothing, no artifact is written and the result says so, with the
trace that explains which gate ended the decision.

The business assumptions come from the caller's policy, but the baseline
those assumptions are measured against does not: what doing nothing would
cost depends on how many customers actually churned, so it belongs to the
partition the candidates are judged on. The workflow derives it from the
same ``y_test`` the costs are charged on and passes selection an effective
policy carrying that figure. The caller's policy is left as it was.

Nothing here reads a file, fits a pipeline, computes a metric, charges a
cost, or applies a threshold of its own. Every one of those is delegated
to the module that owns it.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from churner.data.prepare_modeling_data import prepare_modeling_data
from churner.evaluation.compare import ModelEvidence, compare_models
from churner.evaluation.cost import calculate_business_cost, calculate_do_nothing_cost
from churner.evaluation.evaluate import evaluate_model
from churner.evaluation.select import SELECTED, SelectionPolicy, select
from churner.packaging.model import save_model
from churner.training.split_modeling_data import split_modeling_data
from churner.training.train import train_logistic, train_tree

# --- Candidate identifiers ---
# The keys both state mappings are kept under, and the names a selection
# comes back with. They are stated once so the fitted pipeline and the
# evidence for one model cannot be filed under different names.
LOGISTIC_REGRESSION_ID = "logistic_regression"
DECISION_TREE_ID = "decision_tree"


@dataclass(frozen=True)
class WorkflowResult:
    """The outcome of one train-and-promote run.

    ``selected_model_id`` is the key the chosen candidate was stored
    under, or ``None`` when no candidate met the policy. The status that
    accompanies it is carried over from selection as ``selection_status``,
    so a reader can tell a run that promoted a model from one that
    correctly promoted nothing.

    ``artifact_path`` is where the selected pipeline was written. It is
    ``None`` when no model is selected, because a run that selects
    nothing produces no artifact: there is nothing to persist, and no
    path is invented for it.

    ``decision_trace`` records which selection stages ran and how they
    ended. It is an audit record of the decision, not an explanation of
    it. No metric, cost, or fitted model is copied in here; those stay on
    the objects the workflow produced along the way.
    """

    selected_model_id: str | None
    selection_status: str
    artifact_path: str | None
    decision_trace: list[dict[str, object]]


def train_and_promote(
    dataframe: pd.DataFrame,
    artifact_path: str | Path,
    selection_policy: SelectionPolicy,
) -> WorkflowResult:
    """Train both baselines, select at most one, and persist it.

    The run proceeds in order: the frame is prepared and split once, so
    both candidates are fitted and measured on the same partition; each
    fitted pipeline is scored for its technical metrics and charged for
    its errors at the weights the policy states; the evidence is gathered
    into one comparison and the policy applied to it. If a model is
    named, the matching fitted pipeline is written to ``artifact_path``
    and that path is reported. If none is, nothing is written.

    The business cost is charged at the policy's own weights and positive
    label, so the figure the selector compares against its baseline was
    produced under the same assumptions the policy records. The baseline
    itself is calculated here from the held-out target, and selection is
    given a copy of the policy carrying it. Nothing else on the policy
    changes, and the caller's object is not written to.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Already-loaded customer records holding at least the columns
        ``prepare_modeling_data`` requires. Only read from.
    artifact_path : str | Path
        Where to write the selected pipeline, filename included. Used
        only when a model is selected.
    selection_policy : SelectionPolicy
        The conditions a candidate has to meet, and the cost weights its
        errors are charged at. Its ``baseline_do_nothing_cost`` is
        replaced by the figure measured on this run's test partition.
        Only read from.

    Returns
    -------
    WorkflowResult
        The selected identifier, ``"SELECTED"``, the path written to, and
        the trace; or ``None``, ``"NO_ACCEPTABLE_MODEL"``, ``None``, and
        the trace of the stage that ended the decision.
    """
    features, target = prepare_modeling_data(dataframe)
    X_train, X_test, y_train, y_test = split_modeling_data(features, target)

    baseline_do_nothing_cost = calculate_do_nothing_cost(
        y_true=y_test,
        false_negative_cost=selection_policy.false_negative_cost,
        positive_label=selection_policy.positive_label,
    )
    effective_policy = replace(
        selection_policy,
        baseline_do_nothing_cost=baseline_do_nothing_cost,
    )

    fitted_models: dict[str, Pipeline] = {
        LOGISTIC_REGRESSION_ID: train_logistic(X_train, y_train),
        DECISION_TREE_ID: train_tree(X_train, y_train),
    }

    model_evidence: dict[str, ModelEvidence] = {}
    for model_id, fitted_model in fitted_models.items():
        evaluation = evaluate_model(fitted_model, X_test, y_test)
        cost = calculate_business_cost(
            y_true=y_test,
            y_pred=fitted_model.predict(X_test),
            false_positive_cost=selection_policy.false_positive_cost,
            false_negative_cost=selection_policy.false_negative_cost,
            positive_label=selection_policy.positive_label,
        )
        model_evidence[model_id] = ModelEvidence(evaluation=evaluation, cost=cost)

    comparison = compare_models(model_evidence)
    selection = select(comparison, effective_policy)

    if selection.selection_status != SELECTED or selection.selected_model_id is None:
        return WorkflowResult(
            selected_model_id=None,
            selection_status=selection.selection_status,
            artifact_path=None,
            decision_trace=selection.decision_trace,
        )

    selected_model = fitted_models[selection.selected_model_id]
    save_model(selected_model, artifact_path)

    return WorkflowResult(
        selected_model_id=selection.selected_model_id,
        selection_status=selection.selection_status,
        artifact_path=str(artifact_path),
        decision_trace=selection.decision_trace,
    )
