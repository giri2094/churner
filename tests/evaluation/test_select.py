"""Unit tests for the selection policy and the model selector.

These tests establish two contracts. ``SelectionPolicy`` is frozen
configuration with stated defaults that refuses values it could not mean.
``select`` applies that configuration to gathered evidence: two gates,
then a cost-tolerance window, then a preference for recall inside it,
with deterministic tie-breaking and an explicit refusal when nothing
qualifies.

Every candidate below is built by hand with the recall, ROC-AUC, and cost
the stage under test needs, so a failure points at the policy being
applied wrongly rather than at a model happening to score a certain way.
The boundary cases use round numbers whose window edge is exact: a
cheapest cost of 100 at a 5% tolerance admits 105 and excludes 105.01.

Nothing here trains a model, scores a pipeline, calculates a business
cost, or reads a dataset.
"""

import pytest

from churner.evaluation.compare import ModelEvidence, compare_models
from churner.evaluation.cost import CostResult
from churner.evaluation.evaluate import EvaluationResult
from churner.evaluation.select import (
    NO_ACCEPTABLE_MODEL,
    SELECTED,
    SelectionPolicy,
    SelectionResult,
    select,
)

# --- Candidate evidence ---
# The selector reads recall, ROC-AUC, and total cost. The remaining
# metrics and the two error counts are filled in with fixed placeholders,
# so a test that changes behaviour has changed one of the three figures
# the policy actually consults.
UNUSED_ACCURACY = 0.80
UNUSED_PRECISION = 0.50
UNUSED_F1 = 0.55
UNUSED_FALSE_POSITIVE_COUNT = 0
UNUSED_FALSE_NEGATIVE_COUNT = 0

# The do-nothing baseline used by most cases: what the errors would cost
# if no model were deployed at all.
BASELINE_COST = 500.0

EXPECTED_SUCCESSFUL_TRACE = [
    {"stage": "technical_gate", "status": "passed"},
    {"stage": "business_cost_gate", "status": "passed"},
    {"stage": "cost_tolerance", "status": "within"},
    {"stage": "selection", "criterion": "recall"},
]


def make_evidence(
    recall: float,
    roc_auc: float,
    total_cost: float,
    precision: float = UNUSED_PRECISION,
) -> ModelEvidence:
    """Build candidate evidence from the figures the policy consults."""
    return ModelEvidence(
        evaluation=EvaluationResult(
            accuracy=UNUSED_ACCURACY,
            precision=precision,
            recall=recall,
            f1=UNUSED_F1,
            roc_auc=roc_auc,
        ),
        cost=CostResult(
            false_positive_count=UNUSED_FALSE_POSITIVE_COUNT,
            false_negative_count=UNUSED_FALSE_NEGATIVE_COUNT,
            total_cost=total_cost,
        ),
    )


def make_policy(**overrides: object) -> SelectionPolicy:
    """Build a policy on the project defaults, overriding as needed."""
    return SelectionPolicy(baseline_do_nothing_cost=BASELINE_COST, **overrides)


# --- Selection policy: immutability and defaults ---


def test_policy_is_frozen():
    """A policy cannot be edited after it has been stated."""
    policy = make_policy()

    with pytest.raises(Exception):
        policy.minimum_recall = 0.10


def test_policy_defaults_are_the_project_thresholds():
    """The stated defaults are the ones this project decided on."""
    policy = make_policy()

    assert policy.minimum_recall == 0.50
    assert policy.minimum_roc_auc == 0.70
    assert policy.cost_tolerance == 0.05
    assert policy.false_positive_cost == 1
    assert policy.false_negative_cost == 5
    assert policy.positive_label == "Yes"


def test_policy_baseline_cost_has_no_default():
    """The do-nothing baseline has to be supplied.

    It is a property of the dataset and the business assumptions, not of
    the policy, so there is no sensible figure to fall back on.
    """
    with pytest.raises(TypeError):
        SelectionPolicy()


def test_policy_holds_no_selection_behaviour():
    """The policy states conditions; it does not apply them."""
    policy = make_policy()

    assert not hasattr(policy, "select_model")
    assert not hasattr(policy, "select")


# --- Selection policy: validation ---


def test_negative_cost_tolerance_is_rejected():
    """A negative tolerance would exclude the cheapest candidate."""
    with pytest.raises(ValueError, match="cost_tolerance must not be negative"):
        make_policy(cost_tolerance=-0.05)


def test_negative_baseline_cost_is_rejected():
    """Doing nothing cannot pay."""
    with pytest.raises(
        ValueError, match="baseline_do_nothing_cost must not be negative"
    ):
        SelectionPolicy(baseline_do_nothing_cost=-1.0)


def test_negative_cost_weights_are_rejected():
    """A mistake cannot be worth less than nothing."""
    with pytest.raises(ValueError, match="false_positive_cost must not be negative"):
        make_policy(false_positive_cost=-1.0)

    with pytest.raises(ValueError, match="false_negative_cost must not be negative"):
        make_policy(false_negative_cost=-5.0)


def test_thresholds_outside_the_metric_range_are_rejected():
    """A floor above 1.0 or below 0.0 could never be met."""
    with pytest.raises(ValueError, match="minimum_recall must lie between"):
        make_policy(minimum_recall=1.5)

    with pytest.raises(ValueError, match="minimum_roc_auc must lie between"):
        make_policy(minimum_roc_auc=-0.1)


def test_zero_tolerance_and_zero_weights_are_accepted():
    """Refusing any extra cost is a valid policy, not an invalid one."""
    policy = make_policy(
        cost_tolerance=0.0, false_positive_cost=0.0, false_negative_cost=0.0
    )

    assert policy.cost_tolerance == 0.0


# --- Successful selection ---


def test_a_qualifying_candidate_is_selected():
    """One candidate that meets the policy is named as the selection."""
    comparison = compare_models(
        {"logistic_regression": make_evidence(recall=0.60, roc_auc=0.82, total_cost=400.0)}
    )

    result = select(comparison, make_policy())

    assert isinstance(result, SelectionResult)
    assert result.selection_status == SELECTED
    assert result.selected_model_id == "logistic_regression"


def test_thresholds_are_met_exactly():
    """Recall and ROC-AUC sitting on their floors pass the gate."""
    comparison = compare_models(
        {"boundary": make_evidence(recall=0.50, roc_auc=0.70, total_cost=400.0)}
    )

    result = select(comparison, make_policy())

    assert result.selection_status == SELECTED
    assert result.selected_model_id == "boundary"


def test_low_precision_does_not_block_selection():
    """Precision is not a gate; its cost is charged, not thresholded."""
    comparison = compare_models(
        {
            "imprecise": make_evidence(
                recall=0.75, roc_auc=0.80, total_cost=400.0, precision=0.20
            )
        }
    )

    result = select(comparison, make_policy())

    assert result.selection_status == SELECTED


# --- Technical gate ---


def test_roc_auc_below_the_floor_fails_the_technical_gate():
    """A model that ranks too poorly is not selected, however cheap it is."""
    comparison = compare_models(
        {"weak_ranker": make_evidence(recall=0.90, roc_auc=0.69, total_cost=10.0)}
    )

    result = select(comparison, make_policy())

    assert result.selection_status == NO_ACCEPTABLE_MODEL
    assert result.selected_model_id is None


def test_recall_below_the_floor_fails_the_technical_gate():
    """A model that misses too many churners is not selected."""
    comparison = compare_models(
        {"low_recall": make_evidence(recall=0.49, roc_auc=0.90, total_cost=10.0)}
    )

    result = select(comparison, make_policy())

    assert result.selection_status == NO_ACCEPTABLE_MODEL
    assert result.selected_model_id is None


def test_only_technically_acceptable_candidates_can_be_selected():
    """A failing candidate is dropped rather than competing on cost.

    The rejected model is both cheaper and higher-recall, so selecting on
    cost or recall before the gate would have chosen it.
    """
    comparison = compare_models(
        {
            "rejected": make_evidence(recall=0.95, roc_auc=0.60, total_cost=50.0),
            "accepted": make_evidence(recall=0.55, roc_auc=0.75, total_cost=400.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "accepted"


# --- Business cost gate ---


def test_cost_above_the_baseline_is_rejected():
    """A model whose errors cost more than doing nothing is refused."""
    comparison = compare_models(
        {"expensive": make_evidence(recall=0.90, roc_auc=0.90, total_cost=500.01)}
    )

    result = select(comparison, make_policy())

    assert result.selection_status == NO_ACCEPTABLE_MODEL
    assert result.selected_model_id is None


def test_cost_exactly_at_the_baseline_passes():
    """Costing the same as doing nothing is not worse than doing nothing."""
    comparison = compare_models(
        {"at_baseline": make_evidence(recall=0.60, roc_auc=0.80, total_cost=BASELINE_COST)}
    )

    result = select(comparison, make_policy())

    assert result.selection_status == SELECTED
    assert result.selected_model_id == "at_baseline"


def test_candidates_above_the_baseline_are_dropped_before_the_window():
    """An unaffordable candidate does not set the tolerance window.

    The rejected model is the cheapest of the two on nothing but its own
    terms; it is still above the baseline, so it takes no part.
    """
    comparison = compare_models(
        {
            "unaffordable": make_evidence(recall=0.95, roc_auc=0.90, total_cost=600.0),
            "affordable": make_evidence(recall=0.55, roc_auc=0.75, total_cost=300.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "affordable"


# --- Cost tolerance window ---


def test_a_candidate_on_the_window_edge_is_inside():
    """A cost exactly at the upper bound competes on recall.

    The cheapest affordable cost is 100, so at a 5% tolerance the window
    reaches 105. The dearer model is on that edge and has the higher
    recall, so it wins.
    """
    comparison = compare_models(
        {
            "cheapest": make_evidence(recall=0.55, roc_auc=0.80, total_cost=100.0),
            "on_edge": make_evidence(recall=0.80, roc_auc=0.80, total_cost=105.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "on_edge"


def test_a_candidate_past_the_window_edge_is_outside():
    """A cost just over the upper bound cannot win on recall.

    The same window reaches 105. At 105.01 the higher-recall model is
    outside it, so the cheapest model is selected instead.
    """
    comparison = compare_models(
        {
            "cheapest": make_evidence(recall=0.55, roc_auc=0.80, total_cost=100.0),
            "past_edge": make_evidence(recall=0.80, roc_auc=0.80, total_cost=105.01),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "cheapest"


def test_zero_tolerance_admits_only_the_cheapest_cost():
    """With no tolerance, extra cost buys no recall."""
    comparison = compare_models(
        {
            "cheapest": make_evidence(recall=0.55, roc_auc=0.80, total_cost=100.0),
            "dearer": make_evidence(recall=0.90, roc_auc=0.80, total_cost=100.01),
        }
    )

    result = select(comparison, make_policy(cost_tolerance=0.0))

    assert result.selected_model_id == "cheapest"


# --- Champion selection and tie-breaking ---


def test_higher_recall_wins_inside_the_window():
    """Inside the window the policy prefers recall over a lower cost.

    This is the deliberate trade: 4 units of extra cost are spent to find
    more of the customers who leave.
    """
    comparison = compare_models(
        {
            "cheaper_lower_recall": make_evidence(
                recall=0.55, roc_auc=0.80, total_cost=100.0
            ),
            "dearer_higher_recall": make_evidence(
                recall=0.72, roc_auc=0.80, total_cost=104.0
            ),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "dearer_higher_recall"


def test_lower_cost_wins_when_recall_is_equal():
    """Cost decides only once recall cannot."""
    comparison = compare_models(
        {
            "dearer": make_evidence(recall=0.70, roc_auc=0.85, total_cost=104.0),
            "cheaper": make_evidence(recall=0.70, roc_auc=0.75, total_cost=100.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "cheaper"


def test_model_id_breaks_a_tie_in_both_recall_and_cost():
    """Identical evidence is resolved by the identifier, not by order.

    The candidates are inserted with the later identifier first, so a
    selector relying on insertion order would return ``"zeta"``.
    """
    comparison = compare_models(
        {
            "zeta": make_evidence(recall=0.70, roc_auc=0.80, total_cost=100.0),
            "alpha": make_evidence(recall=0.70, roc_auc=0.90, total_cost=100.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selected_model_id == "alpha"


def test_selection_does_not_depend_on_insertion_order():
    """Reordering the candidates does not change the winner."""
    first_order = compare_models(
        {
            "logistic_regression": make_evidence(
                recall=0.55, roc_auc=0.80, total_cost=100.0
            ),
            "decision_tree": make_evidence(recall=0.72, roc_auc=0.75, total_cost=104.0),
        }
    )
    second_order = compare_models(
        {
            "decision_tree": make_evidence(recall=0.72, roc_auc=0.75, total_cost=104.0),
            "logistic_regression": make_evidence(
                recall=0.55, roc_auc=0.80, total_cost=100.0
            ),
        }
    )

    assert (
        select(first_order, make_policy()).selected_model_id
        == select(second_order, make_policy()).selected_model_id
    )


# --- No acceptable model ---


def test_no_technically_acceptable_candidate_selects_nothing():
    """When every candidate fails the metric floors, none is chosen."""
    comparison = compare_models(
        {
            "low_recall": make_evidence(recall=0.20, roc_auc=0.90, total_cost=100.0),
            "weak_ranker": make_evidence(recall=0.90, roc_auc=0.55, total_cost=100.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selection_status == NO_ACCEPTABLE_MODEL
    assert result.selected_model_id is None


def test_no_business_acceptable_candidate_selects_nothing():
    """Passing the metric floors is not enough to be deployed.

    Both candidates rank and recall well, and both cost more than doing
    nothing, so the policy names neither instead of taking the least bad.
    """
    comparison = compare_models(
        {
            "dear": make_evidence(recall=0.80, roc_auc=0.85, total_cost=900.0),
            "dearer": make_evidence(recall=0.90, roc_auc=0.88, total_cost=1200.0),
        }
    )

    result = select(comparison, make_policy())

    assert result.selection_status == NO_ACCEPTABLE_MODEL
    assert result.selected_model_id is None


# --- Decision trace ---


def test_successful_trace_records_every_stage():
    """A selection records the two gates, the window, and the criterion."""
    comparison = compare_models(
        {"logistic_regression": make_evidence(recall=0.60, roc_auc=0.82, total_cost=400.0)}
    )

    result = select(comparison, make_policy())

    assert result.decision_trace == EXPECTED_SUCCESSFUL_TRACE


def test_technical_failure_trace_stops_at_the_first_gate():
    """A failure at the technical gate records that gate only."""
    comparison = compare_models(
        {"low_recall": make_evidence(recall=0.20, roc_auc=0.90, total_cost=100.0)}
    )

    result = select(comparison, make_policy())

    assert result.decision_trace == [
        {"stage": "technical_gate", "status": "no_acceptable_candidates"}
    ]


def test_business_failure_trace_records_both_gates():
    """A failure at the cost gate records the gate that had passed."""
    comparison = compare_models(
        {"dear": make_evidence(recall=0.80, roc_auc=0.85, total_cost=900.0)}
    )

    result = select(comparison, make_policy())

    assert result.decision_trace == [
        {"stage": "technical_gate", "status": "passed"},
        {"stage": "business_cost_gate", "status": "no_acceptable_candidates"},
    ]


def test_result_does_not_restate_metrics_or_costs():
    """The result names a model; the evidence stays where it was."""
    comparison = compare_models(
        {"logistic_regression": make_evidence(recall=0.60, roc_auc=0.82, total_cost=400.0)}
    )

    result = select(comparison, make_policy())

    assert not hasattr(result, "recall")
    assert not hasattr(result, "total_cost")
    assert not hasattr(result, "evaluation")


def test_result_is_frozen():
    """A recorded decision cannot be rewritten after the fact."""
    comparison = compare_models(
        {"logistic_regression": make_evidence(recall=0.60, roc_auc=0.82, total_cost=400.0)}
    )

    result = select(comparison, make_policy())

    with pytest.raises(Exception):
        result.selected_model_id = "decision_tree"


# --- Isolation and determinism ---


def test_comparison_is_not_modified():
    """Selecting reads the comparison and leaves it as it was."""
    logistic_evidence = make_evidence(recall=0.55, roc_auc=0.80, total_cost=100.0)
    tree_evidence = make_evidence(recall=0.72, roc_auc=0.75, total_cost=104.0)
    comparison = compare_models(
        {"logistic_regression": logistic_evidence, "decision_tree": tree_evidence}
    )

    select(comparison, make_policy())

    assert list(comparison.candidates) == ["logistic_regression", "decision_tree"]
    assert comparison.candidates["logistic_regression"] is logistic_evidence
    assert comparison.candidates["decision_tree"] is tree_evidence


def test_policy_is_not_modified():
    """Selecting does not adjust the conditions it was given.

    A policy that widened its own tolerance or lowered a floor during
    selection would make the recorded decision unreproducible.
    """
    policy = make_policy()

    select(
        compare_models(
            {"dear": make_evidence(recall=0.80, roc_auc=0.85, total_cost=900.0)}
        ),
        policy,
    )

    assert policy == make_policy()


def test_the_same_inputs_produce_the_same_result():
    """Selection is deterministic: no sampling, no state carried over."""
    comparison = compare_models(
        {
            "logistic_regression": make_evidence(
                recall=0.55, roc_auc=0.80, total_cost=100.0
            ),
            "decision_tree": make_evidence(recall=0.72, roc_auc=0.75, total_cost=104.0),
        }
    )
    policy = make_policy()

    first_result = select(comparison, policy)
    second_result = select(comparison, policy)

    assert first_result == second_result
