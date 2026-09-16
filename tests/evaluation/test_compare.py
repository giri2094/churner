"""Unit tests for the model-comparison collection function.

These tests establish the public contract of ``compare_models``: it
gathers named ``ModelEvidence`` values into a frozen ``ModelComparison``,
keeps the caller's insertion order, copies the mapping so later edits
cannot reach the comparison, and refuses an empty dictionary.

The evidence below is written out here rather than produced by
``evaluate_model`` and ``calculate_business_cost``, so a test fails if
comparison alters or replaces the objects it was given rather than
collecting them. The numbers are labels for identity checks, not a
ranking.

Nothing here trains a model, scores a pipeline, or calculates a cost.
Ranking, metric differences, and a winning model are out of scope.
"""

import pytest

from churner.evaluation.compare import ModelComparison, ModelEvidence, compare_models
from churner.evaluation.cost import CostResult
from churner.evaluation.evaluate import EvaluationResult

# --- Named evidence used as comparison inputs ---
# Distinct objects with distinct names, so a test can see whether the
# comparison kept the caller's keys, order, and evidence objects. No test
# below asks which of these models is better.
LOGISTIC_EVIDENCE = ModelEvidence(
    evaluation=EvaluationResult(
        accuracy=0.80,
        precision=0.55,
        recall=0.40,
        f1=0.46,
        roc_auc=0.82,
    ),
    cost=CostResult(
        false_positive_count=120,
        false_negative_count=224,
        total_cost=1240.0,
    ),
)
TREE_EVIDENCE = ModelEvidence(
    evaluation=EvaluationResult(
        accuracy=0.77,
        precision=0.50,
        recall=0.62,
        f1=0.55,
        roc_auc=0.74,
    ),
    cost=CostResult(
        false_positive_count=232,
        false_negative_count=142,
        total_cost=942.0,
    ),
)


# --- Return type ---


def test_a_single_candidate_returns_a_model_comparison():
    """A comparison of one named model is still a ``ModelComparison``."""
    comparison = compare_models({"logistic": LOGISTIC_EVIDENCE})

    assert isinstance(comparison, ModelComparison)
    assert len(comparison.candidates) == 1


def test_multiple_candidates_are_accepted():
    """Several named models are collected in one comparison record."""
    comparison = compare_models(
        {"logistic": LOGISTIC_EVIDENCE, "tree": TREE_EVIDENCE}
    )

    assert isinstance(comparison, ModelComparison)
    assert len(comparison.candidates) == 2


# --- Evidence stored per model ---


def test_candidates_hold_model_evidence():
    """Each value is the evidence gathered for that model.

    Evaluation and cost are reached through the evidence object rather
    than restated on it.
    """
    comparison = compare_models({"logistic": LOGISTIC_EVIDENCE})

    evidence = comparison.candidates["logistic"]

    assert isinstance(evidence, ModelEvidence)
    assert evidence.evaluation.recall == 0.40
    assert evidence.cost.total_cost == 1240.0


def test_model_identity_lives_only_in_the_key():
    """The evidence itself carries no model identifier.

    The name is the key the evidence is stored under, so it is recorded
    in exactly one place.
    """
    comparison = compare_models({"logistic": LOGISTIC_EVIDENCE})

    assert list(comparison.candidates) == ["logistic"]
    assert not hasattr(comparison.candidates["logistic"], "model_id")


def test_comparison_contains_the_supplied_model_names():
    """The keys on the comparison are the identifiers the caller supplied."""
    comparison = compare_models(
        {"logistic": LOGISTIC_EVIDENCE, "tree": TREE_EVIDENCE}
    )

    assert set(comparison.candidates) == {"logistic", "tree"}


def test_comparison_preserves_insertion_order():
    """The identifiers appear in the order they were inserted, not sorted.

    Alphabetical order of these two names would put logistic first.
    """
    candidates = {"tree": TREE_EVIDENCE, "logistic": LOGISTIC_EVIDENCE}

    comparison = compare_models(candidates)

    assert list(comparison.candidates) == ["tree", "logistic"]


# --- Isolation from the caller's dictionary ---


def test_returned_mapping_is_a_separate_dictionary():
    """The comparison does not keep the caller's dictionary by identity."""
    candidates = {"logistic": LOGISTIC_EVIDENCE, "tree": TREE_EVIDENCE}

    comparison = compare_models(candidates)

    assert comparison.candidates is not candidates


def test_mutating_the_input_does_not_modify_the_comparison():
    """A later insert into the caller's mapping cannot reach the comparison."""
    candidates = {"logistic": LOGISTIC_EVIDENCE}

    comparison = compare_models(candidates)
    candidates["tree"] = TREE_EVIDENCE

    assert "tree" not in comparison.candidates
    assert list(comparison.candidates) == ["logistic"]


# --- Preservation of the supplied evidence ---


def test_supplied_evidence_objects_are_preserved():
    """The values are the same ``ModelEvidence`` objects that were handed in.

    Comparison collects evidence; it does not rebuild it.
    """
    comparison = compare_models(
        {"logistic": LOGISTIC_EVIDENCE, "tree": TREE_EVIDENCE}
    )

    assert comparison.candidates["logistic"] is LOGISTIC_EVIDENCE
    assert comparison.candidates["tree"] is TREE_EVIDENCE


def test_evidence_is_frozen():
    """Neither half of a model's evidence can be replaced after the fact."""
    with pytest.raises(Exception):
        LOGISTIC_EVIDENCE.cost = TREE_EVIDENCE.cost


# --- Input validation ---


def test_empty_dictionary_raises_value_error():
    """A comparison with no models is refused."""
    with pytest.raises(ValueError):
        compare_models({})


def test_empty_dictionary_error_states_that_evidence_is_required():
    """The refusal says that at least one ``ModelEvidence`` is required."""
    with pytest.raises(ValueError, match="at least one ModelEvidence"):
        compare_models({})
