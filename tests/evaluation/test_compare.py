"""Unit tests for the model-comparison collection function.

These tests establish the public contract of ``compare_models``: it
gathers named ``EvaluationResult`` values into a frozen
``ModelComparison``, keeps the caller's insertion order, copies the
mapping so later edits cannot reach the comparison, and refuses an
empty dictionary.

The results below are written out here rather than produced by
``evaluate_model``, so a test fails if comparison alters or replaces
the objects it was given rather than collecting them. The numbers are
labels for identity checks, not a ranking.

Nothing here trains a model or scores a pipeline. Ranking, metric
differences, and a winning model are out of scope.
"""

import pytest

from churner.evaluation.compare import ModelComparison, compare_models
from churner.evaluation.evaluate import EvaluationResult

# --- Named results used as comparison inputs ---
# Distinct objects with distinct names, so a test can see whether the
# comparison kept the caller's keys, order, and result objects. No test
# below asks which of these models is better.
LOGISTIC_RESULT = EvaluationResult(
    accuracy=0.80,
    precision=0.55,
    recall=0.40,
    f1=0.46,
    roc_auc=0.82,
)
TREE_RESULT = EvaluationResult(
    accuracy=0.77,
    precision=0.50,
    recall=0.62,
    f1=0.55,
    roc_auc=0.74,
)


# --- Return type ---


def test_a_single_evaluation_returns_a_model_comparison():
    """A comparison of one named result is still a ``ModelComparison``."""
    comparison = compare_models({"logistic": LOGISTIC_RESULT})

    assert isinstance(comparison, ModelComparison)
    assert len(comparison.evaluations) == 1


def test_multiple_evaluations_are_accepted():
    """Several named results are collected in one comparison record."""
    comparison = compare_models(
        {"logistic": LOGISTIC_RESULT, "tree": TREE_RESULT}
    )

    assert isinstance(comparison, ModelComparison)
    assert len(comparison.evaluations) == 2


# --- Named evaluations ---


def test_comparison_contains_the_supplied_model_names():
    """The keys on the comparison are the names the caller supplied."""
    comparison = compare_models(
        {"logistic": LOGISTIC_RESULT, "tree": TREE_RESULT}
    )

    assert set(comparison.evaluations) == {"logistic", "tree"}


def test_comparison_preserves_insertion_order():
    """The names appear in the order they were inserted, not sorted.

    Alphabetical order of these two names would put logistic first.
    """
    evaluations = {"tree": TREE_RESULT, "logistic": LOGISTIC_RESULT}

    comparison = compare_models(evaluations)

    assert list(comparison.evaluations) == ["tree", "logistic"]


# --- Isolation from the caller's dictionary ---


def test_returned_mapping_is_a_separate_dictionary():
    """The comparison does not keep the caller's dictionary by identity."""
    evaluations = {"logistic": LOGISTIC_RESULT, "tree": TREE_RESULT}

    comparison = compare_models(evaluations)

    assert comparison.evaluations is not evaluations


def test_mutating_the_input_does_not_modify_the_comparison():
    """A later insert into the caller's mapping cannot reach the comparison."""
    evaluations = {"logistic": LOGISTIC_RESULT}

    comparison = compare_models(evaluations)
    evaluations["tree"] = TREE_RESULT

    assert "tree" not in comparison.evaluations
    assert list(comparison.evaluations) == ["logistic"]


# --- Preservation of the supplied results ---


def test_supplied_evaluation_results_are_preserved():
    """The values are the same ``EvaluationResult`` objects that were handed in.

    Comparison collects results; it does not rebuild them.
    """
    comparison = compare_models(
        {"logistic": LOGISTIC_RESULT, "tree": TREE_RESULT}
    )

    assert comparison.evaluations["logistic"] is LOGISTIC_RESULT
    assert comparison.evaluations["tree"] is TREE_RESULT


# --- Input validation ---


def test_empty_dictionary_raises_value_error():
    """A comparison with no models is refused."""
    with pytest.raises(ValueError):
        compare_models({})


def test_empty_dictionary_error_states_that_a_result_is_required():
    """The refusal says that at least one ``EvaluationResult`` is required."""
    with pytest.raises(ValueError, match="at least one EvaluationResult"):
        compare_models({})
