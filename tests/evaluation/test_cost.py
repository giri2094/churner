"""Unit tests for the business-cost calculations.

These tests establish the public contract of
``calculate_business_cost``: it counts false positives and false
negatives from the labels themselves, charges them at the weights it was
given, treats the positive class as something the caller states, refuses
inputs it cannot count, and leaves those inputs as it found them.

They also establish the contract of ``calculate_do_nothing_cost``, the
business baseline a model has to beat: a strategy that predicts the
negative class for everyone misses every actual positive, so its cost is
the number of positive outcomes charged at the false-negative weight.

The four-row case below is written out here rather than produced by a
model, so every expected count and total is hand-calculable and a failure
points at the calculation rather than at agreement with itself.

Nothing here trains a model, scores a pipeline, or asserts anything about
accuracy, precision, recall, F1, or ROC-AUC.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from churner.evaluation.cost import (
    CostResult,
    calculate_business_cost,
    calculate_do_nothing_cost,
)

# --- Illustrative business weights ---
# A missed churner is treated as five times as expensive as an
# unnecessary retention offer. Restated here so the expected totals below
# are readable, not because the figures are measured.
FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0

POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

# --- Hand-calculated six-row case ---
# Pairing the two sequences row by row gives: true negative, true
# positive, false positive, false negative, false negative, true
# negative. So 1 false positive and 2 false negatives, and at the weights
# above a total cost of 1 * 1 + 2 * 5 = 11.
ACTUAL_LABELS = ["No", "Yes", "No", "Yes", "Yes", "No"]
PREDICTED_LABELS = ["No", "Yes", "Yes", "No", "No", "No"]

EXPECTED_FALSE_POSITIVE_COUNT = 1
EXPECTED_FALSE_NEGATIVE_COUNT = 2
EXPECTED_TOTAL_COST = 11.0


# --- Return type ---


def test_calculation_returns_a_cost_result():
    """The calculation hands back a ``CostResult`` and nothing else."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert isinstance(result, CostResult)


def test_cost_result_is_frozen():
    """A recorded cost cannot be altered after it was calculated."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    with pytest.raises(Exception):
        result.total_cost = 0.0


# --- Error counts ---


def test_false_positives_are_counted():
    """A negative outcome predicted as positive is a false positive."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == EXPECTED_FALSE_POSITIVE_COUNT


def test_false_negatives_are_counted():
    """A positive outcome predicted as negative is a false negative."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_negative_count == EXPECTED_FALSE_NEGATIVE_COUNT


def test_correct_predictions_are_not_counted_as_errors():
    """Predictions that match the outcome carry no cost.

    Every row here is predicted correctly, so both error counts and the
    total are zero even though the weights are non-zero.
    """
    result = calculate_business_cost(
        ACTUAL_LABELS,
        ACTUAL_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == 0
    assert result.false_negative_count == 0
    assert result.total_cost == 0.0


# --- Total cost ---


def test_total_cost_charges_each_error_at_its_weight():
    """The total is the two counts multiplied by their weights and added."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.total_cost == EXPECTED_TOTAL_COST


def test_total_cost_follows_the_supplied_weights():
    """Different weights charge the same errors differently.

    The same six rows, with a false negative worth 10 rather than 5,
    cost 1 * 2 + 2 * 10 = 22.
    """
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        false_positive_cost=2.0,
        false_negative_cost=10.0,
        positive_label=POSITIVE_LABEL,
    )

    assert result.total_cost == 22.0


def test_total_cost_is_a_float():
    """The total is a ``float`` even when integer weights are supplied."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        1,
        5,
        POSITIVE_LABEL,
    )

    assert isinstance(result.total_cost, float)


# --- Explicit positive label ---


def test_positive_label_decides_which_errors_are_which():
    """Naming the other class as positive swaps the two counts.

    With ``"No"`` as the positive class, the single row that was a false
    positive becomes a false negative, and the two false negatives
    become false positives. The total therefore becomes
    2 * 1 + 1 * 5 = 7.
    """
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        positive_label=NEGATIVE_LABEL,
    )

    assert result.false_positive_count == EXPECTED_FALSE_NEGATIVE_COUNT
    assert result.false_negative_count == EXPECTED_FALSE_POSITIVE_COUNT
    assert result.total_cost == 7.0


def test_labels_other_than_yes_and_no_are_supported():
    """The calculation is not tied to this project's churn strings.

    Pairing these rows gives one false positive and one false negative,
    so the cost is 1 * 1 + 1 * 5 = 6.
    """
    result = calculate_business_cost(
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        positive_label=1,
    )

    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert result.total_cost == 6.0


# --- Accepted input kinds ---


def test_a_series_and_an_array_are_accepted():
    """The held-out labels and a prediction array can be passed as they are.

    These are the objects the workflow actually produces: a pandas
    ``Series`` from the split and a numpy array from ``predict``.
    """
    result = calculate_business_cost(
        pd.Series(ACTUAL_LABELS),
        np.array(PREDICTED_LABELS),
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == EXPECTED_FALSE_POSITIVE_COUNT
    assert result.false_negative_count == EXPECTED_FALSE_NEGATIVE_COUNT
    assert result.total_cost == EXPECTED_TOTAL_COST


def test_a_series_with_a_non_default_index_is_read_positionally():
    """Rows are paired in order, not by index label.

    The held-out labels carry the index they had in the full dataset,
    while a prediction array has none. Pairing has to be positional for
    the counts to mean anything.
    """
    result = calculate_business_cost(
        pd.Series(ACTUAL_LABELS, index=[101, 7, 55, 3, 900, 42]),
        np.array(PREDICTED_LABELS),
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.total_cost == EXPECTED_TOTAL_COST


# --- Input validation ---


def test_mismatched_lengths_raise_value_error():
    """A prediction without an outcome cannot be charged."""
    with pytest.raises(ValueError, match="same observations"):
        calculate_business_cost(
            ACTUAL_LABELS,
            PREDICTED_LABELS[:-1],
            FALSE_POSITIVE_COST,
            FALSE_NEGATIVE_COST,
            POSITIVE_LABEL,
        )


def test_empty_inputs_raise_value_error():
    """There is no cost to count without observations."""
    with pytest.raises(ValueError, match="at least one observation"):
        calculate_business_cost(
            [],
            [],
            FALSE_POSITIVE_COST,
            FALSE_NEGATIVE_COST,
            POSITIVE_LABEL,
        )


def test_negative_false_positive_cost_raises_value_error():
    """A mistake cannot be worth less than nothing."""
    with pytest.raises(ValueError, match="false_positive_cost must not be negative"):
        calculate_business_cost(
            ACTUAL_LABELS,
            PREDICTED_LABELS,
            -1.0,
            FALSE_NEGATIVE_COST,
            POSITIVE_LABEL,
        )


def test_negative_false_negative_cost_raises_value_error():
    """A mistake cannot be worth less than nothing."""
    with pytest.raises(ValueError, match="false_negative_cost must not be negative"):
        calculate_business_cost(
            ACTUAL_LABELS,
            PREDICTED_LABELS,
            FALSE_POSITIVE_COST,
            -5.0,
            POSITIVE_LABEL,
        )


def test_zero_cost_weights_are_accepted():
    """Charging nothing for a kind of error is a valid assumption.

    With false positives free, only the two false negatives cost
    anything: 2 * 5 = 10.
    """
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        0.0,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == EXPECTED_FALSE_POSITIVE_COUNT
    assert result.total_cost == 10.0


def test_both_weights_zero_gives_no_cost():
    """Errors are still counted when neither of them is charged for."""
    result = calculate_business_cost(
        ACTUAL_LABELS,
        PREDICTED_LABELS,
        0.0,
        0.0,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == EXPECTED_FALSE_POSITIVE_COUNT
    assert result.false_negative_count == EXPECTED_FALSE_NEGATIVE_COUNT
    assert result.total_cost == 0.0


def test_a_third_label_raises_value_error():
    """A non-binary target is refused rather than counted as errors."""
    with pytest.raises(ValueError, match="binary classification"):
        calculate_business_cost(
            ["No", "Yes", "Maybe"],
            ["No", "Yes", "Yes"],
            FALSE_POSITIVE_COST,
            FALSE_NEGATIVE_COST,
            POSITIVE_LABEL,
        )


def test_a_positive_label_absent_from_binary_labels_raises_value_error():
    """A positive label that does not belong to this data is refused.

    Both ``"Yes"`` and ``"No"`` would count as the negative class here,
    which would report every churned customer as correctly retained.
    """
    with pytest.raises(ValueError, match="binary classification"):
        calculate_business_cost(
            ACTUAL_LABELS,
            PREDICTED_LABELS,
            FALSE_POSITIVE_COST,
            FALSE_NEGATIVE_COST,
            positive_label="Churned",
        )


def test_labels_of_one_class_only_are_accepted():
    """A test set holding one class is countable, not an error.

    No customer here churned and none was flagged, so there is nothing
    to charge for.
    """
    result = calculate_business_cost(
        ["No", "No", "No"],
        ["No", "No", "No"],
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert result.false_positive_count == 0
    assert result.false_negative_count == 0
    assert result.total_cost == 0.0


# --- Isolation from the caller's inputs ---


def test_caller_lists_are_not_mutated():
    """The sequences handed over are still what they were afterwards."""
    actual_labels = list(ACTUAL_LABELS)
    predicted_labels = list(PREDICTED_LABELS)

    calculate_business_cost(
        actual_labels,
        predicted_labels,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert actual_labels == ACTUAL_LABELS
    assert predicted_labels == PREDICTED_LABELS


def test_caller_series_and_array_are_not_mutated():
    """A label series and a prediction array are only read from."""
    actual_labels = pd.Series(ACTUAL_LABELS)
    predicted_labels = np.array(PREDICTED_LABELS)

    calculate_business_cost(
        actual_labels,
        predicted_labels,
        FALSE_POSITIVE_COST,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert_series_equal(actual_labels, pd.Series(ACTUAL_LABELS))
    assert list(predicted_labels) == PREDICTED_LABELS


# --- Do-nothing baseline: cost of the strategy that predicts no churn ---

# --- Hand-calculated four-row case ---
# Two of these four outcomes are positive, and the do-nothing strategy
# predicts the negative class for all four, so both positives are missed
# and the baseline costs 2 * 5 = 10. No predictions are supplied, because
# the strategy's predictions are implied rather than made.
BASELINE_LABELS = ["No", "Yes", "No", "Yes"]
EXPECTED_BASELINE_COST = 10.0


def test_do_nothing_cost_charges_every_actual_positive():
    """Predicting no churn misses every churner, at the weight given."""
    cost = calculate_do_nothing_cost(
        BASELINE_LABELS,
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert cost == EXPECTED_BASELINE_COST


def test_do_nothing_cost_is_zero_without_positives():
    """Nobody churned, so doing nothing cost nothing."""
    cost = calculate_do_nothing_cost(
        ["No", "No", "No"],
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert cost == 0.0


def test_do_nothing_cost_charges_all_rows_when_all_are_positive():
    """Every outcome positive means every one of them is missed: 3 * 5 = 15."""
    cost = calculate_do_nothing_cost(
        ["Yes", "Yes", "Yes"],
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert cost == 15.0


def test_do_nothing_cost_follows_the_supplied_positive_label():
    """The baseline is not tied to this project's churn strings.

    Two of these four rows carry the positive label ``1``, so the
    baseline is 2 * 5 = 10 just as it is for ``"Yes"``.
    """
    cost = calculate_do_nothing_cost(
        [0, 1, 1, 0],
        FALSE_NEGATIVE_COST,
        positive_label=1,
    )

    assert cost == 10.0


def test_do_nothing_cost_with_a_zero_weight_is_accepted():
    """Charging nothing for a missed churner is a valid assumption."""
    cost = calculate_do_nothing_cost(
        BASELINE_LABELS,
        0.0,
        POSITIVE_LABEL,
    )

    assert cost == 0.0


def test_do_nothing_cost_rejects_a_negative_weight():
    """A missed churner cannot be worth less than nothing."""
    with pytest.raises(ValueError, match="false_negative_cost must not be negative"):
        calculate_do_nothing_cost(
            BASELINE_LABELS,
            -5.0,
            POSITIVE_LABEL,
        )


def test_do_nothing_cost_rejects_empty_labels():
    """There is no baseline to charge for without observations."""
    with pytest.raises(ValueError, match="at least one observation"):
        calculate_do_nothing_cost(
            [],
            FALSE_NEGATIVE_COST,
            POSITIVE_LABEL,
        )


def test_do_nothing_cost_accepts_a_series_and_an_array():
    """The held-out labels can be passed as the workflow produces them.

    The split hands back a pandas ``Series``; an array of the same labels
    has to give the same baseline.
    """
    series_cost = calculate_do_nothing_cost(
        pd.Series(BASELINE_LABELS),
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )
    array_cost = calculate_do_nothing_cost(
        np.array(BASELINE_LABELS),
        FALSE_NEGATIVE_COST,
        POSITIVE_LABEL,
    )

    assert series_cost == EXPECTED_BASELINE_COST
    assert array_cost == EXPECTED_BASELINE_COST
