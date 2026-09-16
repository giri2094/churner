"""Business cost of the prediction errors a model made on a test set.

This module is the step that sits beside technical evaluation rather than
inside it: it translates the two kinds of mistake a binary classifier can
make into one monetary-style figure. A false positive spends retention
effort on a customer who was going to stay; a false negative withholds it
from a customer who left. Those are different costs, and a metric such as
F1 or ROC-AUC does not express the difference between them.

It is separate from ``evaluate`` because the two answer different
questions. Accuracy, precision, recall, F1, and ROC-AUC describe how well
a model discriminates, and they hold regardless of what the business
spends. Cost describes what those mistakes are worth here, and it changes
when the business assumptions change while the model stays the same.

The error counts are taken from the predictions directly, not derived from
precision or recall. A rounded metric cannot be turned back into a count
without knowing the class balance, and a count is what the cost formula
multiplies.

The weights used by this project are illustrative business assumptions,
not measured figures: a missed churner is treated as five times as
expensive as an unnecessary retention offer. They live on
``SelectionPolicy`` so that one configuration object states them once.

Nothing here trains a model, scores a pipeline, computes a technical
metric, splits data, or writes back to the labels it is handed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Labels arrive as the held-out ``y_test`` series and as the array that
# ``Pipeline.predict`` returns, so both are accepted alongside a plain
# sequence. Whatever is handed over is only read from.
LabelSequence = Sequence[object] | pd.Series | np.ndarray


@dataclass(frozen=True)
class CostResult:
    """Business cost of one model's errors on one set of predictions.

    The two counts are the mistakes that carry a cost; correct
    predictions are not recorded because they do not contribute to the
    total. The labels, the predictions, the cost weights, and the model
    are not stored here.
    """

    false_positive_count: int
    false_negative_count: int
    total_cost: float


def validate_cost_inputs(
    actual_labels: list[object],
    predicted_labels: list[object],
    false_positive_cost: float,
    false_negative_cost: float,
    positive_label: object,
) -> None:
    """Refuse inputs this cost calculation cannot be performed on.

    Four things are checked: that every prediction has a matching
    outcome, that there is something to count, that neither weight is
    negative, and that the labels present describe two classes with
    respect to the supplied positive label.

    The label check compares what is present against
    ``positive_label`` rather than against a fixed pair of strings. Any
    label that is not the positive one counts as the negative class, so
    at most one such label may appear. Two of them would mean either a
    non-binary target or a ``positive_label`` that does not belong to
    this data, and both would otherwise be counted silently as errors.

    Parameters
    ----------
    actual_labels : list[object]
        The recorded outcomes, already materialised as a list.
    predicted_labels : list[object]
        The model's predicted labels, already materialised as a list.
    false_positive_cost : float
        Cost charged per false positive. Zero is permitted.
    false_negative_cost : float
        Cost charged per false negative. Zero is permitted.
    positive_label : object
        The label standing for the positive class.

    Raises
    ------
    ValueError
        If the two label sequences differ in length, if they are empty,
        if either cost weight is negative, or if more than one label
        other than ``positive_label`` is present.
    """
    if len(actual_labels) != len(predicted_labels):
        raise ValueError(
            f"y_true and y_pred must describe the same observations; got "
            f"{len(actual_labels)} outcome(s) and {len(predicted_labels)} "
            "prediction(s)."
        )

    if not actual_labels:
        raise ValueError(
            "y_true and y_pred must contain at least one observation; got "
            "empty inputs, and there is no cost to count."
        )

    if false_positive_cost < 0:
        raise ValueError(
            f"false_positive_cost must not be negative; got {false_positive_cost}. "
            "A negative weight would credit the business for a mistake."
        )

    if false_negative_cost < 0:
        raise ValueError(
            f"false_negative_cost must not be negative; got {false_negative_cost}. "
            "A negative weight would credit the business for a mistake."
        )

    observed_labels = set(actual_labels) | set(predicted_labels)
    negative_labels = observed_labels - {positive_label}
    if len(negative_labels) > 1:
        raise ValueError(
            f"Business cost is defined for binary classification against an "
            f"explicit positive_label; got positive_label={positive_label!r} and "
            f"{sorted(map(repr, negative_labels))} as other label(s). Either the "
            "target is not binary or positive_label does not belong to these labels."
        )


def calculate_business_cost(
    y_true: LabelSequence,
    y_pred: LabelSequence,
    false_positive_cost: float,
    false_negative_cost: float,
    positive_label: object,
) -> CostResult:
    """Charge one model's prediction errors at the supplied cost weights.

    The two error counts are taken from the labels themselves: a false
    positive is a negative outcome predicted as positive, and a false
    negative is a positive outcome predicted as negative. The total is
    those counts multiplied by their weights and added.

    The positive class is supplied by the caller rather than assumed, so
    the same function serves a target whose churn label is not ``"Yes"``.
    Both label sequences are copied and only read from.

    Parameters
    ----------
    y_true : LabelSequence
        Recorded outcomes, ordinarily the held-out ``y_test``. Only read
        from.
    y_pred : LabelSequence
        Predicted labels aligned to ``y_true``, ordinarily the array
        returned by ``Pipeline.predict``. Only read from.
    false_positive_cost : float
        Cost charged for each negative outcome predicted as positive.
        Zero is permitted.
    false_negative_cost : float
        Cost charged for each positive outcome predicted as negative.
        Zero is permitted.
    positive_label : object
        The label standing for the positive class. For this project's
        ``Churn`` target that is ``"Yes"``.

    Returns
    -------
    CostResult
        The two error counts and the cost they add up to.

    Raises
    ------
    ValueError
        If the two label sequences differ in length, if they are empty,
        if either cost weight is negative, or if the labels present are
        not binary with respect to ``positive_label``.
    """
    actual_labels = list(y_true)
    predicted_labels = list(y_pred)

    validate_cost_inputs(
        actual_labels,
        predicted_labels,
        false_positive_cost,
        false_negative_cost,
        positive_label,
    )

    false_positive_count = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels)
        if actual != positive_label and predicted == positive_label
    )
    false_negative_count = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels)
        if actual == positive_label and predicted != positive_label
    )

    return CostResult(
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        total_cost=float(
            false_positive_count * false_positive_cost
            + false_negative_count * false_negative_cost
        ),
    )
