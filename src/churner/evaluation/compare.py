"""Comparison of already-evaluated Telco churn classification models.

This module is the comparison step that sits after evaluation: it gathers
named ``EvaluationResult`` values and returns them as one frozen record.
The results are stored as they were handed over, so any ranking or
selection remains the caller's.

There is one comparison object rather than a ranked table, because this
layer does not decide which metric matters or which model won.

Nothing here trains a model, scores a pipeline, or computes a difference
between metrics. The evaluations are only collected.
"""

from dataclasses import dataclass

from churner.evaluation.evaluate import EvaluationResult


@dataclass(frozen=True)
class ModelComparison:
    """Named evaluation results of one or more fitted models.

    Each key is the caller-supplied name of a model; each value is the
    ``EvaluationResult`` already computed for that model. Rankings,
    metric differences, and a winning model are not stored here.
    """

    evaluations: dict[str, EvaluationResult]


def compare_models(
    evaluations: dict[str, EvaluationResult],
) -> ModelComparison:
    """Collect named evaluation results into one comparison record.

    The mapping is copied so later changes to the caller's dictionary do
    not alter the comparison, and the copy keeps the original insertion
    order. An empty mapping is refused, because a comparison with no
    models is not a comparison.

    Parameters
    ----------
    evaluations : dict[str, EvaluationResult]
        Named results, ordinarily from ``evaluate_model``. Only read
        from. At least one entry is required.

    Returns
    -------
    ModelComparison
        The supplied evaluations, stored under the names and in the
        order they were given.

    Raises
    ------
    ValueError
        If ``evaluations`` is empty.
    """
    if not evaluations:
        raise ValueError(
            "evaluations must contain at least one EvaluationResult; "
            "got an empty dictionary."
        )

    return ModelComparison(evaluations=dict(evaluations))
