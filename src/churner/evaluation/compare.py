"""Comparison of already-measured Telco churn classification models.

This module is the comparison step that sits after evaluation and cost
calculation: it gathers the evidence collected for each named model and
returns it as one frozen record. The evidence is stored as it was handed
over, so any ranking or selection remains the caller's.

There is one comparison object rather than a ranked table, because this
layer does not decide which metric matters or which model won. That
decision belongs to ``select``, which applies an explicit policy to the
evidence gathered here.

A model's identity lives only in the mapping key. Nothing about a model
is recorded twice: the technical metrics stay in ``EvaluationResult``,
the business cost stays in ``CostResult``, and the name stays in the key
that reaches both.

Nothing here trains a model, scores a pipeline, calculates a cost, or
computes a difference between metrics. The evidence is only collected.
"""

from dataclasses import dataclass

from churner.evaluation.cost import CostResult
from churner.evaluation.evaluate import EvaluationResult


@dataclass(frozen=True)
class ModelEvidence:
    """Everything measured about one model, gathered for comparison.

    The two halves answer different questions and are kept as the objects
    that produced them: ``evaluation`` describes how well the model
    discriminates, and ``cost`` describes what its mistakes are worth
    under the business assumptions in force. Neither half's fields are
    restated here, and neither is the model's name: the name is the key
    this evidence is stored under.
    """

    evaluation: EvaluationResult
    cost: CostResult


@dataclass(frozen=True)
class ModelComparison:
    """Named evidence for one or more measured models.

    Each key is the caller-supplied identifier of a model; each value is
    the ``ModelEvidence`` already gathered for it. Rankings, metric
    differences, and a winning model are not stored here.
    """

    candidates: dict[str, ModelEvidence]


def compare_models(
    candidates: dict[str, ModelEvidence],
) -> ModelComparison:
    """Collect named model evidence into one comparison record.

    The mapping is copied so later changes to the caller's dictionary do
    not alter the comparison, and the copy keeps the original insertion
    order. An empty mapping is refused, because a comparison with no
    models is not a comparison.

    Parameters
    ----------
    candidates : dict[str, ModelEvidence]
        Model identifiers mapped to the evidence gathered for each, the
        evaluation ordinarily from ``evaluate_model`` and the cost from
        ``calculate_business_cost``. Only read from. At least one entry
        is required.

    Returns
    -------
    ModelComparison
        The supplied evidence, stored under the identifiers and in the
        order they were given.

    Raises
    ------
    ValueError
        If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError(
            "candidates must contain at least one ModelEvidence; "
            "got an empty dictionary."
        )

    return ModelComparison(candidates=dict(candidates))
