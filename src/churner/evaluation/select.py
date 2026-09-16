"""Selection of one model from the evidence gathered for several.

This module is the decision step that sits after comparison: it applies an
explicit policy to the evidence collected for each candidate and names at
most one of them. It is the first layer in the workflow that is allowed to
prefer one model over another, and the last one before deployment becomes
a question.

The decision is split in two on purpose. ``SelectionPolicy`` states what
"acceptable" means as configuration — thresholds, the do-nothing baseline,
the cost tolerance, the business weights. The selector reads that
configuration and applies it. Keeping the rules out of the policy object
is what lets the rules be reviewed as data, and what stops two
half-policies from existing in different places.

The selector works from evidence alone. It never sees a fitted model, a
prediction, or a dataset, so it cannot quietly re-measure anything. If the
numbers it is handed were produced incorrectly, it will faithfully select
on them; what it will not do is produce different numbers of its own.

When nothing satisfies the policy, that is the answer. The selector does
not lower a threshold, widen the tolerance, or fall back to the least-bad
candidate, because the point of stating a policy in advance is to make a
failure to meet it visible rather than negotiable.

Nothing here trains a model, scores a pipeline, calculates a business
cost, reads data, saves a model, or deploys one.
"""

from dataclasses import dataclass

from churner.evaluation.compare import ModelComparison, ModelEvidence

# --- Technical gate defaults ---
# A churn model that ranks no better than 0.70 ROC-AUC, or that finds
# fewer than half the customers who leave, is not worth acting on for
# this project. Both figures are project assumptions, not measurements.
MINIMUM_RECALL = 0.50
MINIMUM_ROC_AUC = 0.70

# --- Business cost defaults ---
# The window within which two candidates are treated as costing about the
# same, so that a small cost difference can be traded for higher recall.
COST_TOLERANCE = 0.05

# Illustrative weights: a missed churner is treated as five times as
# expensive as an unnecessary retention offer. They are stated here, on
# the policy, so the figure that produced a candidate's cost and the
# figure the policy assumes cannot drift apart unnoticed.
FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0

# The data dictionary documents ``Churn`` as Yes/No, where "Yes" means the
# customer left.
POSITIVE_LABEL = "Yes"

# --- Selection statuses ---
SELECTED = "SELECTED"
NO_ACCEPTABLE_MODEL = "NO_ACCEPTABLE_MODEL"

# --- Decision trace vocabulary ---
# Small, fixed strings so the trace can be read by a later audit or
# deployment step rather than only by a person.
TECHNICAL_GATE_STAGE = "technical_gate"
BUSINESS_COST_GATE_STAGE = "business_cost_gate"
COST_TOLERANCE_STAGE = "cost_tolerance"
SELECTION_STAGE = "selection"

PASSED_STATUS = "passed"
NO_ACCEPTABLE_CANDIDATES_STATUS = "no_acceptable_candidates"
WITHIN_STATUS = "within"
RECALL_CRITERION = "recall"


@dataclass(frozen=True)
class SelectionPolicy:
    """The conditions a model has to meet to be selected.

    This object holds configuration only. It states the thresholds, the
    baseline it compares costs against, how much extra cost is tolerable
    in exchange for recall, and the business weights those costs assume.
    It does not decide anything: applying these values to candidates is
    the selector's work, so the rules and the numbers they read stay in
    separate places.

    ``baseline_do_nothing_cost`` has no default because it is a property
    of the dataset and the business assumptions rather than of the
    policy: it is what the errors would cost if no model were used at all.
    """

    baseline_do_nothing_cost: float
    minimum_recall: float = MINIMUM_RECALL
    minimum_roc_auc: float = MINIMUM_ROC_AUC
    cost_tolerance: float = COST_TOLERANCE
    false_positive_cost: float = FALSE_POSITIVE_COST
    false_negative_cost: float = FALSE_NEGATIVE_COST
    positive_label: str = POSITIVE_LABEL

    def __post_init__(self) -> None:
        """Refuse a policy that could not be met, or that inverts a cost.

        The two thresholds are metric floors, so they have to lie within
        the range a metric can take: a minimum above 1.0 could never be
        satisfied. The three cost figures and the tolerance have to be
        non-negative, since a negative one would either credit the
        business for a mistake or shrink the tolerance window below the
        cheapest candidate.

        Raises
        ------
        ValueError
            If a threshold falls outside ``0.0``–``1.0``, or if the
            tolerance, the baseline cost, or either weight is negative.
        """
        if not 0.0 <= self.minimum_recall <= 1.0:
            raise ValueError(
                f"minimum_recall must lie between 0.0 and 1.0; got "
                f"{self.minimum_recall}. A floor outside that range could "
                "never be met by a recall figure."
            )

        if not 0.0 <= self.minimum_roc_auc <= 1.0:
            raise ValueError(
                f"minimum_roc_auc must lie between 0.0 and 1.0; got "
                f"{self.minimum_roc_auc}. A floor outside that range could "
                "never be met by an ROC-AUC figure."
            )

        if self.cost_tolerance < 0:
            raise ValueError(
                f"cost_tolerance must not be negative; got "
                f"{self.cost_tolerance}. A negative tolerance would exclude "
                "the cheapest candidate from its own window."
            )

        if self.baseline_do_nothing_cost < 0:
            raise ValueError(
                f"baseline_do_nothing_cost must not be negative; got "
                f"{self.baseline_do_nothing_cost}. Doing nothing cannot pay."
            )

        if self.false_positive_cost < 0:
            raise ValueError(
                f"false_positive_cost must not be negative; got "
                f"{self.false_positive_cost}. A negative weight would credit "
                "the business for a mistake."
            )

        if self.false_negative_cost < 0:
            raise ValueError(
                f"false_negative_cost must not be negative; got "
                f"{self.false_negative_cost}. A negative weight would credit "
                "the business for a mistake."
            )


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of applying one policy to one comparison.

    ``selected_model_id`` is the key the chosen candidate was stored
    under, or ``None`` when no candidate met the policy. No metric and no
    cost is copied in here: those stay on the evidence the caller already
    holds, reachable through that identifier.

    ``decision_trace`` records which stages ran and how they ended. It is
    an audit record of the decision, not an explanation of it.
    """

    selected_model_id: str | None
    selection_status: str
    decision_trace: list[dict[str, object]]


def passes_technical_gate(evidence: ModelEvidence, policy: SelectionPolicy) -> bool:
    """Report whether one candidate clears both metric floors.

    ROC-AUC asks whether the model ranks churners above the rest at all;
    recall asks whether it finds enough of them at its operating point.
    Both are required, and a figure exactly on a floor passes. Precision
    is deliberately not a gate: its cost is expressed through the
    business-cost stage instead of through a threshold.

    Parameters
    ----------
    evidence : ModelEvidence
        The evidence gathered for one candidate. Only read from.
    policy : SelectionPolicy
        The thresholds to apply. Only read from.

    Returns
    -------
    bool
        ``True`` when ROC-AUC and recall both meet or exceed their floors.
    """
    return (
        evidence.evaluation.roc_auc >= policy.minimum_roc_auc
        and evidence.evaluation.recall >= policy.minimum_recall
    )


def passes_business_cost_gate(
    evidence: ModelEvidence, policy: SelectionPolicy
) -> bool:
    """Report whether one candidate costs no more than doing nothing.

    A model whose errors cost more than the baseline is worse than not
    intervening at all, however well it scores technically. A cost equal
    to the baseline passes, since it is not worse.

    Parameters
    ----------
    evidence : ModelEvidence
        The evidence gathered for one candidate. Only read from.
    policy : SelectionPolicy
        The baseline to compare against. Only read from.

    Returns
    -------
    bool
        ``True`` when the candidate's total cost is at most the baseline.
    """
    return evidence.cost.total_cost <= policy.baseline_do_nothing_cost


def selection_order_key(
    candidate: tuple[str, ModelEvidence],
) -> tuple[float, float, str]:
    """Order candidates inside the tolerance window, best first.

    Recall leads, negated so that the highest recall sorts first. Total
    cost breaks a tie in recall, lowest first. The identifier breaks a
    tie in both, so the outcome is decided by the evidence and the name
    rather than by the order the candidates happened to be inserted in.

    Parameters
    ----------
    candidate : tuple[str, ModelEvidence]
        One identifier and its evidence, as taken from the comparison.

    Returns
    -------
    tuple[float, float, str]
        A sort key whose smallest value is the preferred candidate.
    """
    model_id, evidence = candidate
    return (-evidence.evaluation.recall, evidence.cost.total_cost, model_id)


def select(
    comparison: ModelComparison,
    policy: SelectionPolicy,
) -> SelectionResult:
    """Apply a selection policy to gathered model evidence.

    Candidates pass through two gates and then a preference. The
    technical gate keeps models that rank and recall well enough. The
    business-cost gate keeps those that cost no more than doing nothing.
    Among the survivors, the cheapest cost defines a tolerance window,
    and the model with the highest recall inside that window is chosen —
    so a small amount of extra cost is spent on finding more churners,
    deliberately, rather than the cheapest model winning automatically.

    An empty outcome is a real outcome. If either gate leaves nothing,
    the result names no model and says so; no threshold is relaxed and no
    least-bad candidate is promoted. Neither the comparison nor the
    policy is modified, and the same inputs always yield the same result.

    Parameters
    ----------
    comparison : ModelComparison
        The evidence gathered for the candidates, ordinarily from
        ``compare_models``. Only read from.
    policy : SelectionPolicy
        The conditions to apply. Only read from.

    Returns
    -------
    SelectionResult
        The chosen identifier and ``"SELECTED"``, or ``None`` and
        ``"NO_ACCEPTABLE_MODEL"``, together with the trace of the stages
        that ran.
    """
    technically_acceptable = {
        model_id: evidence
        for model_id, evidence in comparison.candidates.items()
        if passes_technical_gate(evidence, policy)
    }
    if not technically_acceptable:
        return SelectionResult(
            selected_model_id=None,
            selection_status=NO_ACCEPTABLE_MODEL,
            decision_trace=[
                {
                    "stage": TECHNICAL_GATE_STAGE,
                    "status": NO_ACCEPTABLE_CANDIDATES_STATUS,
                }
            ],
        )

    affordable = {
        model_id: evidence
        for model_id, evidence in technically_acceptable.items()
        if passes_business_cost_gate(evidence, policy)
    }
    if not affordable:
        return SelectionResult(
            selected_model_id=None,
            selection_status=NO_ACCEPTABLE_MODEL,
            decision_trace=[
                {"stage": TECHNICAL_GATE_STAGE, "status": PASSED_STATUS},
                {
                    "stage": BUSINESS_COST_GATE_STAGE,
                    "status": NO_ACCEPTABLE_CANDIDATES_STATUS,
                },
            ],
        )

    lowest_cost = min(evidence.cost.total_cost for evidence in affordable.values())
    upper_bound = lowest_cost * (1 + policy.cost_tolerance)
    within_window = {
        model_id: evidence
        for model_id, evidence in affordable.items()
        if evidence.cost.total_cost <= upper_bound
    }

    selected_model_id, _ = min(within_window.items(), key=selection_order_key)

    return SelectionResult(
        selected_model_id=selected_model_id,
        selection_status=SELECTED,
        decision_trace=[
            {"stage": TECHNICAL_GATE_STAGE, "status": PASSED_STATUS},
            {"stage": BUSINESS_COST_GATE_STAGE, "status": PASSED_STATUS},
            {"stage": COST_TOLERANCE_STAGE, "status": WITHIN_STATUS},
            {"stage": SELECTION_STAGE, "criterion": RECALL_CRITERION},
        ],
    )
