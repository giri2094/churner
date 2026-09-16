# Model Selection Architecture

**Project:** `churner`
**Implementation:** `src/churner/evaluation/cost.py`, `src/churner/evaluation/compare.py`, `src/churner/evaluation/select.py`
**Tests:** `tests/evaluation/test_cost.py`, `tests/evaluation/test_compare.py`, `tests/evaluation/test_select.py`

---

## Purpose

This document explains the engineering reasoning behind the model selection layer, not merely its contents. It describes the step that turns several measured models into at most one named candidate, and the business-cost calculation that step depends on.

It exists because model choice is where a project quietly stops being a measurement exercise and starts being a decision. The failure it is arranged against is not a wrong metric; it is a decision made after the numbers are known. Once two `EvaluationResult` values are on the table, it is easy to notice that one model has better recall, decide that recall is what matters, and call that a policy. The layer documented here states the conditions first, as data, and then applies them mechanically — including the case where the answer is that no model qualifies.

The layer sits after evaluation:

```text
fitted sklearn Pipeline
    → evaluate_model()             (technical metrics; already implemented)
    → calculate_business_cost()    (business cost; this document)
    → ModelEvidence                (the two halves for one model)
    → compare_models()             (named evidence, no ranking)
    → select()                     (policy applied; this document)
    → SelectionResult
```

Each arrow is a widening of what is known and a narrowing of what is undecided. Evaluation knows metrics and decides nothing. Cost knows business weights and decides nothing. Comparison knows every candidate and still decides nothing. Selection decides, and it is the only step that does.

---

## Current Architecture

```text
y_test, predictions          fitted pipeline, X_test, y_test
        ↓                                   ↓
calculate_business_cost()           evaluate_model()
        ↓                                   ↓
   CostResult                        EvaluationResult
        └───────────────┬───────────────────┘
                        ↓
                  ModelEvidence
                        ↓
        {"logistic_regression": ModelEvidence,
         "decision_tree":       ModelEvidence}
                        ↓
              compare_models()
                        ↓
                 ModelComparison        SelectionPolicy
                        └────────┬────────────┘
                                 ↓
                             select()
                                 ↓
                          SelectionResult
```

| Object | Kind | Responsibility |
|--------|------|----------------|
| `CostResult` | frozen dataclass | Error counts and what they cost |
| `EvaluationResult` | frozen dataclass | The five technical metrics |
| `ModelEvidence` | frozen dataclass | Both halves of what is known about one model |
| `ModelComparison` | frozen dataclass | Named evidence for every candidate |
| `SelectionPolicy` | frozen dataclass | The conditions a model must meet |
| `SelectionResult` | frozen dataclass | The decision and the stages that produced it |
| `calculate_business_cost` | function | Charges prediction errors at supplied weights |
| `compare_models` | function | Collects named evidence |
| `select` | function | Applies a policy to a comparison |

Every result and configuration object in this layer is frozen. These objects are records of one decision made from one set of measurements. A mutable one could be edited after the decision was recorded, which would make the record describe something that never happened.

---

## Why Business Cost Is Separate From Technical Evaluation

Accuracy, precision, recall, F1, and ROC-AUC describe how well a model discriminates. They are properties of the model and the test set, and they do not change when the business changes its mind about what a mistake is worth. Business cost is the opposite: the same predictions cost different amounts under different assumptions about retention spend and customer value.

Keeping them in separate modules keeps that difference visible. `evaluate.py` needs a fitted pipeline, `X_test`, and `y_test`, and it imports sklearn. `cost.py` needs two label sequences and two weights, and it imports neither sklearn nor `evaluate`. Nothing in the cost layer knows what a model is, and that is what allows a business assumption to be revised without touching the measurement code, and a metric to be added without touching the cost code.

The direction of the dependency matters too. If cost lived inside `evaluate_model`, every evaluation would have to commit to a set of weights in order to produce metrics, and a metrics-only reading would no longer be available. As it stands, evaluation happens without any cost assumption at all, and cost is layered on top for the models a decision is actually going to be made about.

### Cost is counted, not derived

The two error counts come from the labels directly:

```python
false_positive_count = sum(
    1
    for actual, predicted in zip(actual_labels, predicted_labels)
    if actual != positive_label and predicted == positive_label
)
```

They are not reconstructed from precision and recall. That reconstruction is possible in principle — the confusion counts can be recovered from the metrics plus the class balance — but it would make the cost figure depend on rounding in three published numbers, and it would make the cost layer depend on the evaluation layer for something it already has in front of it. Counting is both shorter and exact.

### The weights are assumptions, not findings

This project charges:

```text
false positive = 1
false negative = 5
```

These are **illustrative business assumptions**, not measured or optimal figures. Nothing in this repository establishes that missing a churner is five times as costly as an unnecessary retention offer. The real ratio would come from the margin on a retained customer, the cost of the retention offer, and the rate at which offers succeed — none of which this project has.

The ratio is stated as a ratio for a reason. A false positive spends retention budget on a customer who was going to stay: the loss is the cost of the intervention. A false negative loses a customer, or at least the chance of keeping them: the loss is closer to their remaining value. Treating the second as larger is defensible; treating it as exactly five times larger is a placeholder. What the architecture guarantees is that the placeholder is a parameter in one place, so replacing it with a researched figure changes configuration rather than code.

### The positive class is supplied, not assumed

`calculate_business_cost` takes `positive_label` from the caller. This is a deliberate difference from `evaluate_model`, which fixes `POSITIVE_LABEL = "Yes"` as a module constant.

The reason is that the cost function's arithmetic is entirely symmetric in the two classes: swapping which label is positive turns every false positive into a false negative, and the total changes accordingly. There is nothing in the function that is about churn. Fixing the label there would embed this project's target in a calculation that has no other connection to it, and it would hide the one decision — which error is which — that determines the answer.

That choice is checked rather than trusted. The validation compares the labels present against the supplied positive label, and refuses when more than one other label appears. A `positive_label` that does not belong to the data would otherwise make every class the negative one, reporting a perfect zero cost for a model that found nothing.

---

## Why `ModelEvidence` Combines Evaluation and Cost

Selection needs three figures about each candidate: ROC-AUC, recall, and total cost. The first two live on `EvaluationResult` and the third on `CostResult`, and those two objects are produced by different functions with different inputs.

`ModelEvidence` is the smallest object that holds both:

```python
@dataclass(frozen=True)
class ModelEvidence:
    evaluation: EvaluationResult
    cost: CostResult
```

It has two fields and no behaviour. It does not restate `recall` or `total_cost`, because a restated figure is a figure that can disagree with its source. The selector reaches `evidence.evaluation.recall` and `evidence.cost.total_cost`, which is one attribute longer than a flattened object would need and cannot drift from what was measured.

*Alternative considered:* **passing two dictionaries into `select`**, one of evaluations and one of costs, keyed by the same identifiers. Rejected because it makes the pairing an invariant the selector has to defend: keys present in one mapping and absent from the other, or evidence assembled from two different test runs. `ModelEvidence` makes the pair a single object, so a candidate cannot exist half-measured.

It lives in `compare.py` rather than in a module of its own because it exists for comparison's sake; it is the value type `ModelComparison` stores. Placing it there also keeps `cost.py` independent of `evaluate.py`, which a shared third module holding both imports would not.

---

## Why the Model ID Remains the Dictionary Key

Model identity appears in exactly one place: the key of the candidates mapping.

```python
{
    "logistic_regression": ModelEvidence(...),
    "decision_tree": ModelEvidence(...),
}
```

`ModelEvidence` has no `model_id` field. Adding one would create two sources for the same fact, and nothing would keep them equal — evidence stored under `"decision_tree"` could carry `model_id="logistic_regression"`, and every reader would then have to decide which one to believe. A mapping already enforces what the identifier is for: it is unique, it is how the evidence is found, and it cannot contradict itself.

The consequence for the selector is that the identifier is only ever available alongside its evidence, as an item of the mapping. That is why the sort key is built from a `(model_id, evidence)` pair rather than read off the evidence object, and why `SelectionResult` reports a key rather than an object. A caller that wants the metrics of the selected model looks them up:

```python
evidence = comparison.candidates[result.selected_model_id]
```

---

## Why `SelectionPolicy` Is Configuration Only

`SelectionPolicy` holds seven values and no methods other than validation. There is no `policy.select_model(...)`.

The reason is that a policy is a statement about what is acceptable, and it should be readable, reviewable, and storable as such. Once behaviour lives on it, the statement and its interpretation become one object, and reading the policy is no longer enough to know what it will do. Two policies could then differ in behaviour while holding identical values.

Keeping them apart also keeps the counts honest: there is one selection algorithm in this project, in one function, and any number of policies that it can be applied to. A test can vary the tolerance or a threshold and re-apply the same code, which is exactly how the boundary cases below are tested.

The fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `baseline_do_nothing_cost` | *required* | What the errors would cost with no model at all |
| `minimum_recall` | `0.50` | Recall floor for the technical gate |
| `minimum_roc_auc` | `0.70` | ROC-AUC floor for the technical gate |
| `cost_tolerance` | `0.05` | Extra cost tolerated in exchange for recall |
| `false_positive_cost` | `1` | Illustrative weight of an unnecessary offer |
| `false_negative_cost` | `5` | Illustrative weight of a missed churner |
| `positive_label` | `"Yes"` | The documented churn class |

`baseline_do_nothing_cost` has no default because it is a property of the dataset and the weights, not of the policy: it is what the churned customers in the test partition would cost if nobody were contacted. There is no figure that would be right to fall back on, so it is required.

The two cost weights and `positive_label` sit on the policy even though `select` never reads them. They are the assumptions under which every candidate's cost was calculated, and the policy is the object that records the terms of the decision. Keeping them here means the weights that produced the costs and the baseline those costs are compared against are stated together, where a mismatch between them is visible.

Validation refuses values the policy could not mean: a metric floor outside `0.0`–`1.0` could never be met, a negative tolerance would exclude the cheapest candidate from its own window, and a negative cost would credit the business for a mistake. Zero is valid throughout — a zero tolerance is a policy that refuses to pay anything for recall, which is a real position.

---

## Why the Selector Applies the Policy

`select` is a function, not a method, and it holds no state:

```python
select(
    comparison: ModelComparison,
    policy: SelectionPolicy,
) -> SelectionResult
```

Everything it needs arrives as an argument, and everything it concludes leaves as a return value. It cannot train, score, or re-measure anything, because it is never handed a model, a prediction, or a dataset — only figures that were already recorded. The strongest property this gives is negative: a selector that cannot compute a metric cannot select on a metric that was never reported.

It also does not modify what it was given. Each stage builds a new mapping of survivors rather than removing entries from the comparison, so the caller's comparison still holds every candidate afterwards, including the rejected ones. That matters for auditing: the record of what was rejected is as much a part of the decision as the winner.

Determinism follows from the same arrangement. There is no sampling, no clock, no shared state between calls, and the final tie-break is the identifier rather than insertion order, so the same comparison and policy always produce the same result.

---

## The Selection Algorithm

Five stages, in order.

```text
all candidates
      ↓  Stage 1 — technical gate:  ROC-AUC >= 0.70 AND recall >= 0.50
technically acceptable
      ↓  Stage 2 — business cost gate:  total_cost <= baseline_do_nothing_cost
affordable
      ↓  Stage 3 — cost tolerance window:  cost <= cheapest * (1 + 0.05)
inside the window
      ↓  Stage 4 — champion:  highest recall
      ↓  Stage 5 — tie-break:  lower cost, then smaller model ID
selected model
```

### Stage 1 — The technical gates

A candidate passes only if **both** hold:

```text
roc_auc >= minimum_roc_auc      (0.70)
recall  >= minimum_recall       (0.50)
```

The two floors ask different questions, which is why both are required. ROC-AUC asks whether the model ranks churners above non-churners at all, across thresholds. Recall asks whether it actually finds them at the operating point it will be used at. A model can rank respectably and still flag almost nobody; a model can flag nearly everybody and rank no better than chance. Neither is worth acting on.

A figure exactly on a floor passes. The floors are minimums, and a model that meets the stated minimum has met the policy — reading `0.70` as "strictly greater than 0.70" would mean the published threshold was not the real one.

**Precision is deliberately not a gate.** The harm precision measures is wasted retention effort, and that harm is already priced: every false positive is charged at `false_positive_cost` in Stage 2. Gating on precision as well would charge the same mistake twice, once as a threshold and once as a cost, and it would let a model be refused for a precision figure without any statement of what that imprecision was worth. The thresholds cover what cost cannot express; cost covers the rest.

Both floors are project assumptions. They are not derived from a business requirement, and they are not tuned. They encode a minimal usefulness bar: find at least half the churn, and rank better than a coin flip by a clear margin.

### Stage 2 — The business-cost gate

Among technically acceptable candidates, only those satisfying:

```text
total_cost <= baseline_do_nothing_cost
```

are retained. The baseline is what the errors would cost with no model in use: every churner missed, nobody contacted unnecessarily. A model whose weighted errors cost more than that is worse than not intervening, no matter how it scores.

This gate is what makes the technical thresholds insufficient on their own. A model can clear both floors and still lose money, typically by flagging so aggressively that the false positives outweigh the churn it catches. Equality passes, because costing the same as doing nothing is not worse than doing nothing.

The gate applies after the technical one, so an unaffordable candidate never reaches Stage 3 and cannot set the tolerance window. A cheap, useless model does not get to define what "about the same cost" means for the models that are actually in contention.

### Stage 3 — The 5% cost-tolerance window

Among the affordable candidates:

```text
lowest_cost = min(total_cost of affordable candidates)
upper_bound = lowest_cost * (1 + cost_tolerance)
candidate is inside when total_cost <= upper_bound
```

With a tolerance of 0.05 and a cheapest cost of 100, the window reaches 105. A cost of 105 is inside; 105.01 is outside.

The window exists because the cost figures are not precise enough to be ranked by their last digit. They rest on the illustrative 1-and-5 weights, on one test partition, and on one classification threshold per model. A difference of a few percent between two candidates is well within what a different plausible weight ratio or a different random split would move. Treating such a difference as decisive would be reading precision into the figures that they do not have.

So the window defines "costs about the same". Within it, cost stops being the discriminator and something else takes over.

### Stage 4 — Recall wins inside the window

Among the candidates inside the window, the one with the **highest recall** is selected.

This is the substantive preference in the policy, and it means the cheapest model does not automatically win. If two candidates cost 100 and 104, and the dearer one has recall 0.72 against 0.55, the dearer one is selected: 4 units of cost are spent to find substantially more of the customers who are leaving.

The reasoning is that the two figures are not equally trustworthy at this scale. A small cost difference is inside the noise described above. A recall difference is a direct statement about how many churning customers the model actually surfaces, and it is the figure a retention programme is built on. Preferring recall inside the window trades a quantity the project cannot measure precisely for one it can.

The trade is bounded on both sides, which is what makes it safe to state so bluntly. Recall can only win among candidates that already passed both technical floors and already cost no more than doing nothing, and it can only win within a 5% cost band. There is no path by which higher recall justifies an unaffordable model.

### Stage 5 — Deterministic tie-breaking

```text
1. highest recall
2. then lowest total cost
3. then lexicographically smallest model ID
```

Cost returns as the second criterion: once recall cannot separate two candidates, paying more for the same recall has nothing to recommend it.

The third rule is a determinism device and nothing more. Two candidates with identical recall and identical cost are, as far as this policy can see, the same model twice; the policy has no further grounds for preferring either. Something still has to be returned, and the alternative — whichever happened to be first in the mapping — would make the outcome depend on the order the caller assembled the dictionary in. That order is not part of the policy, so it should not be able to decide anything. Sorting on the identifier makes the outcome a function of the evidence and the names alone.

The whole ordering is one sort key:

```python
def selection_order_key(candidate):
    model_id, evidence = candidate
    return (-evidence.evaluation.recall, evidence.cost.total_cost, model_id)
```

The negated recall makes the highest recall sort first, so the preferred candidate is the minimum of the key. Expressing all three rules as one tuple keeps them in one place and in a fixed order; three separate comparison passes would let the priority between them drift.

---

## `NO_ACCEPTABLE_MODEL`

If no candidate passes Stage 1, or if candidates pass Stage 1 but none passes Stage 2, the result is:

```python
SelectionResult(
    selected_model_id=None,
    selection_status="NO_ACCEPTABLE_MODEL",
    decision_trace=[...],
)
```

The selector does not lower a threshold, widen the tolerance, choose the least-bad candidate, or invent an identifier. This is the point of stating a policy before the numbers are known: if the conditions turn out to be unmet, that fact is the finding, and quietly relaxing them would destroy the only thing the policy was for.

`selected_model_id` is `None` rather than a placeholder string, so a caller that forgets to check the status gets a failure at the point of use rather than a lookup against a model that does not exist.

Nothing about this outcome is an error, so nothing is raised. "No candidate met the policy" is a legitimate result of applying a policy, and it belongs in the return value where it can be recorded alongside the trace of how it happened. An exception would push a normal outcome into control flow meant for broken inputs, where it could be caught and swallowed.

What follows from that status is not this layer's business. Retraining, revising the weights, collecting more data, or deciding to deploy nothing are all responses a caller might take, and each is a decision with its own owner.

---

## The Decision Trace

`SelectionResult.decision_trace` is a short list of small dictionaries recording which stages ran and how each ended:

```python
[
    {"stage": "technical_gate", "status": "passed"},
    {"stage": "business_cost_gate", "status": "passed"},
    {"stage": "cost_tolerance", "status": "within"},
    {"stage": "selection", "criterion": "recall"},
]
```

A failure at the first gate records only that gate:

```python
[{"stage": "technical_gate", "status": "no_acceptable_candidates"}]
```

and a failure at the second records the gate that had already passed:

```python
[
    {"stage": "technical_gate", "status": "passed"},
    {"stage": "business_cost_gate", "status": "no_acceptable_candidates"},
]
```

The trace is an audit record: it says which stage the decision reached and what happened there. It is deliberately not an explanation of the reasoning, not a per-candidate log, and not a human-readable narrative. It holds fixed strings so a later deployment or audit step can read it programmatically, and it holds no metrics or costs, because those are on the evidence the caller already has.

It is short on purpose. A trace that grew with the number of candidates would become a log, and a log invites being read instead of the evidence.

---

## Responsibility Boundaries

| Responsibility | Where it lives | Status |
|----------------|----------------|--------|
| Feature schema | `src/churner/schema/features.py` | Implemented |
| Data preparation | `src/churner/data/prepare_modeling_data.py` | Implemented |
| Train/test splitting | `src/churner/training/split_modeling_data.py` | Implemented |
| Preprocessing | `src/churner/preprocessing/preprocessors.py` | Implemented |
| Model pipeline | `src/churner/preprocessing/pipelines.py` | Implemented |
| Training | `src/churner/training/train.py` | Implemented |
| Technical evaluation | `src/churner/evaluation/evaluate.py` | Implemented |
| Business cost | `src/churner/evaluation/cost.py` | Implemented (this document) |
| Evidence collection | `src/churner/evaluation/compare.py` | Implemented (this document) |
| Model selection | `src/churner/evaluation/select.py` | Implemented (this document) |
| Deployment | — | Not designed |

The cost layer explicitly does **not** train models, evaluate ROC-AUC, compute accuracy, precision, or recall, split data, mutate its inputs, or know anything about a fitted model.

The selector explicitly does **not** train models, evaluate models, calculate business costs, read datasets, read predictions, save models, deploy models, modify the comparison, or modify the policy.

### The boundary between selection and deployment

Selection ends at a name. `SelectionResult` says which candidate met the policy, or that none did, and it stops there.

It does not serialise a model, write to a registry, promote anything to an environment, or set a serving threshold. Those steps have properties selection does not: they change the state of a system outside this process, they are not repeatable without consequence, and they need to be reversible. Selection is a pure function of evidence and configuration, and running it twice does nothing at all.

Keeping the boundary here also keeps the interesting decision reviewable in isolation. The reason a model was chosen is a policy applied to figures, and it can be re-derived at any time from the same comparison and policy. Had selection also deployed, the record of the choice would be entangled with the outcome of a deployment, and rerunning it to check the reasoning would mean deploying again.

A deployment layer will therefore consume a `SelectionResult` rather than produce one, and `NO_ACCEPTABLE_MODEL` is a case it has to handle: the decision not to deploy is made here, but carrying it out is not.

---

## Testing Strategy

The three modules have dedicated suites at `tests/evaluation/test_cost.py`, `tests/evaluation/test_compare.py`, and `tests/evaluation/test_select.py`. Every candidate in the selection tests is hand-built with the recall, ROC-AUC, and cost the stage under test needs, rather than produced by training a model, so a failure points at the policy being applied wrongly rather than at a model happening to score a certain way.

The boundary cases use figures whose edges are exact. The cost case pairs six rows into one false positive and two false negatives, giving a hand-calculable total of 11 at the project weights. The tolerance window uses a cheapest cost of 100, so the 5% upper bound is 105: one candidate is placed on the edge at 105 and one just past it at 105.01.

Business cost covers: both error counts, the total at supplied weights, the effect of naming the other class as positive, non-string labels, `Series` and `ndarray` inputs, positional pairing under a non-default index, mismatched lengths, empty inputs, negative weights, zero weights, a third label, a positive label absent from the data, single-class inputs, and non-mutation of the caller's labels.

Comparison covers: evidence stored per model, identity held only in the key, insertion order, dictionary copying, preservation of the exact evidence objects, and refusal of an empty mapping.

Policy covers: immutability, the stated defaults, the required baseline, absence of selection behaviour, and refusal of negative costs, a negative tolerance, and out-of-range thresholds.

Selection covers: a successful selection, both technical failures, exact threshold boundaries, precision not gating, cost above and exactly at the baseline, both sides of the tolerance edge, zero tolerance, recall winning inside the window, cost breaking a recall tie, the identifier breaking a full tie, independence from insertion order, both `NO_ACCEPTABLE_MODEL` paths, all three trace shapes, non-mutation of the comparison and the policy, and repeatability.

| Suite | Result |
|-------|--------|
| Business cost (`test_cost.py`) | **23 passed** |
| Comparison (`test_compare.py`) | **12 passed** |
| Selection (`test_select.py`) | **35 passed** |
| Evaluation package (`tests/evaluation`) | **86 passed** |
| Full test suite | **336 passed** |

Those are the results of the current implementation. No model has been selected from real measurements: nothing here reports which of the two baselines would win on the Telco test partition, or what either would cost.

---

## Design Decisions and Tradeoffs

### Recall preference versus lowest cost

Inside the tolerance window the policy prefers recall.

*Alternative considered:* **selecting the lowest-cost candidate outright.** It is simpler and needs no tolerance parameter. Rejected because it takes the cost figures more seriously than they deserve: it would let a difference of one unit in a total built from illustrative weights outrank a materially better ability to find churning customers. The window is the compromise — cost decides at scales where it is meaningful, recall decides at scales where it is not.

*Alternative considered:* **maximising recall subject to the cost gate alone.** Rejected as the opposite error. Without a window, recall could justify any cost up to the do-nothing baseline, which is a large budget to hand to a single metric.

### A tolerance window versus a weighted score

The window is a hard band, not a soft trade-off.

*Alternative considered:* **a single composite score**, such as a weighted sum of scaled recall and scaled cost. Rejected because the weights of such a score are unreviewable in practice. A stakeholder can be shown "recall at least 0.50, cost no worse than doing nothing, and within 5% on cost we prefer recall" and can agree or disagree with each clause. A composite coefficient hides the same decision inside a number nobody can argue with, and it removes the possibility of the honest answer that no candidate qualifies.

### Two separate gates versus one combined filter

The technical and business gates are applied in sequence, and the trace records them separately.

*Alternative considered:* **one predicate combining all three conditions.** It would be marginally shorter. Rejected because the two failures mean different things and call for different responses: failing the technical gate says the models are not good enough, while failing only the cost gate says they are good but not worth their errors under these assumptions. Collapsing them would produce one indistinguishable `NO_ACCEPTABLE_MODEL`, and the sequencing would be lost — including the rule that only affordable candidates set the tolerance window.

### `NO_ACCEPTABLE_MODEL` as a status versus an exception

The refusal is a returned value.

*Alternative considered:* **raising an exception when nothing qualifies.** Rejected because an unmet policy is a normal outcome of applying a policy, not a broken input. As a return value it carries its trace, it can be recorded, and a deployment layer can handle it explicitly. As an exception it would be caught somewhere and turned into a fallback, which is exactly the behaviour the policy exists to prevent.

### `positive_label` as a parameter of cost but a constant in evaluation

The two modules treat the churn label differently, on purpose.

`evaluate_model` fixes `"Yes"` because it is written against this project's documented target and its ROC-AUC column lookup depends on it. `calculate_business_cost` takes the label as an argument because its arithmetic is symmetric in the two classes and contains nothing specific to churn; the label is the whole content of the decision it makes. `SelectionPolicy` then records the label used, so the terms of a recorded decision include which class was treated as positive.

---

## Current Limitations

* **The FP/FN weights are illustrative.** The 1-and-5 ratio is a project assumption. No business figure supports it.
* **The thresholds are illustrative.** `0.70` ROC-AUC and `0.50` recall are a usefulness bar this project chose, not a requirement anyone stated.
* **The 5% tolerance is a judgement.** It reflects an unquantified belief about how precise the cost figures are, not a measured uncertainty.
* **`baseline_do_nothing_cost` is supplied, not computed.** Nothing in this layer derives it from a test partition, and nothing checks that it was computed with the same weights the policy records.
* **Cost is single-threshold.** Each candidate's cost comes from its default class predictions. No cut-off is searched, and a model's cost-optimal threshold may differ from the one it was measured at.
* **Costs are compared on one partition.** The figures rest on one stratified random split. No confidence interval, no repeated split, and no cross-validated cost is available.
* **Cost is unweighted per customer.** Every false negative costs the same, though customers differ in value and in how likely a retention offer is to work.
* **Selection is one-shot.** There is no champion/challenger comparison against a currently deployed model, and no record of previous selections.
* **Nothing here has been run on real measurements.** No selection outcome for the two baselines is reported in this project.

---

## Future Evolution

Work that may follow, none of which is designed or implemented:

* deriving `baseline_do_nothing_cost` from the test partition and the policy weights, in one place
* threshold selection — choosing each model's operating point by cost rather than accepting the default
* per-customer cost weighting, using tenure or charges as a proxy for value
* uncertainty on the cost figures, from repeated splits or bootstrapping
* champion/challenger selection against a deployed model
* a deployment layer that consumes a `SelectionResult`, including the `NO_ACCEPTABLE_MODEL` case
* recording selections over time, so a policy change is visible as a change in outcome

These are listed to place the current layer in context, not as a specification. The commitment this layer makes to them is the boundary: whatever deploys a model will receive a decision that was made from recorded evidence under a stated policy, and will be able to re-derive that decision without re-measuring anything.
