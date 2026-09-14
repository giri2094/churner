# Model Evaluation Architecture

**Project:** `churner`
**Implementation:** `src/churner/evaluation/evaluate.py`
**Tests:** `tests/evaluation/test_evaluate.py`

---

## Purpose

This document explains the engineering reasoning behind the baseline model evaluation layer, not merely its contents. It describes one specific step of the ML workflow: the measurement of an already-fitted classification pipeline on held-out test records.

It exists because evaluation is easy to get wrong silently. A metric that looks like a test score can in fact be a measurement of the test set against itself, if preprocessing is refitted on those records, if the classifier is trained again, or if ROC-AUC is computed from hard class labels rather than from ranking scores. Those mistakes do not always raise an error. The layer documented here is narrow on purpose: it scores a fitted object and returns five numbers, and it is arranged so that the high-impact failures — refitting, test leakage, and scoring the wrong kind of output — are the ones the contract makes hard to commit.

The layer sits after training:

```text
raw CSV
    → prepare_modeling_data()      (deterministic; already implemented)
    → split_modeling_data()        (evaluation boundary; already implemented)
    → train_logistic / train_tree  (fit on training records only)
    → fitted sklearn Pipeline
    → evaluate_model()             (this document)
    → EvaluationResult
```

Evaluation is separate from training because the two operations answer different questions. Training produces a fitted object from records the model is allowed to learn from. Evaluation measures that object on records it is not allowed to have learned from. Combining them would hide the boundary that gives the reported numbers their meaning.

---

## Current Architecture

The modelling path that reaches evaluation is:

```text
raw dataset
        ↓
prepare_modeling_data()
        ↓
X, y
        ↓
split_modeling_data()
        ↓
X_train / y_train          X_test / y_test
        ↓                          │
train_logistic() or                │
train_tree()                       │
        ↓                          │
fitted sklearn Pipeline            │
        ↓                          │
        └──────────┬───────────────┘
                   ↓
        evaluate_model(fitted_model, X_test, y_test)
                   ↓
             EvaluationResult
```

Two objects are involved, and they are not interchangeable.

**The fitted pipeline** is the thing being measured. It is the product of training: a `preprocessor` step whose statistics were learned from `X_train`, followed by a `classifier` step whose parameters were learned from the preprocessed training matrix and `y_train`. After `fit`, that object already knows how to take a raw feature frame, apply the training-time transformations, and emit class labels or class probabilities. Evaluation does not assemble this object.

**The evaluator** is the thing that measures. It receives the fitted pipeline, the held-out predictors, and the held-out labels. It asks the pipeline for predictions and scores, computes five sklearn metrics, and returns those numbers in an `EvaluationResult`. It does not own preprocessing, it does not own the classifier, and it does not decide which records are test records.

The current implementation provides exactly one public evaluator:

| Function | Input | Output |
|----------|-------|--------|
| `evaluate_model(fitted_model, X_test, y_test)` | A fitted sklearn `Pipeline`, plus held-out features and labels | `EvaluationResult` |

There is no `evaluate_logistic()` and no `evaluate_tree()`. Both baseline trainers return a fitted `Pipeline` that exposes `predict` and `predict_proba`, so both satisfy the same scoring contract.

The metrics themselves are not defined here. They come from sklearn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`). This module decides *which outputs of the fitted pipeline those functions are handed*, which is the decision that determines what the five numbers mean.

---

## Evaluation Contract

The public function is:

```python
evaluate_model(
    fitted_model,
    X_test,
    y_test,
) -> EvaluationResult
```

`fitted_model` is an already-fitted sklearn `Pipeline`, ordinarily the object returned by `train_logistic` or `train_tree`. The evaluator does not construct a pipeline, does not call `fit`, and does not replace any step. It calls `predict` and `predict_proba` on the object it was given.

`X_test` and `y_test` are the held-out evaluation partition, ordinarily the test pair returned by `split_modeling_data`. They are an evaluation boundary, not a second training set. The evaluator reads them and does not write back to them. It is not handed `X_train` or `y_train` at all, so training records cannot enter the score through this function.

The result is only the five baseline metrics. Predictions, probabilities, the pipeline, and the test frames are used to compute those metrics and are then discarded.

---

## EvaluationResult

The return type is a frozen dataclass with exactly five fields:

| Field | Source |
|-------|--------|
| `accuracy` | class predictions from `predict()` |
| `precision` | class predictions from `predict()`, positive label `"Yes"` |
| `recall` | class predictions from `predict()`, positive label `"Yes"` |
| `f1` | class predictions from `predict()`, positive label `"Yes"` |
| `roc_auc` | positive-class scores from `predict_proba()` |

A dataclass is used instead of a dictionary because the result is a fixed contract, not an open bag of values. The five names are part of the interface: a caller reaches `result.recall`, not `result["recall"]`. A misspelt field is a type error or an attribute error at the point of use, rather than a missing key discovered later. The types are stated once, as `float`, so the object cannot quietly hold a model, a frame, or an array in a metric slot.

It is frozen so that a caller cannot alter a reported number after the fact. The metrics describe one scoring of one fitted object on one test partition. Mutating them would make that description drift away from what was computed.

The dataclass stores evaluation results only. It does not store:

* the fitted model
* `X_test` or `y_test`
* the class predictions
* the probability matrix
* any preprocessor or classifier

Those objects belong either to the caller or to the calculation that produced the numbers. Keeping them out of `EvaluationResult` is what stops the result type from becoming a second home for the pipeline, and what keeps a later comparison of two results a comparison of metrics rather than of leftover state.

---

## Why Evaluation Receives the Already-Fitted Pipeline

The pipeline that training returns already carries the preprocessing state learned from `X_train`: the numerical medians, the logistic scaling parameters where they exist, the modal categories, and the one-hot mappings. `predict` and `predict_proba` apply that state and then score. Evaluation needs exactly that behaviour. The test frame is still a raw feature DataFrame of the same shape training saw; the fitted pipeline is what turns it into the matrix the classifier expects.

If evaluation rebuilt preprocessing, it would have to decide what to fit it on. Fitting it on `X_test` would learn medians, scales, and categories from the records being scored. Fitting it on some other frame would pair the classifier with statistics it was not trained against. Either way, the object being measured would no longer be the object that was trained.

If evaluation retrained the classifier, the reported metrics would describe a different model. The test labels would have been used twice: once as the thing the model was fitted to, and again as the thing it was scored against. That is not a test.

Receiving the fitted pipeline therefore does three related jobs:

1. **The same fitted preprocessing is applied to `X_test`.** Missing amounts are filled with the training median, not with a median computed from the test rows. Categories unseen at training time are encoded with the mapping learned then, not with a mapping that has seen the test set.
2. **The classifier is not retrained.** Coefficients, splits, and class priors stay as training left them.
3. **The object being evaluated is identical to the object that was trained.** A later comparison between logistic and tree scores is a comparison of two fitted pipelines, not of two ad-hoc scoring paths that happen to share a name.

This is what preserves the training/evaluation boundary, and what reduces the risk of preprocessing leakage through this layer. The evaluator never calls `fit` or `fit_transform`. The test suite replaces `fit` on the handed-over pipeline and snapshots the imputer statistics and classifier parameters, so a refit cannot hide behind unchanged numbers.

### An important precision

Receiving a sklearn `Pipeline` does **not** magically prevent data leakage.

`Pipeline.fit` fits every step on whatever frame it is handed. If the pipeline was fitted on the full dataset, evaluation will faithfully apply statistics that already include the test rows. The evaluator cannot see that this has happened. It will still return five numbers, and those numbers will still look like a test score.

Leakage prevention depends on the pipeline having been fitted only on training data, which is the contract of `train_logistic` and `train_tree`: they accept `X_train` and `y_train` only. Evaluation then applies what that fit learned. The guarantee is compositional. This layer keeps the correct call the only call it offers; it does not inspect the pipeline for evidence of an earlier misuse.

---

## Class Predictions vs Continuous Scores

The fitted pipeline exposes two different outputs, and they answer two different questions.

**`predict()` produces hard class labels.** Each row is assigned `"Yes"` or `"No"`. That assignment is the model's decision at its default classification threshold. Accuracy, precision, recall, and F1 are computed from those labels compared with `y_test`. They describe one operating point: how the model behaves when it is forced to choose a class.

**`predict_proba()` produces a continuous score per class.** For each row the pipeline returns a pair of values that sum to one, in the order of `fitted_model.classes_`. ROC-AUC is computed from the column that corresponds to the positive label `"Yes"`. That score is used as a ranking signal: given two customers, one who churned and one who did not, how often the model assigns the higher positive-class score to the one who churned.

ROC-AUC therefore evaluates ranking across thresholds rather than one fixed classification threshold. A model can rank well and still look mediocre at a single cut-off, or the reverse. The two families of metric are retained because they are not substitutes for one another. The test suite includes a four-row case in which the class labels yield an ROC-AUC of 0.50 while the probabilities yield 0.75; a result of 0.75 cannot have come from the labels alone.

The values returned by `predict_proba()` are **model scores**, not automatically calibrated probabilities. Logistic Regression and a decision tree both expose `predict_proba()`, but that shared method name does not mean that a score of 0.8 corresponds to an observed churn frequency of 80%. Calibration is a separate question, and this layer does not ask it. The scores are used because ROC-AUC needs a continuous ranking, not because they have been shown to be probabilities in the frequency sense.

---

## Metric Rationale

The five metrics are a baseline set, not an exhaustive one. They are retained together because each answers a question the others do not, and because a churn dataset with a minority positive class makes any single number easy to over-read.

Every class-based metric is a summary of the same four counts:

| | Predicted retained (`"No"`) | Predicted churned (`"Yes"`) |
|---|---|---|
| **Actually retained (`"No"`)** | True negative (TN) | False positive (FP) |
| **Actually churned (`"Yes"`)** | False negative (FN) | True positive (TP) |

From those counts:

* **Accuracy** is `(TP + TN) / (TP + TN + FP + FN)` — the share of test rows whose predicted class matches the true class.
* **Precision** is `TP / (TP + FP)` — of the customers the model flags as churn, how many actually churned.
* **Recall** is `TP / (TP + FN)` — of the customers who actually churned, how many the model flagged.
* **F1** is the harmonic mean of precision and recall — a single number that falls when either side of that tradeoff collapses.
* **ROC-AUC** is not a function of one confusion matrix. It summarises, across thresholds, how well the positive-class scores rank churned customers above retained ones.

Accuracy alone is insufficient when the classes are imbalanced. The Telco target is. The exploratory baseline recorded 5,174 retained customers (73.46%) and 1,869 churned customers (26.54%). A classifier that predicted `"No"` for every row would be right on roughly three quarters of the records and would still have found none of the churn. Accuracy would report that majority-class success; precision, recall, F1, and ROC-AUC would not.

The business reading of false positives and false negatives is a tradeoff, not a hierarchy.

A **false positive** is a retained customer scored as likely to leave. Acting on it spends retention effort — contact, discount, intervention — on an account that was not going to churn. The cost is operational: budget, goodwill, and the dilution of whatever treatment is being offered.

A **false negative** is a churned customer scored as likely to stay. Missing it withholds that same intervention from an account that did leave. The cost is the lost customer, or at least the lost chance to try to keep them.

Which of those costs dominates is not a property of the metric formulae, and it is not settled by this layer. Precision emphasises the cost of acting on the wrong people; recall emphasises the cost of missing people who leave; F1 refuses to let either side go to zero; ROC-AUC asks whether the ranking underneath the chosen threshold is any good. The five numbers are reported so that a later decision about operating point, intervention cost, or model choice has something to work from. None of them is designated the most important metric.

---

## Positive Class Handling

The data dictionary documents `Churn` as `"Yes"` / `"No"`, where `"Yes"` means the customer left during the last period. Precision, recall, F1, and the ROC-AUC ranking are reported for that class. The evaluator states this once as `POSITIVE_LABEL = "Yes"` and passes it to the class-based sklearn metrics as `pos_label`.

The probability column is not assumed to sit at a fixed position. `predict_proba` returns columns in the order of `fitted_model.classes_`, which is determined by the labels seen during `fit`. The evaluator looks up `"Yes"` in that array and takes the matching column:

```text
class_probabilities = fitted_model.predict_proba(X_test)
positive_class_index = list(fitted_model.classes_).index("Yes")
positive_scores = class_probabilities[:, positive_class_index]
```

That lookup is the difference between scoring `P(Yes)` and accidentally scoring `P(No)`. sklearn's current `roc_auc_score` then receives those continuous scores directly; it no longer takes a `pos_label` argument, so choosing the correct column is the entire positive-class decision for ROC-AUC.

The label is fixed to `"Yes"` because this project has one target and one documented positive class. Making it a parameter would suggest a generality the rest of the workflow does not have. If a later target used different strings, the constant would have to change; until then, the churn label is stated explicitly rather than inferred from `classes_[1]`.

---

## Why One Generic Evaluator

`evaluate_model()` is model-agnostic because evaluation does not depend on how the pipeline was assembled. Logistic Regression and the decision tree differ in preprocessing and in estimator, which is why they have separate factories and separate trainers. They do not differ in how they are scored. Both fitted pipelines:

* accept a raw feature frame
* apply the preprocessing they learned during training
* return class labels from `predict()`
* return class scores from `predict_proba()`

Those four properties are the evaluation contract. Once they hold, accuracy, precision, recall, F1, and ROC-AUC are the same functions of the same kinds of output. A second evaluator named after the model would copy those five calls without changing them.

Training is the contrast. `train_logistic` and `train_tree` exist as two functions because each has to construct a different unfitted pipeline before it fits. Evaluation never constructs a pipeline. Duplicating it into `evaluate_logistic()` and `evaluate_tree()` would suggest a difference that is not there, and would let the two scoring paths drift apart — a different positive-class column in one, a class-label ROC-AUC in the other — without any modelling reason.

Model-specific evaluation logic would become justified when the contract itself split. Examples that would count, none of which is present today:

* an estimator that does not expose `predict_proba()` and must be scored from `decision_function` or from labels alone
* a metric that is only meaningful for one of the models
* a different positive class, or a non-binary target
* a ranking or survival model whose outputs are not class probabilities

Until one of those differences exists, one evaluator is the smaller and safer surface.

---

## Responsibility Boundaries

| Responsibility | Where it lives | Status |
|----------------|----------------|--------|
| Feature schema — which columns are amounts, which are categories | `src/churner/schema/features.py` | Implemented |
| Data preparation — deterministic reshaping of the raw frame | `src/churner/data/prepare_modeling_data.py` | Implemented |
| Train/test splitting — establishing the evaluation boundary | `src/churner/training/split_modeling_data.py` | Implemented |
| Preprocessing — how each kind of feature is treated per model | `src/churner/preprocessing/preprocessors.py` | Implemented |
| Model pipeline — joining a preprocessor to a baseline estimator | `src/churner/preprocessing/pipelines.py` | Implemented |
| Training — fitting a pipeline on training records only | `src/churner/training/train.py` | Implemented |
| Evaluation — scoring a fitted pipeline on held-out records | `src/churner/evaluation/evaluate.py` | Implemented (this document) |

The division is deliberate. Preparation is deterministic and learns nothing, which is why it can precede the split. The split decides which records a later metric is allowed to be measured on. Training learns, and only from the training pair. Evaluation measures, and only on the test pair, using the fitted object as it stands.

The evaluator explicitly does **not**:

* load data
* split data
* train models
* fit preprocessing
* refit models
* tune hyperparameters
* mutate `X_test` or `y_test`
* persist or serialise models
* perform production monitoring

It accepts three arguments and returns five numbers. Anything that would change the fitted object, the test frames, or the meaning of the test partition belongs somewhere else.

---

## Leakage Prevention

The correct usage fits on training data only and evaluates on held-out data without refitting anything:

```python
# Correct
X, y = prepare_modeling_data(raw_df)
X_train, X_test, y_train, y_test = split_modeling_data(X, y)
pipeline = train_logistic(X_train, y_train)   # medians, scales, categories, coefficients
result = evaluate_model(pipeline, X_test, y_test)  # applies what fit learned; learns nothing
```

The same pattern holds for `train_tree`. In both cases the pipeline that evaluation receives is the pipeline that training fitted, and `X_test` is a frame that `fit` never saw.

The incorrect usage fits any preprocessing over data that includes the test set, or fits the pipeline on the full dataset and then "evaluates" on `X_test`:

```python
# Incorrect — leakage
preprocessor.fit(pd.concat([X_train, X_test]))   # statistics include test rows
# or, equivalently:
pipeline.fit(X_all, y_all)                       # then evaluate_model(pipeline, X_test, y_test)
```

The evaluator would still run. It would still return an `EvaluationResult`. The numbers would no longer be an estimate of performance on unseen data.

The learned preprocessing state is what leaks, even when the classifier itself never sees a test label:

| Learned state | What it would absorb from the test set |
|---------------|----------------------------------------|
| Imputation statistics | The test rows' medians and modal categories |
| Scaling parameters | The test rows' means and standard deviations |
| Category mappings | Categories present only in the test rows |

Each of these is information about the held-out data that would not have been available at the moment of training. Once it is folded into the transformations, the test score becomes partly a measurement of the test set against itself.

Evaluation's contribution to this boundary is negative: it does not offer a way to refit. It does not accept the training frames, it does not call `fit`, and it does not rebuild a preprocessor around `X_test`. The positive contribution remains with the caller and with training: fit once, on `X_train` and `y_train` only, and hand that object to `evaluate_model`.

---

## Testing Strategy

The evaluation implementation has a dedicated test suite at `tests/evaluation/test_evaluate.py`. The tests pin the public contract rather than the internals of sklearn's metric functions. Where a number is asserted, it is hand-calculated from a four-row scoring case defined in the test module, so a failure points at the evaluator using the wrong inputs — class labels where probabilities belong, for example — rather than at a figure copied from the implementation.

The suite covers:

* **`EvaluationResult` return type.** `evaluate_model()` hands back that dataclass and nothing else.
* **Five numeric metrics.** `accuracy`, `precision`, `recall`, `f1`, and `roc_auc` are all present and are Python `float` values.
* **Controlled hand-calculated metric case.** Predetermined labels and scores produce accuracy 0.50, precision 0.50, recall 0.50, F1 0.50, and ROC-AUC 0.75, each recoverable from the four-row confusion counts and pairwise ranking described in the test module.
* **ROC-AUC continuous-score behaviour.** The same fixture's class predictions, treated as 0/1 scores, would yield ROC-AUC 0.50. The evaluator returns 0.75, which is the ranking of the probabilities.
* **Both baseline pipelines.** A fitted logistic pipeline and a fitted decision tree pipeline are scored by the same `evaluate_model()`, on a raw held-out feature frame.
* **Direct fitted-pipeline usage.** Scoring calls `predict` and `predict_proba` on the object that was passed in, rather than on a rebuilt estimator or an extracted classifier.
* **No refitting.** `fit` is not called. After evaluation, the numerical imputer still holds the training medians and the classifier parameters are unchanged. The held-out amounts sit well away from those medians, so a preprocessor accidentally fitted on the test frame would not recover them.
* **No mutation of `X_test` / `y_test`.** A missing amount in the caller's test frame is filled inside the pipeline only; the caller's copy still records no amount. The label series is left as it was handed over.

| Suite | Result |
|-------|--------|
| Evaluation test suite (`test_evaluate.py`) | **16 passed** |
| Full test suite | **266 passed** |

Those are the results of the current implementation. No further evaluation behaviour has been measured, including performance of either baseline on the full Telco test partition.

---

## Design Decisions and Tradeoffs

### Frozen dataclass versus dictionary

The result is a frozen dataclass rather than a `dict`.

A dictionary would have been enough to carry five numbers, and it would have been easier to extend with an extra key. That openness is the problem. The evaluation contract is closed: five named metrics, each a `float`, and nothing else. A dataclass makes the names and the types part of the type of the object. Freezing it makes the numbers a record of one scoring rather than a mutable report.

*Alternative considered:* **a plain dictionary, or a mutable dataclass that also stored the predictions.** Rejected. Predictions and the test frames are inputs to the calculation, not part of the result. Storing them would invite later code to treat `EvaluationResult` as a container for the run, which is a different object with a different lifetime. A dictionary would accept that quietly. The frozen dataclass refuses it.

### One generic evaluator versus model-specific evaluators

`evaluate_model()` is shared across both baselines.

*Alternative considered:* **`evaluate_logistic()` and `evaluate_tree()`**, mirroring `train_logistic` and `train_tree`. Rejected because the symmetry is false. Training is model-specific: each function has to build a different pipeline. Evaluation is not: both fitted pipelines already satisfy `predict` and `predict_proba`. Two evaluators would duplicate the same five metric calls and create a place for them to diverge. The generic function is the smaller contract, and it is the one the tests exercise against both trainers.

### Evaluation accepts the fitted pipeline versus rebuilding preprocessing

The evaluator scores the object training returned.

*Alternative considered:* **rebuilding or re-fitting a preprocessor inside evaluation**, or extracting the classifier and transforming `X_test` by hand. This is the leakage path the rest of the architecture is arranged to close. It would also reintroduce the pairing problem the pipeline exists to remove: a classifier scored against statistics it was not trained with, or against statistics learned from the test set. Passing the fitted pipeline makes the correct pairing the only pairing.

### Current dependence on `predict_proba()`

ROC-AUC is computed from `predict_proba()`. Both current baselines expose it, so no adapter is introduced for `decision_function` or for estimators that only `predict`.

*Alternative considered:* **a small abstraction that accepted either probabilities or decision scores, or that fell back to class labels.** Rejected as premature. There are two known classifiers, both with `predict_proba()`. Building a fallback would either hide a real limitation — scoring labels as if they were ranks — or add branches that nothing in the project can take. The limitation is recorded below rather than papered over.

### Explicit positive label `"Yes"`

The positive class is a module-level constant, not inferred from `classes_[1]` and not passed in by the caller.

*Alternative considered:* **taking `pos_label` as an argument, or always using the last entry of `classes_`.** A caller-supplied label would suggest that evaluation is target-agnostic, which the rest of this project is not. Using `classes_[1]` would usually be `"Yes"` for this target, because sklearn sorts the labels, but it would couple the metric to an ordering rather than to the documented churn class. Looking `"Yes"` up in `classes_` keeps the name explicit and the column correct even if that order ever changed.

---

## Current Limitations

This is baseline evaluation. It scores one fitted binary classifier on one held-out partition and returns five numbers. The following are properties of the current implementation, not deferred features accidentally omitted from the code.

* **Binary classification is assumed.** The class-based metrics are computed with `pos_label="Yes"`. A third label, or a multiclass target, is outside the contract.
* **The positive label is fixed to `"Yes"`.** That is the documented churn class. It is not a parameter.
* **ROC-AUC currently depends on `predict_proba()`.** An estimator that cannot produce class probabilities cannot be scored by this function as written.
* **There is no threshold optimisation.** Accuracy, precision, recall, and F1 use the model's default class predictions. No cut-off other than that default is searched.
* **There is no probability calibration analysis.** `predict_proba()` values are used as ranking scores. They are not compared with observed frequencies.
* **There is no PR-AUC.** Precision-recall area is not computed, even though the target is imbalanced.
* **There is no production monitoring.** The function scores a batch of held-out rows in process. It does not watch a live model, drift, or a serving path.
* **There is no temporal evaluation.** The split that feeds this layer is a stratified random partition, not a time-based hold-out. Nothing here measures performance on a later period.

No claim is made about how either baseline performs on the Telco test set. That measurement is a use of this layer, not a result it currently reports.

---

## Future Evolution

The evaluation layer is one step of a larger system, and deliberately a narrow one. It ends at five metrics on a fitted pipeline.

Work that may follow, none of which is designed or implemented yet:

* threshold analysis — how precision and recall move as the cut-off moves
* confusion matrix reporting — the four counts behind the class-based metrics
* PR-AUC — a ranking summary that emphasises the positive class on an imbalanced target
* calibration analysis — whether `predict_proba()` scores match observed churn frequencies
* model comparison — a structured reading of two `EvaluationResult` values against the same test partition
* experiment tracking — recording which fitted object produced which metrics
* production monitoring — measuring a deployed model rather than a held-out batch

These are listed to place the current layer in context, not as a specification. Each will bring its own design decisions, and those decisions belong to the day they are made rather than being anticipated here. The commitment this layer does make to them is the boundary itself: whatever compares, tracks, or monitors a model will be working with a fitted pipeline that was scored on held-out data without being refitted, and with a result object that holds metrics rather than the run that produced them.
