# Model Pipeline Architecture

**Project:** `churner`
**Implementation:** `src/churner/preprocessing/pipelines.py`
**Tests:** `tests/preprocessing/test_pipelines.py`

---

## Purpose

This document explains the engineering reasoning behind the baseline sklearn model pipelines, not merely their contents. It describes one specific layer of the ML workflow: the boundary where a model-specific preprocessor is combined with a baseline estimator to form a single fittable object.

It exists because the value of a `Pipeline` in this project is architectural rather than algorithmic. The pipeline does not improve accuracy; it fixes *where* preprocessing state is learned and *when* it is applied. Those are properties that are easy to get wrong silently, and hard to notice afterwards, so the reasoning is recorded here rather than left implicit in the code.

The layer documented here sits after deterministic data preparation and before training and evaluation:

```text
raw CSV
    → prepare_modeling_data()      (deterministic, already implemented)
    → train/test split             (not yet implemented)
    → model pipeline               (this document)
    → training / evaluation        (not yet implemented)
```

---

## Current Architecture

A pipeline in this project spans the path from a raw feature DataFrame to a prediction:

```text
raw feature DataFrame
        ↓
model-specific preprocessor      (step name: "preprocessor")
        ↓
estimator                        (step name: "classifier")
        ↓
prediction
```

The input is the predictor frame that `prepare_modeling_data` returns: `TotalCharges` converted to a numeric amount, `customerID` and `Churn` removed, missing amounts preserved as missing. No further manual transformation is expected of the caller.

The current implementation provides exactly two public factory functions:

| Factory | Preprocessor | Estimator |
|---------|--------------|-----------|
| `create_logistic_pipeline()` | `create_logistic_preprocessor()` | `LogisticRegression(max_iter=1000)` |
| `create_tree_pipeline()` | `create_tree_preprocessor()` | `DecisionTreeClassifier(random_state=42)` |

Both return an **unfitted** `sklearn.pipeline.Pipeline` with two steps, named `preprocessor` and `classifier`. Those names are part of the module's public contract, because they are how a caller reaches the fitted preprocessing or the fitted model afterwards. They are defined once as module-level constants (`PREPROCESSOR_STEP`, `CLASSIFIER_STEP`), as are the two estimator settings (`MAX_ITERATIONS = 1000`, `RANDOM_STATE = 42`).

The preprocessors themselves are not defined here. They come from `src/churner/preprocessing/pipelines.py`'s dependency, `src/churner/preprocessing/preprocessors.py`, which in turn reads the column groups from `src/churner/schema/features.py`. The pipeline module therefore contains no feature definitions of its own.

---

## Train/Test Boundary

Train/test splitting happens **outside** the sklearn `Pipeline`. The pipeline is the object that gets fitted; it is not the thing that decides which records it is fitted on. No splitting code exists in the project yet, so today this boundary is a constraint on future work rather than an implemented component.

Three reasons place the split outside and before the pipeline:

**Splitting establishes the evaluation boundary.** Deciding which records are used to measure a model is an evaluation decision, not a transformation. It determines what any reported metric actually means. Burying it inside an object whose job is to transform and predict would make the most consequential decision in the workflow the least visible.

**Learned preprocessing state must be derived from training data only.** Every transformation in both preprocessors learns something from the data it is fitted on — a median, a mean, a standard deviation, a set of categories. If the split does not happen first, there is no defined notion of "training data" for those statistics to come from.

**The test set must remain unseen during fitting.** A test set is only a test set to the extent that nothing fitted has been influenced by it. Once a median or a category mapping has been computed over the test rows, the test score is no longer an estimate of performance on unseen data, even if the estimator itself never saw those rows.

The practical consequence is that the split must precede the single `fit` call, and the pipeline must be fitted on the training portion alone.

---

## Why Use sklearn Pipeline?

**Preprocessing and estimator become one fitted object.** After `fit`, a single object holds both the learned preprocessing state and the learned model parameters. There is no separate transformer to keep track of, store alongside the model, or accidentally pair with the wrong one.

**`fit()` learns preprocessing state and estimator parameters from the same training data in one call.** The imputation statistics, the scaling parameters, the category mappings, and the model coefficients or tree structure all come from one frame in one call. They cannot drift out of step with each other.

**`predict()` applies the already-fitted preprocessing before prediction.** The caller passes a raw feature frame of the same shape it trained on. The pipeline imputes, scales, and encodes it using what `fit` learned, then hands the resulting matrix to the estimator. Nothing is refitted.

**It reduces the risk of applying transformations inconsistently.** The alternative is a manual sequence the caller must reproduce identically at training time and at prediction time. Every such reproduction is an opportunity for the two to diverge — a forgotten step, a different order, a transformer refitted by accident.

**It keeps training and inference transformation logic aligned.** There is one definition of the transformations, used by both paths, which matters increasingly as later layers (such as an inference service) are added.

### An important precision

A `Pipeline` does **not** magically prevent data leakage.

`Pipeline.fit` fits every step on whatever frame it is handed. If it is handed the full dataset, it will happily learn its medians, scales, and categories from the full dataset, including the test rows. Leakage prevention depends entirely on the pipeline being fitted only on training data.

What the pipeline provides is narrower and still valuable: it makes fitting preprocessing on one frame and applying it to another the *default and natural* usage, rather than something the caller must remember to arrange. The guarantee comes from how it is called; the pipeline only makes the correct call the easy one.

This distinction is stated explicitly in both the module docstring and the test suite, so that the object is not mistaken for a safeguard it is not.

---

## Fit vs Transform

The four operations differ in whether they *learn* state, *apply* state, or both.

| Operation | Learns from data? | Applies learned state? | Used on |
|-----------|-------------------|------------------------|---------|
| `fit` | Yes | — | Training data only |
| `transform` | No | Yes | Any frame, after fitting |
| `fit_transform` | Yes | Yes | Training data only |
| `predict` | No | Yes, then predicts | Any frame, after fitting |

In this project, the state learned during `fit` is concrete:

| Transformation | State learned during `fit` | Applied during `transform` |
|----------------|----------------------------|----------------------------|
| `SimpleImputer(strategy="median")` | One median per numerical column | Missing amounts filled with that median |
| `SimpleImputer(strategy="most_frequent")` | The modal value per categorical column | Missing categories filled with that value |
| `StandardScaler` | One mean and one standard deviation per numerical column | Each amount re-expressed in standard deviations from that mean |
| `OneHotEncoder` | The set of categories seen per categorical column | One indicator per category, with the first dropped and unknown values encoded as all zeros |

Within a pipeline these compose as follows. `Pipeline.fit(X_train, y_train)` calls `fit_transform` on the preprocessor — which both learns the statistics above and produces the training matrix — and then `fit` on the classifier using that matrix. `Pipeline.predict(X)` calls `transform` on the preprocessor, which applies the existing statistics without revisiting them, and then `predict` on the classifier.

Two consequences are worth stating plainly. First, `fit_transform` belongs on training data only; calling it on a test frame would refit the statistics on that frame, which is precisely the boundary violation the architecture is arranged to avoid. Second, the estimator is the last step and has no `transform`, which is why a pipeline ending in a classifier exposes `predict` but not `transform`.

The target is not transformed. `y` is passed through to the estimator as given, so the recorded `Churn` labels (`"Yes"` / `"No"`) are what the model learns and what `predict` returns. Target encoding is not part of this layer.

---

## Logistic Regression Pipeline

Current structure, as implemented:

```text
create_logistic_pipeline()
│
├── "preprocessor"  →  create_logistic_preprocessor()
│     ├── numerical branch    (tenure, MonthlyCharges, TotalCharges)
│     │     └── SimpleImputer(strategy="median")
│     │           └── StandardScaler()
│     │
│     └── categorical branch  (16 categorical features)
│           └── SimpleImputer(strategy="most_frequent")
│                 └── OneHotEncoder(drop="first", handle_unknown="ignore")
│
└── "classifier"    →  LogisticRegression(max_iter=1000)
```

### Why scaling is appropriate here

Logistic Regression fits coefficients by optimising over a weighted sum of the features. The features therefore enter the model on whatever scale they were recorded on, and in this dataset those scales differ by orders of magnitude: `tenure` is measured in months up to 72, while `TotalCharges` reaches the thousands. Without standardisation, a one-unit change in a currency column and a one-unit change in a tenure column are treated as comparable movements when they are not.

Standardisation has two practical effects. The fitted coefficients become comparable to one another, because each is expressed per standard deviation of its own feature rather than per raw unit. And the optimiser works over a better-conditioned surface, which is the sense in which scaled inputs tend to converge more readily.

`StandardScaler` is a plain baseline choice, not an optimal one. Both statistics it learns — the mean and the standard deviation — are themselves sensitive to extreme values, so an outlier in a column shifts and widens the scale every other value in that column is measured against.

---

## Decision Tree Pipeline

Current structure, as implemented:

```text
create_tree_pipeline()
│
├── "preprocessor"  →  create_tree_preprocessor()
│     ├── numerical branch    (tenure, MonthlyCharges, TotalCharges)
│     │     └── SimpleImputer(strategy="median")
│     │
│     └── categorical branch  (16 categorical features)
│           └── SimpleImputer(strategy="most_frequent")
│                 └── OneHotEncoder(drop="first", handle_unknown="ignore")
│
└── "classifier"    →  DecisionTreeClassifier(random_state=42)
```

The single difference from the logistic pipeline is the absence of `StandardScaler`. Imputation is still required, because the estimator cannot be fitted on missing values.

### Why numerical scaling is not required here

A decision tree splits one feature at a time at a threshold, and it evaluates candidate thresholds by the ordering of the values rather than by their magnitudes. A monotonic rescaling moves every candidate threshold along with the values it is derived from, so the same partitions of the data remain available. Standardising would change the numbers a split is *expressed* in without changing which splits can be chosen.

Applying a scaler here would therefore add learned state — a mean and a standard deviation per column, which must be stored, fitted correctly, and carried into inference — in exchange for no change in the model's behaviour. It is omitted for that reason.

### Estimator configuration

Only `random_state=42` is passed. Every other parameter keeps its sklearn default; in particular, this project sets no explicit `max_depth`, `min_samples_split`, or `min_samples_leaf` value. The baseline is deliberately the estimator's own default growth behaviour, and the test suite pins those three parameters at their defaults so that a growth limit cannot be introduced quietly as part of unrelated work.

No claim is made here about how deep the resulting tree becomes or how its leaves turn out. That depends on the data it is fitted on and has not been measured, since no training run has been performed yet.

---

## Responsibility Boundaries

| Responsibility | Where it lives | Status |
|----------------|----------------|--------|
| Feature schema — which columns are amounts, which are categories | `src/churner/schema/features.py` | Implemented |
| Data preparation — deterministic reshaping of the raw frame | `src/churner/data/prepare_modeling_data.py` | Implemented |
| Train/test splitting — establishing the evaluation boundary | Outside the pipeline; the caller's responsibility | Not yet implemented |
| Preprocessing — how each kind of feature is treated per model | `src/churner/preprocessing/preprocessors.py` | Implemented |
| Model pipeline — joining a preprocessor to a baseline estimator | `src/churner/preprocessing/pipelines.py` | Implemented |
| Training and evaluation — fitting and measuring models | Outside this layer | Not yet implemented |

The division is deliberate. Data preparation is deterministic and learns nothing, which is why it can safely precede the split. Preprocessing learns from data, which is why it must follow the split. The pipeline decides how those two are wired to an estimator, and nothing more.

The pipeline factories explicitly do **not**:

* load data
* split data
* fit models at construction time
* transform data
* evaluate models
* save or serialise models
* perform predictions
* perform hyperparameter tuning
* define feature columns
* contain training workflows

Each factory constructs two objects and returns them wrapped in a `Pipeline`. Neither function accepts any data, and nothing in either is fitted at construction time.

---

## Design Decisions and Tradeoffs

### Separate pipelines per model

Two explicit factories exist rather than one configurable builder. The two models differ in both halves — a different preprocessor and a different estimator — so a single parameterised function would have to switch on the model in two places. With separate factories, which model receives which treatment is readable at the call site and at the definition.

*Alternative considered:* **a highly generic pipeline factory**, for example one accepting an estimator and a preprocessor as arguments, or a registry keyed by model name. Rejected as premature. There are two models, both known, and a generic builder would add indirection whose only justification would be a third model that does not exist. The abstraction can be introduced later, when there is evidence about what actually varies.

### Model-specific preprocessing is retained

The two preprocessors are kept distinct rather than unified.

*Alternative considered:* **one generic preprocessor for every model.** This would mean either scaling for the tree as well, adding learned state that changes nothing about its behaviour, or dropping scaling for Logistic Regression, which needs it. Either way, a shared preprocessor would be wrong for one of the two models. The preprocessing requirement genuinely differs, so the code reflects that difference.

Note that the categorical treatment *is* shared, because neither model has a reason to treat categories differently. It is defined once in `preprocessors.py` and built fresh per pipeline.

### The split remains outside the pipeline

*Alternative considered:* **putting train/test splitting inside the pipeline.** Rejected because it conflates two different kinds of operation. A pipeline step is expected to transform every row it is given; a split partitions rows and routes them differently, and is not reusable at inference time, where there is no test set at all. Keeping the split outside also leaves the pipeline usable as a unit by cross-validation utilities, which need to refit the whole object per fold.

### Factories return unfitted objects

A factory that returned a fitted pipeline would have to be handed data, and would carry statistics learned from whatever data built it. Any subsequent evaluation would then be measuring a model whose preprocessing had seen an unknown set of records. Returning unfitted objects keeps the decision about what to fit on with the caller, where the split lives. The test suite asserts this directly, including the absence of fitted attributes on the nested transformers.

Two calls to the same factory also share no objects, so fitting one pipeline cannot affect another.

### `max_iter=1000` is a convergence allowance

Logistic Regression is fitted by an iterative optimiser that stops either when it converges or when it exhausts its iteration budget. The sklearn default of 100 can be the limit reached first, in which case the fit stops early and warns.

Raising the limit to 1000 gives the optimiser more room to converge. It does not guarantee convergence, and it does not change what the model is fitting — the objective, the regularisation, and every other setting are untouched. It changes only how long the optimiser is permitted to work. That is why it is treated as a convergence allowance rather than as tuning.

*Alternative considered:* leaving the default and suppressing the resulting convergence warnings. Rejected: the warning would be reporting a real condition, and silencing it would discard information rather than address it.

### `random_state=42` provides reproducibility

A decision tree considers features in a shuffled order, so equally good candidate splits are resolved by a random draw. Without a fixed seed, two fits on identical data can produce different trees and therefore different reported numbers. Fixing the seed makes the fitted tree, and everything measured from it, reproducible across runs. The specific value carries no meaning.

### Why this design overall

*Alternative considered:* **manually preprocessing and then training separately** — fitting a `ColumnTransformer`, transforming `X_train` and `X_test` by hand, and fitting the estimator on the resulting matrices. This works, and it is what the pipeline is composed of. It was rejected as the primary interface because it requires the caller to reproduce the same transformation sequence at every point of use, keep the fitted transformer paired with the correct model, and remember never to refit on test data. Each of those is a step that can be omitted without any error being raised. The pipeline removes the opportunity rather than relying on discipline.

The overall preference throughout is for a minimal, explicit design that states the boundaries clearly and leaves later decisions to the stage that has the evidence to make them.

---

## Leakage Prevention

The correct usage fits on training data only and predicts for held-out data without refitting anything:

```python
# Correct
pipeline = create_logistic_pipeline()
pipeline.fit(X_train, y_train)      # medians, scales, categories, coefficients
predictions = pipeline.predict(X_test)   # applies what fit learned; learns nothing
```

The incorrect usage fits any preprocessing over data that includes the test set:

```python
# Incorrect — leakage
preprocessor.fit(pd.concat([X_train, X_test]))   # statistics include test rows
# or, equivalently:
pipeline.fit(X_all, y_all)                       # then "evaluating" on part of X_all
```

The learned preprocessing state is what leaks, even when the estimator itself never sees a test label:

| Learned state | What it would absorb from the test set |
|---------------|----------------------------------------|
| Imputation statistics | The test rows' medians and modal categories |
| Scaling parameters | The test rows' means and standard deviations |
| Category mappings | Categories present only in the test rows |

Each of these is information about the held-out data that would not have been available at the moment of training. Once it is folded into the transformations, the test score stops being an estimate of performance on unseen data and becomes partly a measurement of the test set against itself.

The same principle applies to cross-validation: the pipeline should be passed to the cross-validator as a whole, so that it is refitted on each fold's training portion, rather than being fitted once over all folds beforehand.

---

## Testing Status

The pipeline implementation has a dedicated test suite at `tests/preprocessing/test_pipelines.py`, containing 26 test functions covering factory return types, step names, which preprocessor and estimator each pipeline received, estimator configuration, unfitted state, instance independence, single-`fit` behaviour, prediction from raw feature frames, and immutability of the caller's data.

| Suite | Result |
|-------|--------|
| Existing regression suite (excluding `test_pipelines.py`) | **182 passed** |
| Pipeline test suite (`test_pipelines.py`) | **Blocked during collection** |

The pipeline tests could not execute. Importing `sklearn.tree` fails on this machine because Windows Application Control blocks the compiled `sklearn.tree._tree` DLL:

```text
ImportError: DLL load failed while importing _tree:
An Application Control policy has blocked this file.
```

Because `DecisionTreeClassifier` is imported at module scope in both `pipelines.py` and the test module, the failure occurs during pytest collection, before any individual test is selected. There is consequently no subset of the pipeline tests that can be run in isolation. The block is confined to `sklearn.tree`; the other sklearn modules this project uses import successfully.

Three statements about this, precisely:

* This is an **environment limitation, not a confirmed implementation failure**. The tests have not passed, and they have not failed on their assertions either — they have not run.
* The 182 passing tests are the pre-existing suite, unchanged by this work, confirming that the new module introduced no regression in the code that can be exercised.
* **No Windows security policy change, scikit-learn reinstall, or virtual-environment recreation was performed** in order to force the tests to pass. Weakening a host security control to turn a suite green would be the wrong trade, and the resulting green would mean less than the block does.

The correct resolution is to run the suite in an environment where the DLL is permitted, or to have the file allowed through the normal administrative process. Until then, the pipeline test results are genuinely unknown and are recorded as such.

---

## Future Evolution

The sklearn model pipeline is one layer of a larger system, and deliberately a narrow one. It ends at an unfitted object that can be fitted and can predict.

Layers that may follow, none of which is designed or implemented yet:

* a training workflow, including the train/test split this document treats as an external boundary
* model evaluation and metric reporting
* model persistence
* an inference service
* a FastAPI application
* containerisation
* deployment
* CI/CD

These are listed to place the current layer in context, not as a specification. Each will bring its own design decisions, and those decisions belong to the day they are made rather than being anticipated here. The one commitment this layer does make to them is the boundary itself: whatever trains, evaluates, persists, or serves a model will be working with a pipeline object that carries its preprocessing with it, and will be responsible for fitting it on training data alone.
