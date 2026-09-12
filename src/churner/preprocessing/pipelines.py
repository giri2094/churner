"""Baseline model pipelines for the Telco churn predictors.

Each factory joins a model-specific preprocessor to the estimator it was built
for, so that one object spans the path from a raw feature frame to a prediction:

    raw feature DataFrame -> preprocessing -> estimator -> prediction

One ``fit`` then fits both stages on the same records, and ``predict`` applies
the preprocessing that fit learned, so a caller never transforms a frame by
hand. It is also what lets a splitter or a cross-validator refit the whole
pipeline per fold. The pipeline does not enforce that by itself: fitting it on
training records alone remains the caller's responsibility.

There is one factory per model rather than one parameterised builder, because
the two differ in both halves. Both estimators are baselines, left at their
defaults apart from the two constants below, and are here to be measured
against. Nothing here is fitted or handed any data, none of it splits, trains,
evaluates, or saves anything, and the feature columns belong to the
preprocessor rather than to this module.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from churner.preprocessing.preprocessors import (
    create_logistic_preprocessor,
    create_tree_preprocessor,
)

# --- Step names ---
# How a caller reaches the fitted preprocessing or the fitted model.
PREPROCESSOR_STEP = "preprocessor"
CLASSIFIER_STEP = "classifier"

# --- Iteration limit ---
# Logistic Regression is fitted by an iterative optimiser that stops either when
# it converges or when it runs out of iterations, and the default limit of 100
# can be the one reached first. A higher limit gives the optimiser more room to
# converge; it does not guarantee that it will. It is not a tuned parameter.
MAX_ITERATIONS = 1000

# --- Reproducibility ---
# A decision tree considers the features in a shuffled order, so equally good
# candidate splits are resolved by a random draw. A fixed seed makes the fitted
# tree, and every number reported off it, the same on every run.
RANDOM_STATE = 42


def create_logistic_pipeline() -> Pipeline:
    """Build the unfitted baseline Logistic Regression pipeline.

    The logistic preprocessor imputes and standardises the amounts and one-hot
    encodes the categories, and the estimator is fitted on what it produces.
    Apart from the raised iteration limit the estimator keeps its defaults,
    regularisation included, so this stands as a baseline rather than a tuned
    model.

    Returns
    -------
    Pipeline
        An unfitted ``preprocessor``-then-``classifier`` pipeline. Both steps
        are new objects and neither has learned anything yet.
    """
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, create_logistic_preprocessor()),
            (CLASSIFIER_STEP, LogisticRegression(max_iter=MAX_ITERATIONS)),
        ]
    )


def create_tree_pipeline() -> Pipeline:
    """Build the unfitted baseline decision tree pipeline.

    The tree preprocessor imputes the amounts and leaves them on the scale they
    were recorded on, since a split threshold moves with any rescaling. Apart
    from the fixed seed the estimator keeps its defaults: no ``max_depth``,
    ``min_samples_split``, or ``min_samples_leaf`` restriction beyond what
    sklearn already applies. That unrestricted baseline is what any later limit
    on the tree's growth has to be argued against.

    Returns
    -------
    Pipeline
        An unfitted ``preprocessor``-then-``classifier`` pipeline. Both steps
        are new objects and neither has learned anything yet.
    """
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, create_tree_preprocessor()),
            (CLASSIFIER_STEP, DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
