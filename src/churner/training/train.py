"""Fitting of the baseline Telco churn model pipelines.

This module is the training step that sits after the train/test split: it
constructs a model-specific pipeline and fits it on training records alone.
The pipeline factories own how each baseline is assembled; this module owns
only the fit. Evaluation, persistence, and tuning belong elsewhere.

There is one function per model rather than one parameterised trainer,
because the two pipelines are already distinct objects and a generic
``train_model`` would add an indirection that two known baselines do not
need.

Nothing here splits data, scores a model, or writes one to disk. Each
function is handed ``X_train`` and ``y_train`` only, and it returns the
fitted pipeline.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from churner.preprocessing.pipelines import (
    create_logistic_pipeline,
    create_tree_pipeline,
)


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the baseline Logistic Regression pipeline on training records.

    A new unfitted pipeline is built, then fitted on ``X_train`` and
    ``y_train`` only. The test partition is not an argument, so it cannot
    enter the fit.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors, ordinarily from ``split_modeling_data``.
        Only read from.
    y_train : pd.Series
        Training labels aligned to ``X_train``. Only read from.

    Returns
    -------
    Pipeline
        The logistic pipeline after ``fit``. Both the preprocessor and the
        classifier have learned from the training records.
    """
    pipeline = create_logistic_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def train_tree(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the baseline decision tree pipeline on training records.

    A new unfitted pipeline is built, then fitted on ``X_train`` and
    ``y_train`` only. The test partition is not an argument, so it cannot
    enter the fit.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors, ordinarily from ``split_modeling_data``.
        Only read from.
    y_train : pd.Series
        Training labels aligned to ``X_train``. Only read from.

    Returns
    -------
    Pipeline
        The tree pipeline after ``fit``. Both the preprocessor and the
        classifier have learned from the training records.
    """
    pipeline = create_tree_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline
