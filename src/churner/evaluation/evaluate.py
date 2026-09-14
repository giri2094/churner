"""Evaluation of a fitted Telco churn classification pipeline.

This module is the measurement step that sits after training: it scores an
already-fitted pipeline on held-out test records and returns five baseline
metrics. The pipeline is used as it was fitted, so the preprocessing that
runs on ``X_test`` is the state learned during training.

There is one evaluator rather than one function per model, because both
baseline classifiers share the same evaluation contract.

Nothing here splits data, trains a model, refits preprocessing, or is
handed training records. The test set is an evaluation boundary, and it is
only read from.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# The data dictionary documents ``Churn`` as Yes/No, where "Yes" means the
# customer left. Precision, recall, F1, and ROC-AUC are reported for that
# class.
POSITIVE_LABEL = "Yes"


@dataclass(frozen=True)
class EvaluationResult:
    """Baseline classification metrics of one fitted model on a test set.

    Accuracy, precision, recall, and F1 come from the model's class
    predictions. ROC-AUC comes from its positive-class probability, not
    from those labels. The model, the test records, and the predictions
    themselves are not stored here.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def evaluate_model(
    fitted_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Score a fitted classification pipeline on held-out test records.

    Class predictions feed accuracy, precision, recall, and F1. ROC-AUC
    uses the probability of the positive class. The pipeline is not
    fitted, and the test frames are only read from.

    Parameters
    ----------
    fitted_model : Pipeline
        An already-fitted classification pipeline, ordinarily from
        ``train_logistic`` or ``train_tree``. Used as-is; not refitted.
    X_test : pd.DataFrame
        Held-out predictors, ordinarily from ``split_modeling_data``.
        Only read from.
    y_test : pd.Series
        Held-out labels aligned to ``X_test``. Only read from.

    Returns
    -------
    EvaluationResult
        The five baseline metrics computed on this test set.
    """
    class_predictions = fitted_model.predict(X_test)
    class_probabilities = fitted_model.predict_proba(X_test)
    positive_class_index = list(fitted_model.classes_).index(POSITIVE_LABEL)
    positive_scores = class_probabilities[:, positive_class_index]

    return EvaluationResult(
        accuracy=float(accuracy_score(y_test, class_predictions)),
        precision=float(
            precision_score(y_test, class_predictions, pos_label=POSITIVE_LABEL)
        ),
        recall=float(recall_score(y_test, class_predictions, pos_label=POSITIVE_LABEL)),
        f1=float(f1_score(y_test, class_predictions, pos_label=POSITIVE_LABEL)),
        # The score is P(Yes), not the predicted class, so ROC-AUC ranks
        # probabilities rather than 0/1 labels.
        roc_auc=float(roc_auc_score(y_test, positive_scores)),
    )
