"""Prediction from an already-fitted pipeline, one record per customer.

This module is the step that turns a fitted model into answers about named
customers. It is handed a pipeline that is already fitted and a version
string that identifies it, and it returns one result per row: the predicted
class, the probability that the customer churns, and the version of the
model that said so.

The model arrives at construction time rather than being loaded here. Which
artifact is in use, and when it is read from disk, is the caller's decision;
``load_model`` already answers it. A service that loaded its own model would
fix that choice for every caller and would reload it per process rather than
per deployment.

The identifier travels alongside the features rather than inside them. The
feature schema deliberately excludes ``customerID``, because it identifies a
customer rather than describing one, so the two arrive as separate arguments
and are paired by position. That pairing is also why input order is
preserved: results are returned in the order the rows were given, so a
caller can match them to what it sent without looking at the identifiers.

The positive class is found through ``classes_`` rather than assumed to sit
in a particular column of ``predict_proba``. Column order follows the sorted
class labels the estimator was fitted on, which is a property of the fitted
model and not of this module. Reading the index means a model fitted on
differently ordered labels reports the same probability rather than silently
reporting the complement of it.

Nothing here loads a model, preprocesses features, validates a request,
speaks HTTP, or decides what to do with a prediction.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

# The data dictionary documents ``Churn`` as Yes/No, where "Yes" means the
# customer left. ``churn_probability`` is the probability of that class.
POSITIVE_LABEL = "Yes"


@dataclass(frozen=True)
class PredictionResult:
    """One model's answer about one customer.

    ``churn_prediction`` is the class the model predicted, in the Yes/No
    vocabulary it was fitted on. ``churn_probability`` is the probability
    it assigned to ``"Yes"``, which is a separate quantity: a caller that
    applies its own cut-off reads the probability, not the class.

    ``model_version`` records which model produced the two, so a stored
    result stays attributable after the deployed model has moved on.
    """

    customer_id: str
    churn_prediction: str
    churn_probability: float
    model_version: str


class PredictionService:
    """Predicts churn for customers using one fitted pipeline.

    The service holds the model and the version string that names it, so
    a caller makes that pairing once and every result it produces carries
    it. The pipeline is used as it was fitted and is never refitted.
    """

    def __init__(self, fitted_model: Pipeline, model_version: str) -> None:
        """Bind a fitted pipeline to the version string that identifies it.

        Parameters
        ----------
        fitted_model : Pipeline
            An already-fitted classification pipeline, ordinarily from
            ``load_model``. Used as-is; not fitted or refitted here.
        model_version : str
            Identifies the model in the results it produces. Not read
            from the artifact, because an artifact does not know what it
            was deployed as.
        """
        self._fitted_model = fitted_model
        self._model_version = model_version

    def predict(
        self,
        customer_ids: Sequence[str],
        features: pd.DataFrame,
    ) -> list[PredictionResult]:
        """Predict churn for customers, one result per feature row.

        Classes come from ``predict`` and probabilities from
        ``predict_proba``, so the reported probability is the model's own
        score for ``"Yes"`` rather than something derived from the class.
        Results are returned in the order the rows were given, and the
        feature frame is only read from.

        Parameters
        ----------
        customer_ids : Sequence[str]
            Identifiers positionally aligned to the rows of ``features``.
            Only read from.
        features : pd.DataFrame
            Model features, in the raw form the pipeline was fitted on;
            its preprocessing is applied as part of predicting. Only read
            from.

        Returns
        -------
        list[PredictionResult]
            One result per row, in input order.

        Raises
        ------
        ValueError
            If ``customer_ids`` and ``features`` differ in length, since
            the two are paired by position and neither can say which
            pairing was intended.
        """
        if len(customer_ids) != len(features):
            raise ValueError(
                f"Got {len(customer_ids)} customer ids for {len(features)} "
                "feature rows. Ids and rows are paired by position, so a "
                "prediction cannot be attributed to a customer."
            )

        predicted_classes = self._fitted_model.predict(features)
        class_probabilities = self._fitted_model.predict_proba(features)
        positive_class_index = list(self._fitted_model.classes_).index(POSITIVE_LABEL)
        positive_scores = class_probabilities[:, positive_class_index]

        return [
            PredictionResult(
                customer_id=customer_id,
                churn_prediction=str(predicted_class),
                churn_probability=float(positive_score),
                model_version=self._model_version,
            )
            for customer_id, predicted_class, positive_score in zip(
                customer_ids, predicted_classes, positive_scores
            )
        ]
