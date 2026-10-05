"""Translation of a validated request into the frame a pipeline is given.

This module is the one step between the two representations the serving
path uses: a ``CustomerFeatures`` object, which is what arriving JSON has
been validated into, and a one-row ``DataFrame``, which is what a fitted
pipeline is fitted to be handed. Nothing is decided here. The values that
go in come out, under the names the model knows them by, in the order the
project states them.

The columns are read from ``churner.schema.features`` rather than from the
request model's own field order. Both say the same thing today, and that
is the point: one of them is the project's definition of what the model's
features are, and reading it means a reordered schema class cannot quietly
hand a pipeline its columns in a different order. Selecting by name also
makes a feature the request stops carrying a failure here, where the
column is assembled, instead of a silently absent column later.

Nothing is imputed, scaled, or encoded on the way through. Every one of
those is state the pipeline learned from the training records -- a median,
a mean and standard deviation, a set of categories -- and applying a
second version of it at the boundary would mean transforming twice, with
the second transformation fitted on nothing. A missing ``TotalCharges``
therefore stays missing, and a category the encoder never saw stays
spelled as the caller spelled it; the pipeline's imputer and its
``handle_unknown="ignore"`` encoder are what answer for both.

The frame is new on every call and shares nothing with the object it was
built from, so a caller may transform or annotate what it is given without
reaching back into the request it came from.

Nothing here predicts, loads a model, reaches a prediction service, speaks
HTTP, or declares a route.
"""

import pandas as pd

from churner.api.schemas import CustomerFeatures
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# --- The columns a pipeline is fitted to select ---
# The canonical schema, amounts first and then categories, which is the
# order ``churner.schema.features`` states them in. Held as a tuple so the
# module's own definition cannot be edited by a caller that received it.
MODEL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def features_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    """Build the one-row feature frame that a fitted pipeline accepts.

    This is an API-boundary translation and nothing more: the validated
    Pydantic object is restated as the pandas representation the serving
    layer works in. No value is imputed, scaled, encoded, or otherwise
    transformed, because every such step is fitted state the pipeline
    already holds and applies when it predicts.

    Parameters
    ----------
    features : CustomerFeatures
        One customer's validated model features. Only read from; never
        modified.

    Returns
    -------
    pd.DataFrame
        A newly built frame of exactly one row, holding the canonical
        feature columns in canonical order and the values ``features``
        stated. It shares no storage with ``features``, so the caller may
        change it freely. The identifier and the target are absent, the
        canonical schema naming neither as a feature.
    """
    stated_features = features.model_dump()

    return pd.DataFrame(
        [{column: stated_features[column] for column in MODEL_FEATURE_COLUMNS}],
        columns=list(MODEL_FEATURE_COLUMNS),
    )
