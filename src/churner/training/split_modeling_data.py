"""Train/test split of modelling-ready Telco churn data.

This module performs the one split that stands between prepared modelling
data and any model that will be fitted on it. The split is an evaluation
boundary, not a preprocessing step: it decides which records a later metric
is allowed to be measured on, and it learns nothing from the values those
records hold.

It lives outside the sklearn ``Pipeline`` for that reason. A pipeline
transforms every row it is given; a split partitions rows and routes them
differently, and it has no analogue at inference time, where there is no
test set. Performing the split once, here, is also what lets both baseline
models be trained and compared on exactly the same partition.

The inputs are the ``X`` and ``y`` that ``prepare_modeling_data`` returns.
Nothing is imputed, encoded, scaled, fitted, or trained, and no pipeline is
constructed. The four objects handed back are the only product.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Split configuration ---
# The test portion is held out for evaluation. Both baseline models are
# trained and measured on this same partition, which is why the split is
# made once and outside either pipeline.
TEST_SIZE = 0.2

# A fixed seed makes the partition the same on every run, so a later
# measurement can be compared to an earlier one.
RANDOM_STATE = 42


def split_modeling_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition modelling data into a training set and a held-out test set.

    The split is stratified on ``y`` so each class keeps its share of both
    partitions, and it is seeded so the same records produce the same
    partition on every call. Nothing is fitted, transformed, or trained:
    the four objects returned are slices of the inputs, and they are what
    later stages fit on and evaluate against.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor frame, ordinarily the features returned by
        ``prepare_modeling_data``. Only read from.
    y : pd.Series
        Target labels aligned to ``X``, ordinarily the ``Churn`` series
        returned by ``prepare_modeling_data``. Only read from.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``X_train``, ``X_test``, ``y_train``, ``y_test``. The training
        pair holds 80% of the rows and the test pair the remaining 20%,
        with both pairs sharing an index so each row of predictors still
        lines up with its own outcome.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
