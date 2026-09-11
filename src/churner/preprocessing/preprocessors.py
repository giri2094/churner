"""Model-specific preprocessing of the prepared Telco churn feature frame.

This module builds the stage that sits between the train/test split and a
model: it assembles the transformers that turn the predictors handed back by
``prepare_modeling_data`` into the numeric matrix an estimator can be fitted
on. Two are offered, one for Logistic Regression and one for tree models,
because the two ask different things of their inputs.

Every transformation assembled here learns something from the data it is fitted
on: a median, a mean, a standard deviation, the set of categories a column
holds. That is what keeps the stage on this side of the split. Nothing is
fitted while a preprocessor is being built, and none of these functions is
handed any data; the returned object is unfitted, and what it learns is decided
by whatever frame ``fit`` is later called with. Fitting it on training records
alone is what keeps the test records out of it, and that remains the caller's
responsibility: a ``ColumnTransformer`` will fit on whatever it is given.

The two preprocessors differ in one step. Logistic Regression fits coefficients
by optimising over a weighted sum of the features, so a feature recorded in
thousands and one recorded in tens do not enter that sum on comparable terms;
its numerical branch is standardised for that reason. A decision tree splits one
feature at a time on a threshold, and rescaling a feature moves the threshold
with it, so the tree branch leaves the amounts as they are. Both are built
explicitly and separately rather than through one function with a switch, so
which model gets which treatment is readable at the call site.

No feature is created, combined, or dropped here, and no model is trained. The
column lists come from ``churner.schema.features`` so that this module decides
how each kind of feature is treated without also deciding which columns are of
which kind.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The canonical schema, so the split between amounts and categories is stated
# once for the project rather than restated per model.
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# --- Branch names ---
# The names a fitted ``ColumnTransformer`` reports its branches under, which is
# how a caller reaches a fitted transformer to inspect what it learned.
NUMERICAL_BRANCH = "numerical"
CATEGORICAL_BRANCH = "categorical"

# --- Step names ---
IMPUTER_STEP = "imputer"
SCALER_STEP = "scaler"
ENCODER_STEP = "encoder"

# --- Imputation ---
# The median stands in for a missing amount: it is a value the column actually
# takes, and unlike the mean it is not pulled by the long right tail the
# charges columns carry. The most frequent value stands in for a missing
# category, there being no middle of an unordered set to fall back on. Both
# statistics are learned during ``fit``, from the records ``fit`` is given.
NUMERICAL_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"

# --- Categorical encoding ---
# One-hot encoding represents a nominal column as one indicator per category,
# which is what keeps the model from reading an order into values that have
# none.
#
# Dropping the first category leaves one category represented by every
# indicator being zero. That category becomes the reference the others are
# measured against, and its removal is what stops the indicators of a column
# from summing to a constant and being perfectly collinear with a model's
# intercept.
#
# Ignoring an unknown category gives ``transform`` a defined behaviour when it
# meets a value that was not present at fit time: the indicators for that
# column are all zero. That is the same encoding the dropped reference category
# receives, so an unseen value is not distinguishable from the reference one.
DROPPED_CATEGORY = "first"
UNKNOWN_CATEGORY_HANDLING = "ignore"


def build_categorical_branch() -> Pipeline:
    """Assemble the categorical treatment both preprocessors share.

    Missing categories are filled with the most frequent one before encoding,
    since an unfilled missing value would otherwise become a category of its
    own. Neither model has a reason to treat categories differently from the
    other, so both branches are built from this one definition.

    A new pipeline is built on every call, so no fitted state can be shared
    between the preprocessors that hold it.

    Returns
    -------
    Pipeline
        An unfitted imputation-then-encoding pipeline.
    """
    return Pipeline(
        steps=[
            (IMPUTER_STEP, SimpleImputer(strategy=CATEGORICAL_IMPUTATION_STRATEGY)),
            (
                ENCODER_STEP,
                OneHotEncoder(
                    drop=DROPPED_CATEGORY,
                    handle_unknown=UNKNOWN_CATEGORY_HANDLING,
                ),
            ),
        ]
    )


def create_logistic_preprocessor() -> ColumnTransformer:
    """Build the unfitted preprocessor for Logistic Regression.

    Amounts are imputed with the median and then standardised, so each is
    expressed in standard deviations from its own mean and the features enter
    the model's weighted sum on comparable terms. The mean and standard
    deviation come from the records the preprocessor is fitted on.
    ``StandardScaler`` is a plain baseline choice and nothing more: both
    statistics it learns are affected by extreme values, so an outlier in a
    column shifts and widens the scale every value in it is measured against.

    Categories are imputed and one-hot encoded, as described by
    ``build_categorical_branch``.

    Returns
    -------
    ColumnTransformer
        An unfitted transformer holding a ``numerical`` and a ``categorical``
        branch. Nothing has been learned from any data yet; that happens when
        the caller fits it, on the records the caller fits it with.
    """
    numerical_branch = Pipeline(
        steps=[
            (IMPUTER_STEP, SimpleImputer(strategy=NUMERICAL_IMPUTATION_STRATEGY)),
            (SCALER_STEP, StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (NUMERICAL_BRANCH, numerical_branch, list(NUMERICAL_FEATURES)),
            (CATEGORICAL_BRANCH, build_categorical_branch(), list(CATEGORICAL_FEATURES)),
        ]
    )


def create_tree_preprocessor() -> ColumnTransformer:
    """Build the unfitted preprocessor for tree-based models.

    Amounts are imputed with the median and left on the scale they were
    recorded on. A tree splits a single feature at a threshold, and a monotonic
    rescaling moves every candidate threshold along with the values, so
    standardising would change the numbers a split is expressed in without
    changing which splits are available.

    Categories are imputed and one-hot encoded, as described by
    ``build_categorical_branch``.

    Returns
    -------
    ColumnTransformer
        An unfitted transformer holding a ``numerical`` and a ``categorical``
        branch. Nothing has been learned from any data yet; that happens when
        the caller fits it, on the records the caller fits it with.
    """
    return ColumnTransformer(
        transformers=[
            (
                NUMERICAL_BRANCH,
                SimpleImputer(strategy=NUMERICAL_IMPUTATION_STRATEGY),
                list(NUMERICAL_FEATURES),
            ),
            (CATEGORICAL_BRANCH, build_categorical_branch(), list(CATEGORICAL_FEATURES)),
        ]
    )
