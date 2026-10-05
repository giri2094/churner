"""Unit tests for the step between a validated request and a feature frame.

These tests establish the public contract of ``features_to_dataframe``:
given a ``CustomerFeatures`` object, it produces the one-row ``DataFrame``
that a fitted pipeline can be handed. The frame's columns are the canonical
model features, in canonical order, and its single row holds the values the
caller supplied.

What the adapter is for is translation, and these tests are positioned to
fail if it ever becomes more than that. A column list that grew would say
one-hot encoding happened here; a ``TotalCharges`` that arrived missing and
left as a number would say imputation happened here; a value that changed
on the way through would say something was scaled, mapped, or corrected.
All three belong to the fitted pipeline, which learned them from training
records, and none can be redone at the boundary without contradicting what
the pipeline already holds.

The expected columns come from ``churner.schema.features`` rather than from
a list written out again here, so there is one statement of what the
model's features are and the adapter is measured against it.

The feature values are built from that same canonical list, and each
category carries a value naming the column it belongs to. That is what
makes a misplaced column visible: a frame whose columns were shuffled,
shifted by one, or duplicated fails on the values as well as on the header,
which a payload of repeated ``"No"``s would not.

Nothing here preprocesses, predicts, builds a prediction service, loads a
model, or calls a route. The adapter's output is inspected as a frame; what
a model makes of that frame is tested where the model is.
"""

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from churner.api.adapters import features_to_dataframe
from churner.api.schemas import CustomerFeatures
from churner.schema.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# --- The features a frame has to carry, in the order it has to carry them ---
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# The one category the raw file records as a number rather than as text, so
# it is given an integer below while the rest are given strings.
SENIOR_CITIZEN = "SeniorCitizen"

# --- The one customer every case is built from ---
# Amounts chosen so that no two are equal and none is a round figure, and
# categories each carrying a value that names their own column. Nothing here
# has to look like a real customer: what the adapter owes a caller is that
# these values come out where they went in.
NUMERICAL_VALUES = {
    "tenure": 9,
    "MonthlyCharges": 65.25,
    "TotalCharges": 593.30,
}

CATEGORICAL_VALUES = {
    field_name: 1 if field_name == SENIOR_CITIZEN else f"{field_name}-value"
    for field_name in CATEGORICAL_FEATURES
}

FEATURE_VALUES = {**NUMERICAL_VALUES, **CATEGORICAL_VALUES}

# --- A category the training data does not hold ---
# A plausible future offering rather than nonsense, since this is the case
# the serving pipeline's ``handle_unknown="ignore"`` exists for.
UNKNOWN_CATEGORY_FIELD = "InternetService"
UNKNOWN_CATEGORY = "Satellite"

# --- Columns a feature frame must never carry ---
# The first two are excluded from the canonical schema, the identifier
# because it names a customer rather than describing one and the target
# because it is the outcome. The third is the response's spelling of the
# identifier, which the route reads from its path.
FORBIDDEN_COLUMNS = ("customerID", "Churn", "customer_id")


def feature_values(**overrides) -> dict:
    """State every feature of the one customer, with the given ones replaced."""
    return {**FEATURE_VALUES, **overrides}


def customer_features(**overrides) -> CustomerFeatures:
    """Build a validated feature object for that customer."""
    return CustomerFeatures(**feature_values(**overrides))


def first_row(frame: pd.DataFrame) -> dict:
    """Read the frame's only row as a plain mapping of column to value."""
    return frame.iloc[0].to_dict()


# --- What conversion produces ---


def test_a_feature_object_converts_to_a_dataframe():
    """A validated request's features become a pandas frame."""
    frame = features_to_dataframe(customer_features())

    assert isinstance(frame, pd.DataFrame)


def test_the_frame_holds_exactly_one_row():
    """One customer in means one row out, since one answer is expected."""
    frame = features_to_dataframe(customer_features())

    assert len(frame) == 1
    assert frame.shape == (1, len(MODEL_FEATURES))


# --- The columns, and their order ---


def test_the_frame_columns_are_the_canonical_features_in_canonical_order():
    """The header is ``churner.schema.features``, stated in its own order.

    The pipeline's ``ColumnTransformer`` selects by name, so the set is what
    has to match; the order is checked too, because a frame whose columns
    moved is the first sign of a feature list restated by hand rather than
    read from the schema.
    """
    frame = features_to_dataframe(customer_features())

    assert tuple(frame.columns) == MODEL_FEATURES


def test_the_frame_carries_no_column_the_model_was_not_fitted_on():
    """The 19 canonical columns arrive, each once, and no twentieth."""
    frame = features_to_dataframe(customer_features())

    assert len(frame.columns) == len(MODEL_FEATURES)
    assert len(set(frame.columns)) == len(MODEL_FEATURES)


@pytest.mark.parametrize("forbidden_column", FORBIDDEN_COLUMNS)
def test_the_frame_carries_no_identifier_or_target_column(forbidden_column):
    """Neither spelling of the identifier, nor the outcome, is a feature.

    A frame carrying ``Churn`` would offer a model the answer it is being
    asked for, and one carrying an identifier would offer it a column the
    pipeline was never fitted on.
    """
    frame = features_to_dataframe(customer_features())

    assert forbidden_column not in frame.columns


# --- The values that went in are the values that come out ---


def test_every_supplied_value_appears_in_its_own_column():
    """Each feature's value is found under that feature's name.

    Every value differs from every other, so a frame that paired a value
    with the wrong column fails here rather than passing by coincidence.
    """
    frame = features_to_dataframe(customer_features())

    assert first_row(frame) == FEATURE_VALUES


def test_the_amounts_are_not_rescaled_on_the_way_through():
    """The charges arrive as recorded, neither standardised nor rounded.

    Scaling is fitted state the pipeline holds, learned from training
    records, and a boundary that applied its own would be scaling twice.
    """
    recorded = first_row(features_to_dataframe(customer_features()))

    for field_name, stated_value in NUMERICAL_VALUES.items():
        assert recorded[field_name] == pytest.approx(stated_value)


def test_the_amounts_arrive_as_numbers_rather_than_text():
    """The numerical columns are numeric, which is what the imputer needs."""
    frame = features_to_dataframe(customer_features())

    for field_name in NUMERICAL_FEATURES:
        assert is_numeric_dtype(frame[field_name])


def test_an_unknown_category_is_passed_through_unchanged():
    """A category the model never saw reaches it as the caller spelled it.

    Deciding what to do with an unseen value belongs to the fitted
    ``OneHotEncoder``, which is configured to ignore one. An adapter that
    replaced it with a known category, or with a missing value, would answer
    that question on the encoder's behalf.
    """
    features = customer_features(**{UNKNOWN_CATEGORY_FIELD: UNKNOWN_CATEGORY})

    frame = features_to_dataframe(features)

    assert first_row(frame)[UNKNOWN_CATEGORY_FIELD] == UNKNOWN_CATEGORY


# --- An absent amount stays absent ---


def test_an_absent_total_charges_stays_missing():
    """``None`` arrives as a missing value, not as a filled-in figure.

    The pipeline's median imputer is what stands in for a blank
    ``TotalCharges``, and that median was learned from training records. An
    adapter that filled the gap itself would be substituting a number the
    model was not fitted to expect, and the imputer would then have nothing
    left to do.
    """
    frame = features_to_dataframe(customer_features(TotalCharges=None))

    assert frame["TotalCharges"].isna().all()


def test_an_absent_total_charges_leaves_the_other_features_untouched():
    """One missing amount does not disturb the row it is missing from."""
    frame = features_to_dataframe(customer_features(TotalCharges=None))

    assert tuple(frame.columns) == MODEL_FEATURES
    assert len(frame) == 1
    assert first_row(frame)["tenure"] == NUMERICAL_VALUES["tenure"]
    assert first_row(frame)["MonthlyCharges"] == pytest.approx(
        NUMERICAL_VALUES["MonthlyCharges"]
    )


# --- Conversion reads its input and nothing more ---


def test_converting_does_not_mutate_the_feature_object():
    """The object a caller still holds is the one it validated."""
    features = customer_features()
    stated_values = features.model_dump()

    features_to_dataframe(features)

    assert features.model_dump() == stated_values


def test_editing_the_frame_does_not_reach_back_into_the_feature_object():
    """The frame is the caller's to change; the request object is unaffected."""
    features = customer_features()
    frame = features_to_dataframe(features)

    frame.loc[frame.index[0], "tenure"] = 999
    frame.loc[frame.index[0], "Contract"] = "Edited in place"

    assert features.tenure == FEATURE_VALUES["tenure"]
    assert features.Contract == FEATURE_VALUES["Contract"]


def test_two_conversions_of_the_same_object_are_separate_frames():
    """Each call builds its own frame, so one caller cannot edit another's."""
    features = customer_features()

    first = features_to_dataframe(features)
    second = features_to_dataframe(features)

    assert first is not second

    first.loc[first.index[0], "tenure"] = 999

    assert first_row(second)["tenure"] == FEATURE_VALUES["tenure"]
