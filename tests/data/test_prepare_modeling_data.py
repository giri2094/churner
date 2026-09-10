"""Unit tests for the deterministic modelling-data preparation.

These tests establish the public contract of ``prepare_modeling_data``: the
amounts it reads out of the recorded ``TotalCharges`` text, the blanks it
carries across as missing, the columns it hands to the model and the two it
withholds, and the input it refuses.

Most expectations are calculated by hand from the small fixture defined in this
module, so a failure points at the implementation rather than at a figure
copied from the real dataset. One test is deliberately the exception: the 11
blank ``TotalCharges`` entries are a property of the raw file recorded in
``docs/data_dictionary.md``, and the point of that test is that preparation
leaves all 11 of them missing rather than filling or dropping any.
"""

from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype
from pandas.testing import assert_frame_equal, assert_series_equal

from churner.data.load_dataset import load_dataset
from churner.data.prepare_modeling_data import prepare_modeling_data

# The raw file the data dictionary describes, resolved the way the test suite
# already resolves the source directory: from this file, not from the working
# directory the tests happen to be run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Recorded in ``docs/data_dictionary.md``: of the 7,043 records, 11 hold a
# blank ``TotalCharges`` and the remaining 7,032 hold a readable amount.
RAW_RECORD_COUNT = 7043
RAW_BLANK_TOTAL_CHARGES_COUNT = 11

# --- Controlled fixture population ---
# Five customers holding the raw representation: ``TotalCharges`` as text, one
# of the entries blank. The amounts are distinct and none is a round number, so
# a value read into the wrong row cannot pass unnoticed, and 1889.5 has fewer
# decimals than the others so a conversion that rounded would be visible.
#
#   customerID  TotalCharges  expected amount
#   0001-AAA    "29.85"       29.85
#   0002-BBB    "1889.5"      1889.50
#   0003-CCC    " "           missing
#   0004-DDD    "108.15"      108.15
#   0005-EEE    "1840.75"     1840.75
CUSTOMER_ROWS = [
    ("0001-AAA", "Female", 1, 29.85, "29.85", "No"),
    ("0002-BBB", "Male", 34, 56.95, "1889.5", "No"),
    # The blank entry, which is what leaves the raw column stored as text. Its
    # tenure of 0 matches the raw records the data dictionary describes.
    ("0003-CCC", "Male", 0, 53.85, " ", "Yes"),
    ("0004-DDD", "Male", 2, 53.85, "108.15", "Yes"),
    ("0005-EEE", "Female", 45, 42.30, "1840.75", "No"),
]

CUSTOMER_COLUMNS = [
    "customerID",
    "gender",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# Everything the model is handed: the fixture's columns less the identifier and
# the target.
FEATURE_COLUMNS = ["gender", "tenure", "MonthlyCharges", "TotalCharges"]


@pytest.fixture
def customers_df() -> pd.DataFrame:
    """Build the controlled five-customer population described above."""
    return pd.DataFrame(CUSTOMER_ROWS, columns=CUSTOMER_COLUMNS)


# --- Reading TotalCharges as amounts ---


def test_recorded_amounts_are_read_as_numbers(customers_df):
    """Each readable entry becomes the amount its text stands for, unrounded.

    Comparing the whole column also pins down its dtype, so an implementation
    that left the amounts as text would not pass on the values alone.
    """
    features, _ = prepare_modeling_data(customers_df)

    assert_series_equal(
        features["TotalCharges"],
        pd.Series(
            [29.85, 1889.5, float("nan"), 108.15, 1840.75],
            name="TotalCharges",
            dtype="float64",
        ),
    )


def test_charges_column_is_numeric_like_the_monthly_one(customers_df):
    """The converted amounts share the dtype ``MonthlyCharges`` already holds.

    Both columns are currency, and a model reaching for one of them should not
    have to treat it differently from the other.
    """
    features, _ = prepare_modeling_data(customers_df)

    assert is_float_dtype(features["TotalCharges"])
    assert features["TotalCharges"].dtype == features["MonthlyCharges"].dtype


@pytest.mark.parametrize("blank_entry", ["", " ", "   ", "\t"])
def test_blank_charges_become_missing(blank_entry):
    """An entry holding no figure records no amount, whatever whitespace it holds.

    The blank is carried across as missing rather than resolved: no zero is
    substituted, and the record is kept.
    """
    df = pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB"],
            "TotalCharges": ["29.85", blank_entry],
            "Churn": ["No", "Yes"],
        }
    )

    features, _ = prepare_modeling_data(df)

    assert features["TotalCharges"].isna().tolist() == [False, True]
    assert len(features) == 2


def test_blank_charges_are_not_filled_in(customers_df):
    """Nothing stands in for the missing amount, least of all a plausible zero.

    A zero would read as a customer charged nothing, which is a different claim
    from one whose total was never recorded, and it would pull any statistic
    later fitted on the column toward it.
    """
    features, _ = prepare_modeling_data(customers_df)

    blank_amount = features.loc[2, "TotalCharges"]
    assert pd.isna(blank_amount)
    assert features["TotalCharges"].isna().sum() == 1


def test_no_record_is_dropped_for_holding_no_amount(customers_df):
    """The customer with no recorded amount stays in the data, row and index.

    Dropping them would lose their churn outcome as well, which was recorded.
    """
    features, target = prepare_modeling_data(customers_df)

    assert len(features) == len(customers_df)
    assert features.index.tolist() == customers_df.index.tolist()
    assert target.index.tolist() == customers_df.index.tolist()


@pytest.mark.skipif(
    not RAW_DATASET_PATH.exists(),
    reason=f"The raw dataset is not present at {RAW_DATASET_PATH}.",
)
def test_the_raw_datasets_eleven_blanks_stay_missing():
    """All 11 blank entries in the real file survive preparation as missing.

    This is the one expectation read from the raw dataset rather than from a
    fixture, because it is the raw dataset's own documented property. It fails
    if any of the 11 were filled in, and it fails if the records carrying them
    were dropped instead.
    """
    raw_df = load_dataset(str(RAW_DATASET_PATH))

    features, target = prepare_modeling_data(raw_df)

    assert len(features) == RAW_RECORD_COUNT
    assert len(target) == RAW_RECORD_COUNT
    assert features["TotalCharges"].isna().sum() == RAW_BLANK_TOTAL_CHARGES_COUNT
    assert (
        features["TotalCharges"].notna().sum()
        == RAW_RECORD_COUNT - RAW_BLANK_TOTAL_CHARGES_COUNT
    )


# --- Separating the target from the predictors ---


def test_target_holds_the_recorded_churn_labels(customers_df):
    """The target is the ``Churn`` column, carried across as recorded.

    The labels are not encoded here: an encoding fitted before the split would
    be fitted on records the model is later tested against.
    """
    _, target = prepare_modeling_data(customers_df)

    assert_series_equal(target, customers_df["Churn"])


def test_target_is_not_left_among_the_predictors(customers_df):
    """``Churn`` is withheld from the features, so no model can fit on it."""
    features, _ = prepare_modeling_data(customers_df)

    assert "Churn" not in features.columns


def test_identifier_is_not_left_among_the_predictors(customers_df):
    """``customerID`` is withheld too: it says which customer, not what they are."""
    features, _ = prepare_modeling_data(customers_df)

    assert "customerID" not in features.columns


def test_every_other_column_is_handed_to_the_model(customers_df):
    """Only those two columns are withheld, and the rest keep their order.

    Feature selection is a modelling decision made later and with evidence; it
    is not made here by quietly leaving a column out.
    """
    features, _ = prepare_modeling_data(customers_df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert_frame_equal(features[["gender", "tenure"]], customers_df[["gender", "tenure"]])


def test_predictors_and_target_stay_aligned(customers_df):
    """Each row of predictors is paired with the outcome of the same customer.

    The identifier is gone from the features, so the index is what keeps a
    customer's predictors attached to their own label.
    """
    features, target = prepare_modeling_data(customers_df)

    assert len(features) == len(target)
    assert features.index.equals(target.index)
    assert target.tolist() == customers_df["Churn"].tolist()


# --- Guarantees about the caller's frame ---


def test_input_dataframe_is_not_modified(customers_df):
    """Preparation reads the frame it is given and writes nothing back.

    The identifier and the target are excluded from the features, not deleted
    from the source, and the raw text of ``TotalCharges`` is left as recorded.
    """
    original_df = customers_df.copy(deep=True)

    prepare_modeling_data(customers_df)

    assert_frame_equal(customers_df, original_df)
    assert list(customers_df.columns) == CUSTOMER_COLUMNS


def test_results_are_new_objects(customers_df):
    """Editing the prepared data cannot reach back into the caller's frame."""
    features, target = prepare_modeling_data(customers_df)

    features.loc[0, "MonthlyCharges"] = 999.99
    target.loc[0] = "Yes"

    assert customers_df.loc[0, "MonthlyCharges"] == pytest.approx(29.85)
    assert customers_df.loc[0, "Churn"] == "No"


# --- Input validation ---


@pytest.mark.parametrize("absent_column", ["customerID", "TotalCharges", "Churn"])
def test_missing_required_column_raises_value_error(customers_df, absent_column):
    """Any required column being absent is refused rather than worked around."""
    df = customers_df.drop(columns=[absent_column])

    with pytest.raises(ValueError, match=absent_column):
        prepare_modeling_data(df)


@pytest.mark.parametrize("unreadable_entry", ["not recorded", "1,889.5", "$108.15"])
def test_charges_that_are_neither_blank_nor_an_amount_raise_value_error(unreadable_entry):
    """An undocumented entry is refused instead of being coerced to missing.

    Coercing it would file it with the blanks, where the imputation stage would
    later fill it in as though an amount had simply been absent.
    """
    df = pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB"],
            "TotalCharges": ["29.85", unreadable_entry],
            "Churn": ["No", "Yes"],
        }
    )

    with pytest.raises(ValueError, match="TotalCharges"):
        prepare_modeling_data(df)


def test_nothing_is_prepared_before_the_refusal(customers_df):
    """A frame that is refused is left exactly as it was passed in."""
    df = customers_df.assign(TotalCharges=["29.85", "not recorded", " ", "108.15", "1840.75"])
    original_df = df.copy(deep=True)

    with pytest.raises(ValueError):
        prepare_modeling_data(df)

    assert_frame_equal(df, original_df)
