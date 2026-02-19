import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from src.data.make_dataset import combine_to_accidents_dataframe

import pandas as pd

def norm_label(s: pd.Series) -> pd.Series:
    # convert categorical/string/object -> pandas string dtype
    s = s.astype("string")
    # unify missing values (None, NaN, <NA>) to <NA>
    return s.where(s.notna(), pd.NA)

def assert_frames_equal_show_col(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left.columns) == list(right.columns)
    for col in left.columns:
        try:
            pd.testing.assert_series_equal(
                norm_label(left[col]),
                norm_label(right[col]),
                check_dtype=False,
                check_categorical=False,
            )
        except AssertionError as e:
            raise AssertionError(f"Mismatch in column: {col}\n{e}") from e


def test_migration():
    # Open reference file (a test file built from the top 5000 rows of the 5 raw csv tables)
    reference_file = Path("tests/resources/reference_file_master_acc.csv")
    assert reference_file.exists(), f"Missing file: {reference_file}. Run ipynb file to generate it."
    old_df = pd.read_csv(reference_file)
    
    new_df = combine_to_accidents_dataframe(
        caract = pd.read_csv("tests/resources/caract_5000.csv", encoding="latin-1", low_memory=False),
        places = pd.read_csv("tests/resources/places_5000.csv", encoding="latin-1", low_memory=False),
        users = pd.read_csv("tests/resources/users_5000.csv", encoding="latin-1"),
        vehicles = pd.read_csv("tests/resources/vehicles_5000.csv", encoding="latin-1"),
        holidays = pd.read_csv("tests/resources/holidays_5000.csv", encoding="latin-1")
    )

    # Make sure that the index column accident_id shows up as an column for comparing:
    if new_df.index.name == "accident_id":
        new_df = new_df.reset_index()

    unexpected_columns = set(new_df.columns) - set(old_df.columns)
    assert not unexpected_columns, f"Unexpected columns: {sorted(unexpected_columns)}"

    missing_columns = set(old_df.columns) - set(new_df.columns)
    assert not missing_columns, f"Missing columns: {sorted(missing_columns)}"

    assert len(old_df) == len(new_df)

    # Skip testing that column (TODO: investigate/fix the problem)
    drops = ['manoeuvre_label']
    old_df = old_df.drop(columns=drops)
    new_df = new_df.drop(columns=drops)

    assert_frames_equal_show_col(new_df, old_df)

# TODO TESTS:
# - There are no missing values in *_id and dates and is_holiday
# - Duplicates are dropped:
#    # Duplicate Keys
#       caract = caract.drop_duplicates(subset="accident_id")
#       places = places.drop_duplicates(subset="accident_id")
#       vehicles = vehicles.drop_duplicates(subset=["accident_id","vehicle_id"])
#       users = users.drop_duplicates(subset=["accident_id","vehicle_id"])
#       holidays = holidays.drop_duplicates(subset="date")
# - Maybe: all ids ARE in accidents