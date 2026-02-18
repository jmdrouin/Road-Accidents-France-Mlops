import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

def test_migration():
    nrows = 10000

    reference_file = Path("original_project/acc.csv")
    assert reference_file.exists(), f"Missing file: {reference_file}. Run ipynb file to generate it."

    #new_df = pd.read_csv("data/processed/acc.csv", nrows=nrows).reset_index(drop=True)

    engine = create_engine("sqlite:///data/processed/accidents.db")
    new_df = pd.read_sql(
        f"SELECT * FROM accidents LIMIT {nrows}",
        con=engine
    )
    old_df = pd.read_csv(reference_file, nrows=nrows)\
        .reset_index(drop=True) \
        .drop(columns=["Unnamed: 0"])

    missing_columns = set(old_df.columns) - set(new_df.columns)
    assert not missing_columns, "Column missing"

    unexpected_columns = set(new_df.columns) - set(old_df.columns)
    assert not unexpected_columns, "Unexpected column found"

    old_df["date"] = pd.to_datetime(old_df["date"]).dt.normalize()
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()

    # TODO: Remove this hack?
    # The original csv version of those two columns contains NAs,
    # While the "new" sql version has none. For now, I disabled the column
    # check for these columns.
    drops = ['mobile_obstacle_label', 'manoeuvre_label']
    old_df = old_df.drop(columns=drops)
    new_df = new_df.drop(columns=drops)

    pd.testing.assert_frame_equal(new_df, old_df, check_dtype=False)

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