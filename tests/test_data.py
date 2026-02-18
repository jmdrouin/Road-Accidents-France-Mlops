#from src.data import make_dataset as md
import pandas as pd
from pathlib import Path
from src.data.make_dataset import make_accidents_dataframe

def test_migration():
    nrows = 10000

    reference_file = Path("original_project/acc.csv")
    assert reference_file.exists(), f"Missing file: {reference_file}. Run ipynb file to generate it."

    # new_df = make_accidents_dataframe().head(nrows).reset_index(drop=True)
    new_df = pd.read_csv("data/processed/acc.csv", nrows=nrows).reset_index(drop=True)
    old_df = pd.read_csv(reference_file, nrows=nrows).reset_index(drop=True)

    missing_columns = set(old_df.columns) - set(new_df.columns)
    assert not missing_columns, "Column missing"

    unexpected_columns = set(new_df.columns) - set(old_df.columns)
    assert not unexpected_columns, "Unexpected column found"

    pd.testing.assert_frame_equal(new_df, old_df)

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