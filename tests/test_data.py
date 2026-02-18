#from src.data import make_dataset as md
import pandas as pd
from pathlib import Path

def test_migration():
    reference_file = Path("original_project/acc.csv")
    new_file = Path("data/processed/acc.csv")

    assert reference_file.exists(), f"Missing file: {reference_file}. Run ipynb file to generate it."
    assert new_file.exists(), f"Missing file: {new_file}"

    assert(reference_file.stat().st_size == new_file.stat().st_size)

    # Compare first rows of each file
    df_old = pd.read_csv(reference_file, nrows=10000).reset_index(drop=True)
    df_new = pd.read_csv(new_file, nrows=10000).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_old, df_new)

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