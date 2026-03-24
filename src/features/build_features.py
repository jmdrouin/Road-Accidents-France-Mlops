# Run this file as a script:
# uv run python -m src.features.build_features

import pandas as pd
import src.features._cleanup_caract as cleanup_caract
import src.features._cleanup_places as cleanup_places
import src.features._cleanup_users as cleanup_users
import src.features._cleanup_vehicles as cleanup_vehicles
import src.features._cleanup_holidays as cleanup_holidays
import src.features._cleanup_accidents as prepare_accidents_data
import src.data.sql as sql
from src.util.files import single_file_in_folder
from pathlib import Path
import glob, os

def combine_to_accidents_dataframe(
    caract: pd.DataFrame,
    places: pd.DataFrame,
    users: pd.DataFrame,
    vehicles: pd.DataFrame,
    holidays: pd.DataFrame
) -> pd.DataFrame:
    # Cleanup data
    print("Preparing Caract")
    caract = cleanup_caract.cleanup_caract(caract)
    print("Preparing Places")
    places = cleanup_places.cleanup_places(places)
    print("Preparing Users")
    users = cleanup_users.cleanup_users(users)
    print("Preparing Vehicles")
    vehicles = cleanup_vehicles.cleanup_vehicles(vehicles)
    print("Preparing Holidays")
    holidays = cleanup_holidays.cleanup_holidays(holidays)

    print("Merging dataframes")
    # Merge all dataframes into one
    acc = caract.copy() \
        .merge(places, how="left", on="accident_id") \
        .merge(vehicles, how="left", on="accident_id") \
        .merge(users, how="left", on=["accident_id", "vehicle_id"]) \
        .merge(holidays, how="left", on="date")

    print("Processing aggregated accidents data")
    # Dates absent from the holidays dataframe are just "not a holiday"
    acc["is_holiday"] = acc["is_holiday"].fillna(0).astype(int)
    acc["holiday_name"] = acc["holiday_name"].fillna("Not a holiday")
    acc = prepare_accidents_data.prepare_accidents_data(acc)

    return acc

def make_accidents_dataframe_from_sql(source_file):
    tables = ["caract", "places", "users", "vehicles", "holidays"]
    result = sql.read_as_dataframes(source_file, tables)

    print("Processing data")
    return combine_to_accidents_dataframe(
        *[result[table] for table in tables])

def build_features():
    folder = Path("data/raw/latest")
    file_path = single_file_in_folder(folder, "accidents_*.db")
    timestamp = file_path.stem.split("_")[1]

    dest_folder = "data/processed"
    for f in glob.glob(os.path.join(dest_folder, "accidents_*.db")):
        os.remove(f)
    dest_file = f"{dest_folder}/accidents_{timestamp}.db"

    df = make_accidents_dataframe_from_sql(file_path)
    sql.write_dataframe("accidents", df, to_file=dest_file)
    return dest_file

if __name__ == "__main__":
    build_features()