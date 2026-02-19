# Run this file as a script:
# uv run python -m src.data.make_dataset

import pandas as pd
import src.data.visualization as viz
import src.data.cleanup_caract as cleanup_caract
import src.data.cleanup_places as cleanup_places
import src.data.cleanup_users as cleanup_users
import src.data.cleanup_vehicles as cleanup_vehicles
import src.data.cleanup_holidays as cleanup_holidays
import src.data.prepare_accidents_data as prepare_accidents_data
import src.data.sql as sql
from sqlalchemy import create_engine
from tqdm import tqdm

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

def make_accidents_dataframe_from_csv():
    return combine_to_accidents_dataframe(
        caract = pd.read_csv("data/csv/caracteristics.csv", encoding="latin-1", low_memory=False),
        places = pd.read_csv("data/csv/places.csv", encoding="latin-1", low_memory=False),
        users = pd.read_csv("data/csv/users.csv", encoding="latin-1"),
        vehicles = pd.read_csv("data/csv/vehicles.csv", encoding="latin-1"),
        holidays = pd.read_csv("data/csv/holidays.csv", encoding="latin-1")
    )

def make_accidents_dataframe_from_sql():
    tables = ["caracteristics", "places", "users", "vehicles", "holidays"]
    result = sql.read_as_dataframes("data/sql/accidents.db", tables)

    print("Processing data")
    return combine_to_accidents_dataframe(
        *[result[table] for table in tables])

def visualize(
    caract: pd.DataFrame,
    places: pd.DataFrame,
    users: pd.DataFrame,
    vehicles: pd.DataFrame,
    holidays: pd.DataFrame
):
    viz.visualize_overview(caract, places)
    viz.visualize_injury_severity(users)
    viz.visualize_user_vehicle(acc)
    viz.visualize_accidents(acc)

if __name__ == "__main__":
    write_to_sql = True
    acc = make_accidents_dataframe_from_sql()
    if write_to_sql:
        sql.write_dataframe("accidents", acc, to_file="data/processed/accidents.db")
    else:
        output = "data/processed/acc.csv"
        print(f"Writing to {output}")
        acc.to_csv(output, index=True, encoding="utf-8")