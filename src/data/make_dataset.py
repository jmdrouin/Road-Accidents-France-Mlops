# Run this file as a script:
# uv run python -m src.data.make_dataset

import pandas as pd
import src.data.visualization as viz
import src.data.cleanup_caract as cleanup_caract
import src.data.cleanup_places as cleanup_places
import src.data.cleanup_users as cleanup_users
import src.data.cleanup_vehicles as cleanup_vehicles
import src.data.cleanup_holidays as cleanup_holidays
from sqlalchemy import create_engine

# TODO: Use sql instead?

def combine_to_accidents_dataframe(
    caract: pd.DataFrame,
    places: pd.DataFrame,
    users: pd.DataFrame,
    vehicles: pd.DataFrame,
    holidays: pd.DataFrame
) -> pd.DataFrame:

    # Cleanup data
    caract = cleanup_caract.cleanup_caract(caract)
    places = cleanup_places.cleanup_places(places)
    users = cleanup_users.cleanup_users(users)
    vehicles = cleanup_vehicles.cleanup_vehicles(vehicles)
    holidays = cleanup_holidays.cleanup_holidays(holidays)

    # Merge all dataframes into one
    acc = caract.copy() \
        .merge(places, how="left", on="accident_id") \
        .merge(vehicles, how="left", on="accident_id") \
        .merge(users, how="left", on=["accident_id", "vehicle_id"]) \
        .merge(holidays, how="left", on="date")

    # Dates absent from the holidays dataframe are just "not a holiday"
    acc["is_holiday"] = acc["is_holiday"].fillna(0).astype(int)
    acc["holiday_name"] = acc["holiday_name"].fillna("Not a holiday")

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
    engine = create_engine("sqlite:///data/sql/accidents.db")
    return combine_to_accidents_dataframe(
        caract = pd.read_sql_table("caracteristics", con=engine),
        places = pd.read_sql_table("places", con=engine),
        users = pd.read_sql_table("users", con=engine),
        vehicles = pd.read_sql_table("vehicles", con=engine),
        holidays = pd.read_sql_table("holidays", con=engine)
    )

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
    #acc = make_accidents_dataframe_from_csv()
    acc = make_accidents_dataframe_from_sql()
    output = "data/processed/acc.csv"
    print(f"Writing to {output}")
    acc.to_csv(output, index=True, encoding="utf-8")