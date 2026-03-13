import pandas as pd
import src.data.sql as sql

def get_datetime(df):
    return pd.to_datetime({
        "year": 2000 + df["an"],
        "month": df["mois"],
        "day": df["jour"],
        "hour": df["hrmn"] // 100,
        "minute": df["hrmn"] % 100
    })

caract = pd.read_csv("data/raw/caracteristics.csv", encoding="latin-1", low_memory=False)

# Add timestamp directly in the raw data so we can simulate getting this data continuously
caract["timestamp"] = get_datetime(caract)

sql.write_dataframes("data/raw/accidents.db", {
    "caract": caract,
    "places": pd.read_csv("data/raw/places.csv", encoding="latin-1", low_memory=False),
    "users": pd.read_csv("data/raw/users.csv", encoding="latin-1"),
    "vehicles": pd.read_csv("data/raw/vehicles.csv", encoding="latin-1"),
    "holidays": pd.read_csv("data/raw/holidays.csv", encoding="latin-1")
})