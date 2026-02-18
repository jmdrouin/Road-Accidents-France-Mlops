# Run this file as a script:
# uv run python -m src.data.make_dataset

import pandas as pd
import src.data.visualization as viz
import src.data.cleanup_caract as cleanup_caract
import src.data.cleanup_places as cleanup_places
import src.data.cleanup_users as cleanup_users
import src.data.cleanup_vehicles as cleanup_vehicles
import src.data.cleanup_holidays as cleanup_holidays
SHOW_PLOTS = False

def make_accidents_dataframe():
    # Import data

    caract = pd.read_csv("data/csv/caracteristics.csv", encoding="latin-1", low_memory=False)
    caract = cleanup_caract.cleanup_caract(caract)

    places = pd.read_csv("data/csv/places.csv", encoding="latin-1", low_memory=False)
    places = cleanup_places.cleanup_places(places)
    if SHOW_PLOTS:
        viz.visualize_overview(caract, places)

    users = pd.read_csv("data/csv/users.csv", encoding="latin-1")
    users = cleanup_users.cleanup_users(users)
    if SHOW_PLOTS:
        viz.visualize_injury_severity(users)

    vehicles = pd.read_csv("data/csv/vehicles.csv", encoding="latin-1")
    vehicles = cleanup_vehicles.cleanup_vehicles(vehicles)

    holidays = pd.read_csv("data/csv/holidays.csv", encoding="latin-1")
    holidays = cleanup_holidays.cleanup_holidays(holidays)

    # MERGE

    # Step 1: start with accident-level info
    acc = caract.copy()

    # Step 2: merge places
    acc = acc.merge(places, how="left", on="accident_id")

    # Step 3: merge vehicles
    acc = acc.merge(vehicles, how="left", on="accident_id")

    # Step 4: merge users (linked via accident_id + vehicle_id)
    acc = acc.merge(users, how="left", on=["accident_id","vehicle_id"])

    # Step 5: merge holidays (linked via date)
    acc = acc.merge(holidays, how="left", on="date")

    # Fill holiday flags for non-holiday rows
    acc["is_holiday"] = acc["is_holiday"].fillna(0).astype(int)
    acc["holiday_name"] = acc["holiday_name"].fillna("Not a holiday")

    #What the Results Mean
    #1 -  Duplicate Accident IDs: 593,404
    #- This is expected because of the one-to-many relationships:
    #- Each accident (accident_id) can involve multiple vehicles.
    #- Each vehicle can involve multiple users (driver + passengers).
    #- So when we merge caract (accident-level) with vehicles and users, the accident rows get replicated.
    #- This doesn’t mean bad data — it means the master dataframe acc is now granular at the user/vehicle level, not just accident-level.
    #If we want to analyze accident-level statistics, we'll need to deduplicate by accident_id or aggregate.
    #If we want to analyze user-level or vehicle-level statistics, keep the duplicates — they represent real entities.

    #2 - Missing Values in Keys
    #- accident_id, date, vehicle_id → all show 0 missing.
    #- This confirms the merge was clean and no join keys were lost.

    #33 Holiday Flag Distribution
    #- Non-holiday accidents: 1,403,380
    #- Holiday accidents: 30,009
    #- Holidays account for ~2% of all accidents.
    #- This is consistent: most accidents happen on regular days, but holidays are still a meaningful subset for analysis.

    if SHOW_PLOTS:
        viz.visualize_user_vehicle(acc)
        viz.visualize_accidents(acc)

    return acc

if __name__ == "__main__":
    acc = make_accidents_dataframe()
    output = "data/processed/acc.csv"
    print(f"Writing to {output}")
    acc.to_csv(output, index=True, encoding="utf-8")