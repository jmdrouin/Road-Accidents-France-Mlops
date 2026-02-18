### Importing libraries and Data
## Import libraries

import pandas as pd
import visualization as viz
import cleanup_caract
import cleanup_places
import cleanup_users
SHOW_PLOTS = False

# Import data

caract = pd.read_csv("data/csv/caracteristics.csv", encoding="latin-1", low_memory=False)
caract = cleanup_caract.cleanup_caract(caract)

places = pd.read_csv("data/csv/places.csv", encoding="latin-1", low_memory=False)
places = cleanup_places.cleanup_places(places)

if SHOW_PLOTS:
    viz.visualize_overview(caract, places)

users = pd.read_csv("data/csv/users.csv", encoding="latin-1")
users = cleanup_users.cleanup_users(users)

holidays = pd.read_csv("data/csv/holidays.csv", encoding="latin-1")
vehicles = pd.read_csv("data/csv/vehicles.csv", encoding="latin-1")

if SHOW_PLOTS:
    viz.visualize_injury_severity(users)

### VEHICLES Dataframe - Auditing, Cleaning and Preprocessing
# Step 2: rename vehicles columns
rename_map = {
    "Num_Acc": "accident_id",
    "num_veh": "vehicle_id",
    "senc": "traffic_direction",
    "catv": "vehicle_category",
    "occutc": "occupants",
    "obs": "fixed_obstacle",
    "obsm": "mobile_obstacle",
    "choc": "impact_point",
    "manv": "manoeuvre"
}

vehicles = vehicles.rename(columns=rename_map)
print("Renamed columns:", list(vehicles.columns))
#Auditing

# Basic structure and memory usage
print(vehicles.info())

print("\nDtypes:")
print(vehicles.dtypes)

print("\nTop missing (up to 10):")
print(vehicles.isna().sum().sort_values(ascending=False).head(10))

print("\nSample (first 5 rows):")
print(vehicles.head(5).to_string(index=False))

# Shape and column names
print("Shape:", vehicles.shape)
print("Columns:", vehicles.columns.tolist())


for c in vehicles.columns:
    if c in vehicles.columns:
        print(f"\n{c} -- n_unique:", vehicles[c].nunique(dropna=True))
        print("value_sample:", vehicles[c].dropna().sort_values().unique()[:35])

# Missing values per column
print(vehicles.isna().sum())
#For vehicle_category -- n_unique: 33, fixed_obstacle -- n_unique: 17 and manoeuvre -- n_unique: 25 -
#instead of keeping dozens of fine‑grained codes, we can group them into broader, analysis‑friendly categories.

#what we have is 
#vehicle_category_map = {1: "Bicycle",2: "Moped <50cc",3: "Motorcycle <125cc",4: "Motorcycle >125cc",5: "Motorcycle with sidecar",6: "Light quadricycle",7: "Passenger car",
#                        8: "Utility vehicle",9: "Truck <3.5t",10: "Truck 3.5–7.5t",11: "Truck >7.5t",12: "Road tractor",13: "Agricultural tractor",14: "Bus",15: "Coach",
#                        16: "Tramway",17: "Special vehicle", 18: "Train",19: "Other vehicle",20: "Scooter",21: "Electric scooter",30: "Quad >50cc",31: "Mini-motorbike",
#                        32: "Other two-wheel motor vehicle",33: "Other three-wheel motor vehicle",34: "Other four-wheel motor vehicle",35: "Segway",36: "Electric bicycle",
#                        37: "Motorized wheelchair",38: "Personal mobility device",39: "Other personal mobility",40: "Unknown vehicle",99: "Unspecified"}#####
#
#fixed_obstacle_map = { 0: "None", 1: "Vehicle parked", 2: "Tree", 3: "Post", 4: "Wall", 5: "Rail/Barrier", 6: "Building", 7: "Ditch", 8: "Embankment", 
#                      9: "Other fixed obstacle", 10: "Central reservation",11: "Bridge pillar",12: "Tunnel wall",13: "Fence",14: "Traffic island",
#                      15: "Other infrastructure",16: "Animal (fixed)",}###
#
#manoeuvre_map = {0: "No manoeuvre",1: "Straight ahead",2: "Turning left",3: "Turning right",4: "U-turn",5: "Reversing",6: "Changing lane left",
#                 7: "Changing lane right",8: "Overtaking", 9: "Parking", 10: "Starting", 11: "Stopping", 12: "Avoiding obstacle left", 13: "Avoiding obstacle right", 
#                 14: "Crossing intersection", 15: "Entering roundabout", 16: "Leaving roundabout", 17: "Other manoeuvre", 18: "Emergency manoeuvre", 19: "Slipping/skidding", 
#                 20: "Loss of control", 21: "Pedestrian interaction", 22: "Animal interaction",23: "Other special manoeuvre",24: "Unknown manoeuvre"}
## mapping

traffic_direction_map = {
    0: "Same_direction",
    1: "Opposite_direction",
    2: "Other/Unknown"
}
vehicle_category_grouped = {
    1: "Cycle", 2: "Cycle", 36: "Cycle",   # bicycles, mopeds, e-bikes
    3: "Motorcycle", 4: "Motorcycle", 5: "Motorcycle", 20: "Motorcycle", 21: "Motorcycle", 30: "Motorcycle", 31: "Motorcycle", 32: "Motorcycle", 33: "Motorcycle", 34: "Motorcycle",
    6: "Car(Quad/Passenger)", 7: "Car(Quad/Passenger)",
    35: "Personal mobility", 37: "Personal mobility", 38: "Personal mobility", 39: "Personal mobility",
    8: "Utility(Truck/Agri)",9: "Utility(Truck/Agri)", 10: "Utility(Truck/Agri)", 11: "Utility(Truck/Agri)", 12: "Utility(Truck/Agri)",13: "Utility(Truck/Agri)",
    14: "Bus/Coach", 15: "Bus/Coach",
    16: "Tram/Train", 18: "Tram/Train",
    17: "other", 19: "other", 40: "other", 99: "other"
}
# reduces 33 categories down to 8 groups

fixed_obstacle_grouped = {
    0: "None",
    1: "Parked_vehicle",
    2: "Tree/Vegetation",
    3: "Post/Pole",
    4: "Wall/Building", 6: "Wall/Building",
    5: "Barrier/Rail/Fence", 10: "Barrier/Rail/Fence", 11: "Barrier/Rail/Fence", 12: "Barrier/Rail/Fence", 13: "Barrier/Rail/Fence",
    7: "Ditch/Embankment", 8: "Ditch/Embankment",
    9: "other",  15: "other", 14: "other", 16: "other"
}
# This reduces from 17 to 7 groups

mobile_obstacle_map = {
    0: "None",
    1: "Pedestrian",
    2: "Cyclist",
    4: "Animal (moving)",
    5: "Other vehicle",
    6: "Train",
    9: "other"
}

impact_point_map = {
    0: "No_impact",
    1: "Front",
    2: "Front_right",
    3: "Right_side",
    4: "Rear_right",
    5: "Rear",
    6: "Rear_left",
    7: "Left_side",
    8: "Front_left",
    9: "Multiple impacts"
}

manoeuvre_grouped = {
    0: "None",
    1: "Straight_ahead",
    2: "Turning", 3: "Turning", 14: "Turning", 4: "Turning",
    5: "Reversing",
    6: "Lane_change/Overtaking", 7:  "Lane_change/Overtaking", 8:  "Lane_change/Overtaking",
    9: "Park/Start/Stop", 10: "Park/Start/Stop", 11: "Park/Start/Stop",
    12: "Avoiding_obstacle", 13: "Avoiding_obstacle",
    15: "Roundabout", 16: "Roundabout",
    17: "other", 23: "other", 24: "other",
    18: "Emergency_manoeuvre",
    19: "Loss_of_control", 20: "Loss_of_control",
    21: "Interaction", 22: "Interaction"
}
# This reduces from 25 to 12 groups

vehicles["traffic_direction_label"] = vehicles["traffic_direction"].map(traffic_direction_map)
vehicles["vehicle_category_label"] = vehicles["vehicle_category"].map(vehicle_category_grouped)
vehicles["fixed_obstacle_label"] = vehicles["fixed_obstacle"].map(fixed_obstacle_grouped)
vehicles["mobile_obstacle_label"] = vehicles["mobile_obstacle"].map(mobile_obstacle_map)
vehicles["impact_point_label"] = vehicles["impact_point"].map(impact_point_map)
vehicles["manoeuvre_label"] = vehicles["manoeuvre"].map(manoeuvre_grouped)

# Check if any labels are missing (NaN after mapping)
print("\n--- Missing label counts ---")
print("Traffic direction:", vehicles["traffic_direction_label"].isna().sum())
print("Vehicle category:", vehicles["vehicle_category_label"].isna().sum())
print("Fixed obstacle:", vehicles["fixed_obstacle_label"].isna().sum())
print("Mobile obstacle:", vehicles["mobile_obstacle_label"].isna().sum())
print("Impact point:", vehicles["impact_point_label"].isna().sum())
print("Manoeuvre:", vehicles["manoeuvre_label"].isna().sum())

# Check unique values in each new label column
print("\n--- Unique labels ---")
print("Traffic direction:", vehicles["traffic_direction_label"].unique())
print("Vehicle category:", vehicles["vehicle_category_label"].unique())
print("Fixed obstacle:", vehicles["fixed_obstacle_label"].unique())
print("Mobile obstacle:", vehicles["mobile_obstacle_label"].unique())
print("Impact point:", vehicles["impact_point_label"].unique())
print("Manoeuvre:", vehicles["manoeuvre_label"].unique())
## more cleaning

# 1. filling Nans

vehicles["traffic_direction_label"] = vehicles["traffic_direction_label"].fillna("Unknown")
vehicles["fixed_obstacle_label"]   = vehicles["fixed_obstacle_label"].fillna("Unknown")
vehicles["mobile_obstacle_label"]  = vehicles["mobile_obstacle_label"].fillna("Unknown")
vehicles["impact_point_label"]     = vehicles["impact_point_label"].fillna("Unknown")
vehicles["manoeuvre_label"]        = vehicles["manoeuvre_label"].fillna("Unknown")

# 2. Check for outliers in occupants
vehicles["occupants_group"] = vehicles["occupants"].apply(lambda x: str(x) if x <= 10 else "10+")

# 3. Check duplicates
duplicates = vehicles.duplicated(subset=["accident_id", "vehicle_id"]).sum()
print("Duplicate rows:", duplicates)

#4. checking datatypes
print("\n")
print(vehicles.dtypes)

### Cleaning VEHICLES by Dropping columns before merging in master dataframe for visualisation

drop_cols_vehicles = [
    "traffic_direction", "vehicle_category", "occupants",
    "fixed_obstacle", "mobile_obstacle", "impact_point", "manoeuvre",
    "fixed_obstacle_label"
]

vehicles = vehicles.drop(columns=[c for c in drop_cols_vehicles if c in vehicles.columns])

print("Post-drop shape:", vehicles.shape)
print("Remaining columns:", vehicles.columns.tolist())
### HOLIDAYS Dataframe - Auditing, Cleaning and Preprocessing
# Inspect holiday dataframe
print("Columns:", list(holidays.columns))

print("\nDtypes:")
print(holidays.dtypes)

print("\nTop missing (up to 10):")
print(holidays.isna().sum().sort_values(ascending=False).head(10))

print("\nShape:", holidays.shape)

print("\nSample (first 5 rows):")
print(holidays.head(5).to_string(index=False))
# Step 1: rename columns
holidays = holidays.rename(columns={"ds": "date", "holiday": "holiday_name"})

# Step 2: convert to datetime
holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce")

# Step 3: add holiday flag
holidays["is_holiday"] = 1

# Step 4: drop duplicates
holidays = holidays.drop_duplicates(subset="date")

# Step 5: clean holiday_name
holidays["holiday_name"] = holidays["holiday_name"].str.strip().fillna("None")

print("Cleaned holidays shape:", holidays.shape)
print(holidays.head())
### DATA MERGING and Creation of Master Dataframe ACC (short for Accidents)
# List of dataframes to check
dfs = {
    "users": users,
    "caract": caract,
    "places": places,
    "vehicles": vehicles,
    "holidays": holidays
}

for name, df in dfs.items():
    print(f"\n=== {name.upper()} DATAFRAME INFO ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum().head(10))  # top 10 columns with NaNs
    print("\nSample rows:\n", df.head(3))
    print("="*60)
# Final Cleaning Checklist Before Merge

# Key Columns Consistency
print(caract["accident_id"].dtype, places["accident_id"].dtype, vehicles["accident_id"].dtype, users["accident_id"].dtype)
print(caract["date"].dtype, holidays["date"].dtype)

# Duplicate Keys
caract = caract.drop_duplicates(subset="accident_id")
places = places.drop_duplicates(subset="accident_id")
vehicles = vehicles.drop_duplicates(subset=["accident_id","vehicle_id"])
users = users.drop_duplicates(subset=["accident_id","vehicle_id"])
holidays = holidays.drop_duplicates(subset="date")

# Missing Values in Keys
print(caract["accident_id"].isna().sum())
print(places["accident_id"].isna().sum())
print(vehicles[["accident_id","vehicle_id"]].isna().sum())
print(users[["accident_id","vehicle_id"]].isna().sum())
print(holidays["date"].isna().sum())


#- Ensure is_holiday is binary (0/1) and no NaNs remain.
holidays["is_holiday"] = holidays["is_holiday"].fillna(1).astype(int)
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

output = "data/processed/acc.csv"
print(f"Writing to {output}")
acc.to_csv(output, index=True, encoding="utf-8")