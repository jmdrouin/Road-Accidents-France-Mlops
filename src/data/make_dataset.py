### Importing libraries and Data
## Import libraries

import pandas as pd
from src.data import visualization as viz

# Allow calling jupyter's "display" inside normal python script: 
from IPython.display import display

# Import data

caract = pd.read_csv("data/csv/caracteristics.csv", encoding="latin-1", low_memory=False)
holidays = pd.read_csv("data/csv/holidays.csv", encoding="latin-1")
places = pd.read_csv("data/csv/places.csv", encoding="latin-1", low_memory=False)
users = pd.read_csv("data/csv/users.csv", encoding="latin-1")
vehicles = pd.read_csv("data/csv/vehicles.csv", encoding="latin-1")

#data introduction

print("\nCharacteristics head:")
display(caract.head())

print("\nPlaces head:")
display(places.head())

print("\nVehicles head:")
display(vehicles.head())

print("\nUsers head:")
display(users.head())

print("\nHolidays head:")
display(holidays.head())

### CARACT (Characteristics) Dataframe - Auditing, Cleaning and Preprocessing
#column renaming

caract = caract.rename(columns={
    "Num_Acc":"accident_id",
    "jour":"day",
    "mois":"month",
    "an":"year",
    "hrmn":"hourminute",
    "lum":"lighting",
    "dep":"department",
    "com" : "commune_code",
    "agg":"urban_area",
    "int" : "intersection",
    "atm":"weather",
    "col":"collision_type",
    "adr" :"address",
    "lat":"latitude",
    "long":"longitude"
})
caract.head()
# Auditing

# 1: quick overview
print("shape:", caract.shape)
print("\ndtypes (all):\n", caract.dtypes)
print("\nhead(5):\n", caract.head(5))
print("\ninfo():")
caract.info()

# 2: missingness summary (top 30)
miss = caract.isna().sum().sort_values(ascending=False)
print(miss.head(30))

# 3: unique counts for carcteristics columns (small preview)
cols_check = ['accident_id', 'year', 'month', 'day', 'hourminute', 'lighting',
       'urban_area', 'intersection', 'weather', 'collision_type',
       'commune_code', 'address', 'gps', 'latitude', 'longitude',
       'department']
for c in cols_check:
    if c in caract.columns:
        print(f"\n{c} -- n_unique:", caract[c].nunique(dropna=True))
        print("value_sample:", caract[c].dropna().sort_values().unique()[:12])

# 4: primary key duplicates
n_unique = caract["accident_id"].nunique()
print("\n Accident_id rows:", len(caract), "unique Accident_id:", n_unique, "duplicates:", len(caract)-n_unique)
# -- Mapping dictionaries --
lighting_map = {
    1: "Daylight",
    2: "Twilight_or_dawn",
    3: "Night_without_public_lighting",
    4: "Night_with_public_lighting_not_functioning",
    5: "Night_with_public_lighting_on",
}

urban_map = {
    1: "Out_of_agglomeration",
    2: "Built_up_area"
}

intersection_map = {
    0: "Unknown",
    1: "Out_of_intersection",
    2: "Intersection_X",
    3: "Intersection_T",
    4: "Intersection_Y",
    5: ">4_branches",
    6: "Roundabout",
    7: "Place",
    8: "Level_crossing",
    9: "Other"
}

weather_map = {
    1: "Normal",
    2: "Light_rain",
    3: "Heavy_rain",
    4: "Snow_or_hail",
    5: "Fog_or_smoke",
    6: "Strong_wind_or_storm",
    7: "Dazzling",
    8: "Cloudy",
    9: "Other"
}

collision_map = {
    1: "Frontal_two_vehicles",
    2: "Rear_end_two_vehicles",
    3: "Side_two_vehicles",
    4: "Chain_three_plus",
    5: "Multiple_three_plus",
    6: "Other_collision",
    7: "No_collision",
}

gps_map = {
    "M": "Manual_entry",
    "A": "Automatic_device",
    "G": "GPS",
    "R": "Road_reference",
    "Y": "Survey",
    "S": "Satellite",
    "T": "Topographic",
    "C": "Cartographic",
    "P": "Photogrammetric",
    "0": "No_geolocation"
}


# Apply maps safely (only for existing columns)
if "lighting" in caract.columns:
    caract["lighting_label"] = caract["lighting"].map(lighting_map).fillna("Unknown")
if "urban_area" in caract.columns:
    caract["urban_label"] = caract["urban_area"].map(urban_map).fillna("Unknown")
if "intersection" in caract.columns:
    caract["intersection_label"] = caract["intersection"].map(intersection_map).fillna("Unknown")
if "weather" in caract.columns:
    caract["weather_label"] = caract["weather"].map(weather_map)
if "collision_type" in caract.columns:
    caract["collision_label"] = caract["collision_type"].map(collision_map)
if "gps" in caract.columns:
    caract["gps_label"] = caract["gps"].map(gps_map).fillna("Unknown").astype("category")

#latitude and longitude

# 1) Clean raw columns robustly
caract["longitude"] = ( caract["longitude"].astype(str).str.strip().str.replace(",", ".", regex=False))

caract["latitude"] = ( caract["latitude"].astype(str).str.strip().str.replace(",", ".", regex=False))

# 2) Coerce to numeric
caract["longitude"] = pd.to_numeric(caract["longitude"], errors="coerce")
caract["latitude"] = pd.to_numeric(caract["latitude"], errors="coerce")

# 3) Set placeholders to NaN
caract.loc[caract["longitude"] == 0, "longitude"] = pd.NA
caract.loc[caract["latitude"] == 0, "latitude"] = pd.NA

# 4) Generate candidates for normalization
lon_raw = caract["longitude"]
lat_raw = caract["latitude"]

lon_cand = pd.DataFrame({
    "raw": lon_raw,
    "div1e5": lon_raw / 100000,  # common scale in this dataset
    "div1e6": lon_raw / 1000000  # fallback if some rows use 1e6
})

lat_cand = pd.DataFrame({
    "raw": lat_raw,
    "div1e5": lat_raw / 100000,
    "div1e6": lat_raw / 1000000
})

# 5) Choose candidate that falls in valid France ranges
def choose_lon(row):
    for v in [row["raw"], row["div1e5"], row["div1e6"]]:
        if pd.notna(v) and -10 <= v <= 10:
            return v
    return pd.NA

def choose_lat(row):
    for v in [row["raw"], row["div1e5"], row["div1e6"]]:
        if pd.notna(v) and 40 <= v <= 52:
            return v
    return pd.NA

caract["longitude_num"] = lon_cand.apply(choose_lon, axis=1)
caract["latitude_num"]  = lat_cand.apply(choose_lat, axis=1)

# 6) Basic validation
print("Longitude range:", caract["longitude_num"].min(), caract["longitude_num"].max())
print("Negative longitudes:", (caract["longitude_num"] < 0).sum())
print("Latitude range:", caract["latitude_num"].min(), caract["latitude_num"].max())

# 7) Quick sanity sample
print(caract[["latitude","longitude","latitude_num","longitude_num"]].head(10))

print("Latitude range:", caract["latitude_num"].min(), caract["latitude_num"].max())
print("Longitude range:", caract["longitude_num"].min(), caract["longitude_num"].max())

print("Missing latitudes:", caract["latitude_num"].isna().sum())
print("Missing longitudes:", caract["longitude_num"].isna().sum())

outliers = caract[
    (caract["latitude_num"] < 41) | (caract["latitude_num"] > 52) |
    (caract["longitude_num"] < -5) | (caract["longitude_num"] > 10)
]
print("Outlier rows:", len(outliers))

print("Negative longitudes:", (caract["longitude_num"] < 0).sum())
# valid geo
caract["valid_geo"] = (
    caract["latitude_num"].between(41, 52, inclusive="both") &
    caract["longitude_num"].between(-5, 10, inclusive="both")
)

# Quick check
print(caract["valid_geo"].value_counts())
print("Valid lat range:", caract.loc[caract["valid_geo"], "latitude_num"].min(),
      caract.loc[caract["valid_geo"], "latitude_num"].max())
print("Valid lon range:", caract.loc[caract["valid_geo"], "longitude_num"].min(),
      caract.loc[caract["valid_geo"], "longitude_num"].max())
# year_month_day_date (expanding 2-digit years safely and build date)

caract["year"] = pd.to_numeric(caract["year"], errors="coerce")
def expand_year(x):
    if pd.isna(x): return pd.NA
    x = int(x)
    return 2000 + x if x < 100 else x

caract["year"] = pd.to_numeric(caract["year"], errors="coerce").apply(expand_year).astype("Int64")
caract["month"] = pd.to_numeric(caract["month"], errors="coerce").astype("Int64")
caract["day"] = pd.to_numeric(caract["day"], errors="coerce").astype("Int64")
caract["date"] = pd.to_datetime(dict(year=caract["year"], month=caract["month"], day=caract["day"]), errors="coerce")

# quick check
print("dates valid:", caract["date"].notna().sum(), "/", len(caract))

# parse hourminute into hour/minute/time_hhmm 

def parse_hourminute(x):
        try:
            if pd.isna(x): return pd.NA, pd.NA
            s = str(int(x)).zfill(4) if str(x).strip().isdigit() else "".join(ch for ch in str(x) if ch.isdigit()).zfill(4)
            if not s: return pd.NA, pd.NA
            h, m = int(s[:2]), int(s[2:])
            return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else (pd.NA, pd.NA)
        except:
            return pd.NA, pd.NA

if "hour" not in caract.columns or "minute" not in caract.columns:  
    hm = caract["hourminute"].apply(parse_hourminute)
    caract["hour"] = hm.apply(lambda t: t[0])
    caract["minute"] = hm.apply(lambda t: t[1])
    caract["time_hhmm"] = caract.apply(lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}" 
                               if pd.notna(r['hour']) and pd.notna(r['minute']) else pd.NA, axis=1)

print("hour non-null:", caract["hour"].notna().sum())
# time_of_day: based on hour
def get_time_of_day(h):
    if pd.isna(h): return pd.NA
    if 5 <= h < 12: return "Morning"
    elif 12 <= h < 17: return "Afternoon"
    elif 17 <= h < 21: return "Evening"
    else: return "Night"

caract["time_of_day"] = caract["hour"].apply(get_time_of_day).astype("category")
# is_weekend: Saturday (5) or Sunday (6)
caract["is_weekend"] = caract["date"].dt.dayofweek.isin([5,6])
# season: based on month
def get_season(m):
    if pd.isna(m): return pd.NA
    if m in [12,1,2]: return "Winter"
    elif m in [3,4,5]: return "Spring"
    elif m in [6,7,8]: return "Summer"
    else: return "Autumn"

caract["season"] = caract["month"].apply(get_season).astype("category")
# unified dtype conversion applied directly to caract

# accident_id stays int64
caract["accident_id"] = caract["accident_id"].astype("int64")

# numeric codes (nullable int)
int_cols = ["year","month","day","hourminute","lighting","urban_area",
            "intersection","department","hour","minute","commune_code"]
for c in int_cols:
    if c in caract.columns:
        caract[c] = pd.to_numeric(caract[c], errors="coerce").astype("Int64")

# continuous numeric
float_cols = ["weather","collision_type","latitude","longitude","latitude_num","longitude_num"]
for c in float_cols:
    if c in caract.columns:
        caract[c] = pd.to_numeric(caract[c], errors="coerce").astype("float64")

# categorical labels
cat_cols = ["lighting_label","urban_label","intersection_label",
            "weather_label","collision_label","gps_label","time_of_day","season"]
for c in cat_cols:
    if c in caract.columns:
        caract[c] = caract[c].astype("category")

# text fields
text_cols = ["address","gps","time_hhmm"]
for c in text_cols:
    if c in caract.columns:
        caract[c] = caract[c].astype(str)

# datetime
if "date" in caract.columns:
    caract["date"] = pd.to_datetime(caract["date"], errors="coerce")

# boolean flags
bool_cols = ["valid_geo","is_weekend"]
for c in bool_cols:
    if c in caract.columns:
        caract[c] = caract[c].astype("bool")

# quick summary
print("\nDtypes after conversion:\n", caract.dtypes)
print("\nMissing values per column (top 10):\n", caract.isna().sum().sort_values(ascending=False).head(10))
# === Final cleaning checks for caract ===

print("=== Schema (dtypes) ===")
print(caract.dtypes)

print("\n=== Shape ===")
print("Rows:", caract.shape[0], "| Columns:", caract.shape[1])

print("\n=== Missing values (top 15) ===")
print(caract.isna().sum().sort_values(ascending=False).head(15))

print("\n=== Duplicate check ===")
dup_count = caract["accident_id"].duplicated().sum()
print("Duplicate accident_id count:", dup_count)


print("\n=== Category checks ===")
for c in ["lighting_label","urban_label","intersection_label",
          "weather_label","collision_label","gps_label",
          "time_of_day","season"]:
    if c in caract.columns:
        print(f"{c}: {caract[c].nunique(dropna=True)} categories")

print("\n=== Sample derived features ===")
print(caract[["date","hour","time_of_day","is_weekend","season"]].head(10))
### some visualization plots

caract.columns
### Cleaning CARACT by Dropping columns before merging in master dataframe for visualisation

drop_cols_caract = [
    "hourminute", "lighting", "urban_area", "intersection", "weather", "collision_type","commune_code",
    "address", "gps", "latitude", "longitude", "department"
]

caract = caract.drop(columns=[c for c in drop_cols_caract if c in caract.columns])

print("Post-drop shape:", caract.shape)
print("Remaining columns:", caract.columns.tolist())

### PLACES Dataframe - Auditing, Cleaning and Preprocessing
# Inspect places dataframe
print("Columns:", list(places.columns))

print("\nDtypes:")
print(places.dtypes)

print("\nTop missing (up to 10):")
print(places.isna().sum().sort_values(ascending=False).head(10))

print("\nShape:", places.shape)

print("\nSample (first 5 rows):")
display(places.head(5))
places = places.rename(columns={
    "Num_Acc": "accident_id",
    "catr": "road_category",
    "voie": "road_number",
    "v1": "road_number_index",
    "v2": "road_number_suffix",
    "circ": "traffic_regime",
    "nbv": "num_lanes",
    "pr": "milestone",
    "pr1": "milestone_distance",
    "vosp": "reserved_lane",
    "prof": "road_profile",
    "plan": "road_layout",
    "lartpc": "pedestrian_crossing_width",
    "larrout": "road_width",
    "surf": "surface_condition",
    "infra": "infrastructure",
    "situ": "situation",
    "env1": "school_zone"
})
# Auditing

# 1: quick overview
print("shape:", places.shape)
print("\ndtypes (all):\n", places.dtypes)
print("\nhead(5):\n", places.head(5))
print("\ninfo():")
places.info()

# 2: missingness summary (top 30)
miss = places.isna().sum().sort_values(ascending=False)
print(miss.head(30))

# 3: unique counts for mapping
for c in places.columns:
    if c in places.columns:
        print(f"\n{c} -- n_unique:", places[c].nunique(dropna=True))
        print("value_sample:", places[c].dropna().sort_values().unique()[:10])

# 4: primary key duplicates
n_unique = caract["accident_id"].nunique()
print("\n Accident_id rows:", len(caract), "unique Accident_id:", n_unique, "duplicates:", len(caract)-n_unique)
#mapping

road_category_map = {
    1: "Highway",
    2: "National_road",
    3: "Departmental_road",
    4: "Communal_way",
    5: "Off_public_network",
    6: "Parking_lot",
    9: "other"
}

traffic_regime_map = {
    0: "Unknown",
    1: "One_way",
    2: "Bidirectional",
    3: "Separated_carriageways",
    4: "Variable_assignment"
}

reserved_lane_map = {
    0: "Unknown",
    1: "Bike_path",
    2: "Cycle_bank",
    3: "Reserved_channel"
}

road_profile_map = {
    0: "Unknown",
    1: "Flat",
    2: "Slope",
    3: "Hilltop",
    4: "Hill_bottom"
}

road_layout_map = {
    0: "Unknown",
    1: "Straight",
    2: "Curve_left",
    3: "Curve_right",
    4: "S-bend"
}

surface_condition_map = {
    0: "Unknown",
    1: "Normal",
    2: "Wet",
    3: "Puddles",
    4: "Flooded",
    5: "Snow",
    6: "Mud",
    7: "Icy",
    8: "Oil",
    9: "Other"
}

infrastructure_map = {
    0: "Unknown",
    1: "Tunnel",
    2: "Bridge",
    3: "Interchange",
    4: "Railway",
    5: "Roundabout",
    6: "Pedestrian_area",
    7: "Tollzone"
}

situation_map = {
    0: "Unknown",
    1: "On_road",
    2: "On_emergency_stop_band",
    3: "On_verge",
    4: "On_sidewalk",
    5: "On_bike_path"
}

school_zone_map = {
    0: "Not_near_school",
    3: "Near_school",
    99: "Unknown"
}

# --- Applying mappings ---
places["road_category_label"] = places["road_category"].map(road_category_map)
places["traffic_regime_label"] = places["traffic_regime"].map(traffic_regime_map)
places["reserved_lane_label"] = places["reserved_lane"].map(reserved_lane_map)
places["road_profile_label"] = places["road_profile"].map(road_profile_map)
places["road_layout_label"] = places["road_layout"].map(road_layout_map)
places["surface_condition_label"] = places["surface_condition"].map(surface_condition_map)
places["infrastructure_label"] = places["infrastructure"].map(infrastructure_map)
places["situation_label"] = places["situation"].map(situation_map)
places["school_zone_label"] = places["school_zone"].map(school_zone_map)

# --- Filling NaNs with "Unknown" ---
for col in [
    "road_category_label", "traffic_regime_label", "reserved_lane_label",
    "road_profile_label", "road_layout_label", "surface_condition_label",
    "infrastructure_label", "situation_label", "school_zone_label"
]:
    places[col] = places[col].fillna("Unknown")

# Quick check: preview mappings side by side
check_cols = [
    "road_category", "road_category_label",
    "traffic_regime", "traffic_regime_label",
    "reserved_lane", "reserved_lane_label",
    "road_profile", "road_profile_label",
    "road_layout", "road_layout_label",
    "surface_condition", "surface_condition_label",
    "infrastructure", "infrastructure_label",
    "situation", "situation_label",
    "school_zone", "school_zone_label"
]

for i in range(0, len(check_cols), 2):
    col_code = check_cols[i]
    col_label = check_cols[i+1]
    print(f"\n{col_code} → {col_label}")
    print(places[[col_code, col_label]].drop_duplicates().sort_values(by=col_code))
### Visualization

viz.visualize_overview(caract, places)

# Flat terrain dominates accident frequency, but not necessarily severity.
# Slopes deserve attention for targeted safety measures (e.g., signage, speed control).
# Unknown values are present but manageable — no major data quality concern.

places.columns
### Cleaning CARACT by Dropping columns before merging in master dataframe for visualisation

drop_cols_places = [
    "road_category", "road_number", "road_number_index", "road_number_suffix",
    "traffic_regime", "milestone", "milestone_distance",
    "reserved_lane", "road_profile", "road_layout",
    "surface_condition", "infrastructure", "situation", "school_zone"
]

places = places.drop(columns=[c for c in drop_cols_places if c in places.columns])

print("Post-drop shape:", places.shape)
print("Remaining columns:", places.columns.tolist())
### USERS Dataframe - Auditing, Cleaning and Preprocessing
# Step 1: rename users columns
rename_map = {
    "Num_Acc": "accident_id",
    "num_veh": "vehicle_id",
    "place": "seat_position",
    "catu": "user_category",       # driver, passenger, pedestrian
    "grav": "injury_severity",
    "sexe": "sex",
    "trajet": "journey_purpose",
    "secu": "safety_equipment",
    "locp": "pedestrian_location",
    "actp": "pedestrian_action",
    "etatp": "pedestrian_state",
    "an_nais": "birth_year"
}

users = users.rename(columns=rename_map)
print("Renamed columns:", list(users.columns))
# Inspect users dataframe
print("Columns:", list(users.columns))

print("\nDtypes:")
print(users.dtypes)

print("\nTop missing (up to 10):")
print(users.isna().sum().sort_values(ascending=False).head(10))

print("\nShape:", users.shape)

print("\nSample (first 5 rows):")
print(users.head(5).to_string(index=False))

for c in users.columns:
    if c in users.columns:
        print(f"\n{c} -- n_unique:", users[c].nunique(dropna=True))
        print("value_sample:", users[c].dropna().sort_values().unique()[:25])
#   safety_equipment_map = {
#   0:  "None",
    
#  # Belt
#    1:  "Seat belt (unspecified usage)",
#    11: "Seat belt - used",
#    12: "Seat belt - not used",
#    13: "Seat belt - usage undetermined",

#    # Helmet
#    2:  "Helmet (unspecified usage)",
#    21: "Helmet - used",
#    22: "Helmet - not used",
#    23: "Helmet - usage undetermined",

#    # Child device
#    3:  "Child device (unspecified usage)",
#    10: "Child device (unspecified)",   # <-- missing earlier
#    20: "Child device (unspecified)",   # <-- missing earlier
#    30: "Child device (unspecified)",
#    31: "Child device - used",
#    32: "Child device - not used",
#   33: "Child device - usage undetermined",

#    # Reflective equipment
#    40: "Reflective equipment (unspecified)",
#    41: "Reflective equipment - used",
#    42: "Reflective equipment - not used",
#    43: "Reflective equipment - usage undetermined",

#    # Other equipment
#    90: "Other equipment (unspecified)",
#    91: "Other equipment - used",
#    92: "Other equipment - not used",
#   93: "Other equipment - usage undetermined"
#  }

# it has 24 labels, keeping all 24 safety_equipment codes is too granular for analysis. 
# splitting the safety_equipment codes into two separate columns (safety_equipment_type and safety_equipment_usage) is the most structured approach.It will preserve the equipment type and separately capture the usage status.This avoids collapsing too much information into one column, while still reducing complexity compared to 24 codes.Also making analysis easier
#mapping

sex_map = {
    1: "Male",
    2: "Female"
}

user_category_map = {
    1: "Driver",
    2: "Passenger",
    3: "Pedestrian",
    4: "Pedestrian_(rollerblade/scooter)"
}

injury_severity_map = {
    1: "Uninjured",
    2: "Killed",
    3: "Injured_Hospitalized",
    4: "Injured_Slight"
}

seat_position_map = {
    0: "Unknown", 9: "Unknown",
    1: "Driver_seat",
    2: "Front_passenger",
    3: "Rear_left",
    4: "Rear_middle",
    5: "Rear_right",
    6: "other_rear",
    7: "other",
    8: "Outside_vehicle",
}

journey_purpose_map = {
    0: "Unknown",
    1: "Home-work",
    2: "Home-school",
    3: "Shopping",
    4: "Professional use",
    5: "Leisure",
    9: "other"
}

pedestrian_location_map = {
    0: "Not_pedestrian",
    1: "On_pavement_>50m_from_crossing",
    2: "On_pavement_<50m_from_crossing",
    3: "On_crossing_no_signal",
    4: "On_crossing_with_signal",
    5: "On_sidewalk",
    6: "On_verge",
    7: "On_refuge/BAU",
    8: "On_against_aisle"
}

pedestrian_action_map = {
    0: "Unknown",
    1: "Walking_in_vehicle_direction",
    2: "Walking_opposite_vehicle_direction",
    3: "Crossing",
    4: "Masked",
    5: "Playing_or_Running",
    6: "With_animal",
    9: "other"
}

pedestrian_state_map = {
    0: "Not_pedestrian",   
    1: "Alone",            
    2: "Accompanied",     
    3: "In_group"     
}

# Define safety equipment type mapping
type_map = {
    0: "None",
    1: "Seat_belt", 11: "Seat_belt", 12: "Seat_belt", 13: "Seat_belt",
    2: "Helmet",    21: "Helmet",    22: "Helmet",    23: "Helmet",
    3: "Child_device", 10: "Child_device", 20: "Child_device",
    30: "Child_device", 31: "Child_device", 32: "Child_device", 33: "Child_device",
    40: "Reflective_equipment",41: "Reflective_equipment", 42: "Reflective_equipment", 43: "Reflective_equipment",
    90: "other",91: "other", 92: "other", 93: "other"
}

# Define safety equipment usage mapping
usage_map = {
    0: "None",
    1: "Unspecified", 2: "Unspecified", 3: "Unspecified", 10: "Unspecified", 20: "Unspecified", 30: "Unspecified",40: "Unspecified", 90: "Unspecified", 
    11: "Used", 21: "Used", 31: "Used", 41: "Used", 91: "Used",
    12: "Not_used", 22: "Not_used", 32: "Not_used", 42: "Not_used", 92: "Not_used",
    13: "Undetermined", 23: "Undetermined", 33: "Undetermined", 43: "Undetermined", 93: "Undetermined"
}

# Apply mappings to create new label columns
users["sex_label"] = users["sex"].map(sex_map)
users["user_category_label"] = users["user_category"].map(user_category_map)
users["injury_severity_label"] = users["injury_severity"].map(injury_severity_map)
users["seat_position_label"] = users["seat_position"].map(seat_position_map)
users["journey_purpose_label"] = users["journey_purpose"].map(journey_purpose_map)
users["pedestrian_location_label"] = users["pedestrian_location"].map(pedestrian_location_map)
users["pedestrian_action_label"] = users["pedestrian_action"].map(pedestrian_action_map)
users["pedestrian_state_label"] = users["pedestrian_state"].map(pedestrian_state_map)

# Safety equipment split into two columns
users["safety_equipment_type"] = users["safety_equipment"].map(type_map)
users["safety_equipment_usage"] = users["safety_equipment"].map(usage_map)
# Check for unmapped codes (NaN values)
print("\n--- Missing Labels Check ---")
for col in [
    "sex_label", "user_category_label", "injury_severity_label",
    "seat_position_label", "journey_purpose_label",
    "pedestrian_location_label", "pedestrian_action_label",
    "pedestrian_state_label", "safety_equipment_type", "safety_equipment_usage"
]:
    missing = users[col].isna().sum()
    print(f"{col}: {missing} unmapped values")

# Fill NaN values with "Unknown" for all mapped columns
users["seat_position_label"] = users["seat_position_label"].fillna("Unknown")
users["journey_purpose_label"] = users["journey_purpose_label"].fillna("Unknown")
users["pedestrian_location_label"] = users["pedestrian_location_label"].fillna("Unknown")
users["pedestrian_action_label"] = users["pedestrian_action_label"].fillna("Unknown")
users["pedestrian_state_label"] = users["pedestrian_state_label"].fillna("Unknown")
users["safety_equipment_type"] = users["safety_equipment_type"].fillna("Unknown")
users["safety_equipment_usage"] = users["safety_equipment_usage"].fillna("Unknown")


# Distribution checks
print("\n--- Distribution Checks ---")
print("Sex:\n", users["sex_label"].value_counts(dropna=False))
print("\nUser Category:\n", users["user_category_label"].value_counts(dropna=False))
print("\nInjury Severity:\n", users["injury_severity_label"].value_counts(dropna=False))
print("\nSeat Position:\n", users["seat_position_label"].value_counts(dropna=False))
print("\nJourney Purpose:\n", users["journey_purpose_label"].value_counts(dropna=False))
print("\nPedestrian Location:\n", users["pedestrian_location_label"].value_counts(dropna=False))
print("\nPedestrian Action:\n", users["pedestrian_action_label"].value_counts(dropna=False))
print("\nPedestrian State:\n", users["pedestrian_state_label"].value_counts(dropna=False))
print("\nSafety Equipment Type:\n", users["safety_equipment_type"].value_counts(dropna=False))
print("\nSafety Equipment Usage:\n", users["safety_equipment_usage"].value_counts(dropna=False))

viz.visualize_injury_severity(users)

users.columns
### Cleaning USERS by Dropping columns before merging in master dataframe for visualisation

drop_cols_users = [
    "seat_position", "user_category", "injury_severity", "sex", "journey_purpose",
    "safety_equipment", "pedestrian_location", "pedestrian_action", "pedestrian_state"
]

users = users.drop(columns=[c for c in drop_cols_users if c in users.columns])

print("Post-drop shape:", users.shape)
print("Remaining columns:", users.columns.tolist())
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

print("Final acc shape:", acc.shape)
print(acc.head())
# Post-Merge Sanity Checks

# Check duplicates
print("Duplicate accident IDs:", acc["accident_id"].duplicated().sum())

# Check missing values in key columns
print(acc[["accident_id","date","vehicle_id"]].isna().sum())

# Quick distribution of holiday flag
print(acc["is_holiday"].value_counts())
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

viz.visualize_user_vehicle(acc)
viz.visualize_accidents(acc)

acc.to_csv("data/processed/acc.csv", index=True, encoding="utf-8")