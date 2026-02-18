import pandas as pd

class Maps:
    columns = {
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
    }

    lighting = {
        1: "Daylight",
        2: "Twilight_or_dawn",
        3: "Night_without_public_lighting",
        4: "Night_with_public_lighting_not_functioning",
        5: "Night_with_public_lighting_on",
    }

    urban = {
        1: "Out_of_agglomeration",
        2: "Built_up_area"
    }

    intersection = {
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

    weather = {
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

    collision = {
        1: "Frontal_two_vehicles",
        2: "Rear_end_two_vehicles",
        3: "Side_two_vehicles",
        4: "Chain_three_plus",
        5: "Multiple_three_plus",
        6: "Other_collision",
        7: "No_collision",
    }

    gps = {
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


def cleanup_caract(caract: pd.DataFrame):
    ### CARACT (Characteristics) Dataframe - Auditing, Cleaning and Preprocessing
    #column renaming

    print("  [caract] applying name changes")
    caract = caract.rename(columns=Maps.columns)

    # Apply maps safely (only for existing columns)
    if "lighting" in caract.columns:
        caract["lighting_label"] = caract["lighting"].map(Maps.lighting).fillna("Unknown")
    if "urban_area" in caract.columns:
        caract["urban_label"] = caract["urban_area"].map(Maps.urban).fillna("Unknown")
    if "intersection" in caract.columns:
        caract["intersection_label"] = caract["intersection"].map(Maps.intersection).fillna("Unknown")
    if "weather" in caract.columns:
        caract["weather_label"] = caract["weather"].map(Maps.weather)
    if "collision_type" in caract.columns:
        caract["collision_label"] = caract["collision_type"].map(Maps.collision)
    if "gps" in caract.columns:
        caract["gps_label"] = caract["gps"].map(Maps.gps).fillna("Unknown").astype("category")

    #latitude and longitude

    print("  [caract] fixing geographic data")

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

    # valid geo
    caract["valid_geo"] = (
        caract["latitude_num"].between(41, 52, inclusive="both") &
        caract["longitude_num"].between(-5, 10, inclusive="both")
    )

    # year_month_day_date (expanding 2-digit years safely and build date)

    print("  [caract] fixing datetime data")

    caract["year"] = pd.to_numeric(caract["year"], errors="coerce")
    def expand_year(x):
        if pd.isna(x): return pd.NA
        x = int(x)
        return 2000 + x if x < 100 else x

    caract["year"] = pd.to_numeric(caract["year"], errors="coerce").apply(expand_year).astype("Int64")
    caract["month"] = pd.to_numeric(caract["month"], errors="coerce").astype("Int64")
    caract["day"] = pd.to_numeric(caract["day"], errors="coerce").astype("Int64")
    caract["date"] = pd.to_datetime(dict(year=caract["year"], month=caract["month"], day=caract["day"]), errors="coerce")

    if "hour" not in caract.columns or "minute" not in caract.columns:  
        hm = caract["hourminute"].apply(parse_hourminute)
        caract["hour"] = hm.apply(lambda t: t[0])
        caract["minute"] = hm.apply(lambda t: t[1])
        caract["time_hhmm"] = caract.apply(lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}" 
                                if pd.notna(r['hour']) and pd.notna(r['minute']) else pd.NA, axis=1)

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

    print("  [caract] finishing cleanup")
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

    drop_cols_caract = [
        "hourminute", "lighting", "urban_area", "intersection", "weather", "collision_type","commune_code",
        "address", "gps", "latitude", "longitude", "department"
    ]

    caract = caract.drop(columns=[c for c in drop_cols_caract if c in caract.columns])

    caract = caract.drop_duplicates(subset="accident_id")

    return caract
