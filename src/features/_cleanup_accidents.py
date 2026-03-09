import pandas as pd
import numpy as np

def prepare_accidents_data(acc: pd.DataFrame):
    acc = acc.set_index("accident_id")

    drop_cols_acc = [
        "Unnamed: 0","vehicle_id",
        "date","minute", "time_hhmm", "gps_label", "holiday_name", #redundant
        "longitude_num", "latitude_num","valid_geo",  # useful only for spatial analysis , dropping for modeling efficiency
        "mobile_obstacle_label",          # drop if too sparse
        "pedestrian_crossing_width", "mobile_obstacle_label", "reserved_lane_label", #too sparse
        "pedestrian_location_label", "pedestrian_action_label", "pedestrian_state_label" #mostly “Not_pedestrian”
    ]

    acc = acc.drop(columns=[c for c in drop_cols_acc if c in acc.columns])

    # 1. Cast categorical columns
    cat_cols = [
        "lighting_label", "urban_label", "intersection_label", "weather_label", "collision_label",
        "road_category_label", "traffic_regime_label", "road_profile_label", "road_layout_label",
        "surface_condition_label", "infrastructure_label", "situation_label", "school_zone_label",
        "traffic_direction_label", "vehicle_category_label", "impact_point_label", "manoeuvre_label",
        "occupants_group", "sex_label", "user_category_label", "injury_severity_label",
        "seat_position_label", "journey_purpose_label", "safety_equipment_type", "safety_equipment_usage",
        "time_of_day", "season"
    ]
    acc[cat_cols] = acc[cat_cols].astype("category")

    # 2. Cast numeric columns
    acc = acc.astype({
        "year": "int16",
        "month": "int8",
        "day": "int8",
        "hour": "int8",
        "num_lanes": "float32",
        "road_width": "float32",
        "birth_year": "float32"
    })

    # 3. Cast booleans
    acc = acc.astype({
        "is_weekend": "bool",
        "is_holiday": "bool"
    })

    # STEP 1: Clean num_lanes
    # Rule: num_lanes <= 0 or > 10 is invalid → set to NaN
    acc.loc[(acc["num_lanes"] <= 0) | (acc["num_lanes"] > 10), "num_lanes"] = np.nan

    # STEP 2: Clean road_width
    # Rule: road_width <= 0 or > 200 is unrealistic → set to NaN
    acc.loc[(acc["road_width"] <= 0) | (acc["road_width"] > 200), "road_width"] = np.nan

    # STEP 3: Clean birth_year
    # Rule: birth_year < 1900 or > 2010 is invalid → set to NaN
    acc.loc[(acc["birth_year"] < 1900) | (acc["birth_year"] > 2010), "birth_year"] = np.nan

    # STEP 4: Derive age
    # Safe calculation: fill missing birth_year with NaN, then subtract
    acc["age"] = acc["year"] - acc["birth_year"]
    acc["age"] = acc["age"].astype("float32")  # keep float for NaNs

    # STEP 5: Create age groups
    bins = [0, 17, 25, 40, 60, 100]
    labels = ["Child", "Young_Adult", "Adult", "Middle_Aged", "Senior"]
    acc["age_group"] = pd.cut(acc["age"], bins=bins, labels=labels)

    # Define threshold for rare categories
    # We'll group categories that occur in less than 1% of rows
    threshold = 0.01

    # STEP 2: Loop through all categorical columns
    cat_cols = acc.select_dtypes(include="category").columns

    for col in cat_cols:
        # Calculate frequency proportions
        freqs = acc[col].value_counts(normalize=True)
        
        # Identify rare categories
        rare = freqs[freqs < threshold].index
        
        if len(rare) > 0:
            if "other" not in acc[col].cat.categories:
                acc[col] = acc[col].cat.add_categories(["other"])
            acc.loc[acc[col].isin(rare), col] = "other"

    ### FEATURE ENGINEERING
    #### Safety Equipment Feature Engineering 
    # -------------------------------------------------------------
    # STEP 1: Seatbelt usage flag
    # -------------------------------------------------------------
    acc["seatbelt_used"] = (
        (acc["safety_equipment_type"] == "Seat_belt") &
        (acc["safety_equipment_usage"] == "Used")
    ).astype("bool")

    # -------------------------------------------------------------
    # STEP 2: Helmet usage flag
    # -------------------------------------------------------------
    acc["helmet_used"] = (
        (acc["safety_equipment_type"] == "Helmet") &
        (acc["safety_equipment_usage"] == "Used")
    ).astype("bool")

    # -------------------------------------------------------------
    # STEP 3: Any protection used flag
    # -------------------------------------------------------------
    acc["any_protection_used"] = (
        acc["safety_equipment_usage"] == "Used"
    ).astype("bool")

    # -------------------------------------------------------------
    # STEP 4: Protection effectiveness flag
    # -------------------------------------------------------------
    acc["protection_effective"] = (
        (acc["any_protection_used"]) &
        (~acc["safety_equipment_usage"].isin(["Undetermined", "Not_used"]))
    ).astype("bool")

    #### Vehicle & Impact Feature Engineering
    # -------------------------------------------------------------
    # STEP 1: Vehicle grouping
    # -------------------------------------------------------------
    vehicle_map = {
        "Car(Quad/Passenger)": "Car",
        "Motorcycle": "Motorcycle",
        "Cycle": "Bicycle",
        "Utility(Truck/Agri)": "Truck",
        "Bus/Coach": "Bus"
    }
    acc["vehicle_group"] = acc["vehicle_category_label"].map(vehicle_map).fillna("other")

    # -------------------------------------------------------------
    # STEP 2: Impact grouping
    # -------------------------------------------------------------
    impact_map = {
        "Front": "Front",
        "Front_left": "Front",
        "Front_right": "Front",
        "Rear": "Rear",
        "Rear_left": "Rear",
        "Rear_right": "Rear",
        "Left_side": "Side",
        "Right_side": "Side"
    }
    acc["impact_group"] = acc["impact_point_label"].map(impact_map).fillna("other")

    # -------------------------------------------------------------
    # STEP 3: Interaction feature (Vehicle × Impact)
    # -------------------------------------------------------------
    acc["motorcycle_side_impact"] = (
        (acc["vehicle_group"] == "Motorcycle") &
        (acc["impact_group"] == "Side")
    ).astype("bool")

    #### Road & Accident Context Feature Engineering
    # -------------------------------------------------------------
    # STEP 1: Lighting condition flag
    # -------------------------------------------------------------
    acc["is_night"] = acc["lighting_label"].isin([
        "Night_with_public_lighting_on",
        "Night_without_public_lighting",
        "Night_with_public_lighting_not_functioning"
    ]).astype("bool")

    # -------------------------------------------------------------
    # STEP 2: Urban vs Rural flag
    # -------------------------------------------------------------
    acc["is_urban"] = (acc["urban_label"] == "Built_up_area").astype("bool")

    # -------------------------------------------------------------
    # STEP 3: Lane width feature
    # -------------------------------------------------------------
    acc["lane_width"] = acc["road_width"] / acc["num_lanes"].replace(0, np.nan)

    # Clean unrealistic values
    acc.loc[(acc["num_lanes"].isna()) | (acc["road_width"].isna()), "lane_width"] = np.nan

    # -------------------------------------------------------------
    # STEP 4: Road category simplification
    # -------------------------------------------------------------
    road_map = {
        "Highway": "Highway",
        "National_road": "Major_road",
        "Departmental_road": "Major_road",
        "Communal_way": "Local_road"
    }
    acc["road_group"] = acc["road_category_label"].map(road_map).fillna("other")

    # -------------------------------------------------------------
    # STEP 5: Weather condition grouping
    # -------------------------------------------------------------
    weather_map = {
        "Normal": "Clear",
        "Cloudy": "Clear",
        "Light_rain": "Rain",
        "Heavy_rain": "Rain",
        "Snow_or_hail": "Snow",
        "Fog_or_smoke": "Fog",
        "Strong_wind_or_storm": "other",
        "Dazzling": "other",
        "Other": "other"
    }
    acc["weather_group"] = acc["weather_label"].map(weather_map).fillna("other")

    #### Temporal Feature Engineering
    # -------------------------------------------------------------
    # STEP 1: Accident date
    # -------------------------------------------------------------
    acc["date"] = pd.to_datetime(acc[["year", "month", "day"]])

    # -------------------------------------------------------------
    # STEP 2: Day of week
    # -------------------------------------------------------------
    acc["day_of_week"] = acc["date"].dt.day_name()

    # -------------------------------------------------------------
    # STEP 3: Hour group
    # -------------------------------------------------------------
    acc["hour_group"] = pd.cut(
        acc["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        right=False
    )

    # -------------------------------------------------------------
    # STEP 4: Weekend flag
    # -------------------------------------------------------------
    acc["is_weekend"] = acc["day_of_week"].isin(["Saturday", "Sunday"]).astype("bool")

    #### Pruning the dataset
    # -------------------------------------------------------------
    # STEP 1: Define columns to drop
    # -------------------------------------------------------------
    drop_cols = [
        # Raw safety equipment columns (replaced by engineered flags)
        "safety_equipment_type",
        "safety_equipment_usage",
        
        # Raw vehicle/impact columns (replaced by grouped features)
        "vehicle_category_label",
        "impact_point_label",
        
        # Raw road context columns (replaced by grouped features)
        "lighting_label",
        "urban_label",
        "road_category_label",
        "weather_label",
        
        # Raw temporal columns (replaced by engineered features)
        "year", "month", "day", "hour",
        "time_of_day", # redundant, since already have hour_group and is_night.

        "birth_year",   # replaced by age + age_group
        "date",         # replaced by day_of_week, hour_group, season, is_weekend
        "num_lanes",    # captured in lane_width
        "road_width",   # captured in lane_width
        
        # Very imbalanced / sparse
        "infrastructure_label",  # ~89% Unknown
        "school_zone_label",     # ~39% Unknown, ~5% Near_school
        "occupants_group",       # ~99% "0"
        "situation_label",       # ~89% "On_road"
        "traffic_direction_label", # ~92% "Same_direction"
        "intersection_label" , #often skewed (majority “Out_of_intersection”)
        "traffic_regime_label", #skewed (mostly “Bidirectional")
        "road_profile_label", #skewed (mostly "Flat")
        "road_layout_label", #skewed (mostly "Straight")
        
    ]

    # -------------------------------------------------------------
    # STEP 2: Drop them safely
    # -------------------------------------------------------------
    acc = acc.drop(columns=drop_cols, errors="ignore")

    return acc