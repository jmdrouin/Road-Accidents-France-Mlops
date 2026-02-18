import pandas as pd

def cleanup_users(users: pd.DataFrame):
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

    drop_cols_users = [
        "seat_position", "user_category", "injury_severity", "sex", "journey_purpose",
        "safety_equipment", "pedestrian_location", "pedestrian_action", "pedestrian_state"
    ]

    users = users.drop(columns=[c for c in drop_cols_users if c in users.columns])

    return users