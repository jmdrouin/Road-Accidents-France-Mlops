import pandas as pd

class Maps:
    columns={
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
    }

    road_category = {
        1: "Highway",
        2: "National_road",
        3: "Departmental_road",
        4: "Communal_way",
        5: "Off_public_network",
        6: "Parking_lot",
        9: "other"
    }

    traffic_regime = {
        0: "Unknown",
        1: "One_way",
        2: "Bidirectional",
        3: "Separated_carriageways",
        4: "Variable_assignment"
    }

    reserved_lane = {
        0: "Unknown",
        1: "Bike_path",
        2: "Cycle_bank",
        3: "Reserved_channel"
    }

    road_profile = {
        0: "Unknown",
        1: "Flat",
        2: "Slope",
        3: "Hilltop",
        4: "Hill_bottom"
    }

    road_layout = {
        0: "Unknown",
        1: "Straight",
        2: "Curve_left",
        3: "Curve_right",
        4: "S-bend"
    }

    surface_condition = {
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

    infrastructure = {
        0: "Unknown",
        1: "Tunnel",
        2: "Bridge",
        3: "Interchange",
        4: "Railway",
        5: "Roundabout",
        6: "Pedestrian_area",
        7: "Tollzone"
    }

    situation = {
        0: "Unknown",
        1: "On_road",
        2: "On_emergency_stop_band",
        3: "On_verge",
        4: "On_sidewalk",
        5: "On_bike_path"
    }

    school_zone = {
        0: "Not_near_school",
        3: "Near_school",
        99: "Unknown"
    }

def cleanup_places(places: pd.DataFrame):
    places = places.rename(columns=Maps.columns)

    #mapping

    # --- Applying mappings ---
    places["road_category_label"] = places["road_category"].map(Maps.road_category)
    places["traffic_regime_label"] = places["traffic_regime"].map(Maps.traffic_regime)
    places["reserved_lane_label"] = places["reserved_lane"].map(Maps.reserved_lane)
    places["road_profile_label"] = places["road_profile"].map(Maps.road_profile)
    places["road_layout_label"] = places["road_layout"].map(Maps.road_layout)
    places["surface_condition_label"] = places["surface_condition"].map(Maps.surface_condition)
    places["infrastructure_label"] = places["infrastructure"].map(Maps.infrastructure)
    places["situation_label"] = places["situation"].map(Maps.situation)
    places["school_zone_label"] = places["school_zone"].map(Maps.school_zone)

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

    drop_cols_places = [
        "road_category", "road_number", "road_number_index", "road_number_suffix",
        "traffic_regime", "milestone", "milestone_distance",
        "reserved_lane", "road_profile", "road_layout",
        "surface_condition", "infrastructure", "situation", "school_zone"
    ]

    places = places.drop(columns=[c for c in drop_cols_places if c in places.columns])

    return places