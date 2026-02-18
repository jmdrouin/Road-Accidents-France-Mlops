import pandas as pd

def cleanup_vehicles(vehicles: pd.DataFrame):
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

    ## more cleaning

    # 1. filling Nans

    vehicles["traffic_direction_label"] = vehicles["traffic_direction_label"].fillna("Unknown")
    vehicles["fixed_obstacle_label"]   = vehicles["fixed_obstacle_label"].fillna("Unknown")
    vehicles["mobile_obstacle_label"]  = vehicles["mobile_obstacle_label"].fillna("Unknown")
    vehicles["impact_point_label"]     = vehicles["impact_point_label"].fillna("Unknown")
    vehicles["manoeuvre_label"]        = vehicles["manoeuvre_label"].fillna("Unknown")

    # 2. Check for outliers in occupants
    vehicles["occupants_group"] = vehicles["occupants"].apply(lambda x: str(x) if x <= 10 else "10+")

    ### Cleaning VEHICLES by Dropping columns before merging in master dataframe for visualisation

    drop_cols_vehicles = [
        "traffic_direction", "vehicle_category", "occupants",
        "fixed_obstacle", "mobile_obstacle", "impact_point", "manoeuvre",
        "fixed_obstacle_label"
    ]

    vehicles = vehicles.drop(columns=[c for c in drop_cols_vehicles if c in vehicles.columns])

    # Drop duplicate entries
    vehicles = vehicles.drop_duplicates(subset=["accident_id","vehicle_id"])

    return vehicles