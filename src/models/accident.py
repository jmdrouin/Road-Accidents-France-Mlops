from pydantic import create_model, Field
from typing import Literal

features_categorical = {
  "collision_label": {
    "Chain_three_plus": 9621,
    "Frontal_two_vehicles": 22088,
    "Multiple_three_plus": 11570,
    "No_collision": 10813,
    "Other_collision": 38125,
    "Rear_end_two_vehicles": 21976,
    "Side_two_vehicles": 60556
  },
  "surface_condition_label": {
    "Normal": 137588,
    "Unknown": 3691,
    "Wet": 29101,
    "other": 4369
  },
  "manoeuvre_label": {
    "Avoiding_obstacle": 9035,
    "Interaction": 4505,
    "Loss_of_control": 5137,
    "None": 10945,
    "Park/Start/Stop": 6594,
    "Reversing": 1772,
    "Roundabout": 21094,
    "Straight_ahead": 77630,
    "Turning": 23669,
    "other": 14368
  },
  "sex_label": {
    "Female": 45197,
    "Male": 127202,
    "null": 2350
  },
  "user_category_label": {
    "Driver": 170543,
    "other": 1856,
    "null": 2350
  },
  "seat_position_label": {
    "Driver_seat": 170246,
    "other": 2153,
    "null": 2350
  },
  "journey_purpose_label": {
    "Home-school": 2792,
    "Home-work": 25032,
    "Leisure": 76816,
    "Professional use": 23384,
    "Shopping": 4144,
    "Unknown": 23287,
    "other": 16944,
    "null": 2350
  },
  "vehicle_group": {
    "Bicycle": 22564,
    "Bus": 2999,
    "Car": 115935,
    "Motorcycle": 22885,
    "Truck": 5589,
    "other": 4777
  },
  "impact_group": {
    "Front": 101955,
    "Rear": 25201,
    "Side": 35531,
    "other": 12062
  },
  "road_group": {
    "Highway": 12964,
    "Local_road": 83292,
    "Major_road": 72655,
    "other": 5838
  },
  "weather_group": {
    "Clear": 147968,
    "Rain": 19579,
    "Snow": 2272,
    "other": 4930
  },
  "day_of_week": {
    "Friday": 29456,
    "Monday": 23945,
    "Saturday": 24601,
    "Sunday": 19252,
    "Thursday": 25632,
    "Tuesday": 25707,
    "Wednesday": 26156
  },
  "hour_group": {
    "Afternoon": 69580,
    "Evening": 47507,
    "Morning": 45832,
    "Night": 11830
  },
  "season": {
    "Autumn": 40898,
    "Spring": 41529,
    "Summer": 38787,
    "Winter": 53535
  },
  "age_group": {
    "Adult": 57258,
    "Child": 8376,
    "Middle_Aged": 50095,
    "Senior": 15518,
    "Young_Adult": 40543,
    "null": 2959
  }
}

features_binary = {"is_weekend": {
    "0": 130896,
    "1": 43853
  },
  "is_holiday": {
    "0": 171193,
    "1": 3556
  },
  "seatbelt_used": {
    "0": 72468,
    "1": 102281
  },
  "helmet_used": {
    "0": 140524,
    "1": 34225
  },
  "any_protection_used": {
    "0": 37512,
    "1": 137237
  },
  "protection_effective": {
    "0": 37512,
    "1": 137237
  },
  "motorcycle_side_impact": {
    "0": 171149,
    "1": 3600
  },
  "is_night": {
    "0": 131458,
    "1": 43291
  },
  "is_urban": {
    "0": 54295,
    "1": 120454
  }
}

def build_accident_model():
    fields = {}

    # numeric
    fields["age"] = (int | None, Field(default=None, ge=0, le=106))
    fields["lane_width"] = (float | None, Field(default=None, ge=0.5, le=200))

    for name, values in features_categorical.items():
        categories = list(values.keys())
        default = max(values, key=values.get)
        fields[name] = (
            Literal[tuple(categories)],
            Field(default=default)
        )
    
    for name, values in features_binary.items():
        default = int(max(values, key=values.get))
        fields[name] = (
            Literal[0, 1],
            Field(default=default),
        )

    Accident = create_model("Accident", **fields)

    return Accident