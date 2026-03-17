import sqlite3
import pandas as pd
import json

from src.util import single_file_in_folder


FEATURES = {
    "numeric_cols": ["age", "lane_width"],
    "categorical_cols": [
        "collision_label",
        "surface_condition_label",
        "manoeuvre_label",
        "sex_label",
        "user_category_label",
        "seat_position_label",
        "journey_purpose_label",
        "vehicle_group",
        "impact_group",
        "road_group",
        "weather_group",
        "day_of_week",
        "hour_group",
        "season",
        "age_group",
    ],
    "binary_cols": [
        "is_weekend",
        "is_holiday",
        "seatbelt_used",
        "helmet_used",
        "any_protection_used",
        "protection_effective",
        "motorcycle_side_impact",
        "is_night",
        "is_urban",
    ],
}


def load_single_table(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        conn,
    )["name"].tolist()

    if len(tables) != 1:
        raise RuntimeError(f"Expected exactly 1 table, found {len(tables)}: {tables}")

    df = pd.read_sql(f"SELECT * FROM {tables[0]}", conn)
    conn.close()
    return df


def describe_features(df: pd.DataFrame) -> None:

    result = {}

    for col in FEATURES["numeric_cols"]:
        print(f"\n{col} (numeric)")
        print(" min:", df[col].min())
        print(" max:", df[col].max())

    for col in FEATURES["categorical_cols"] + FEATURES["binary_cols"]:
        counts = df[col].value_counts(dropna=False).sort_index()

        result[col] = {
            ("null" if pd.isna(k) else k): int(v)
            for k, v in counts.items()
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    file = single_file_in_folder("data/processed", "accidents_*.db")
    df = load_single_table(file)
    describe_features(df)