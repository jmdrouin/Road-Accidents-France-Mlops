from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class Config:
    test_size=0.2
    random_state=42

class Columns:
    target = "injury_severity_label"
    numeric = ["age", "lane_width"]
    categorical = [
        "collision_label", "surface_condition_label", "manoeuvre_label",
        "sex_label", "user_category_label", "seat_position_label", "journey_purpose_label",
        "vehicle_group", "impact_group", "road_group", "weather_group",
        "day_of_week", "hour_group", "season", "age_group"
    ]

    binary = [
        "is_weekend", "is_holiday", "seatbelt_used", "helmet_used",
        "any_protection_used", "protection_effective",
        "motorcycle_side_impact", "is_night", "is_urban"
    ]

def split(acc: pd.DataFrame):
    # Drop rows with missing target:
    acc = acc.dropna(subset=[Columns.target])
    acc = acc.drop(columns=["date"])

    X = acc.drop(columns=[Columns.target])
    y = acc[Columns.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.test_size, random_state=Config.random_state, stratify=y
    )

    return (X_train, X_test, y_train, y_test)


def prepare_transformers(X_train, y_train):
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(X_train[Columns.numeric])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_imputer.fit(X_train[Columns.categorical])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[Columns.categorical])

    scaler = StandardScaler()
    scaler.fit(X_train[Columns.numeric])

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y_train)

    categorical_feature_names = list(encoder.get_feature_names_out(Columns.categorical))
    names = Columns.numeric + categorical_feature_names + Columns.binary
    feature_names = [name.replace(" ", "_") for name in names]

    return (
        label_encoder, encoder, num_imputer, cat_imputer, scaler, feature_names
    )

#def transform(X_train, X_test, y_train, y_test):
def transform(X, y, label_encoder, encoder, num_imputer, cat_imputer, scaler, feature_names):
    X = X.copy()

    X[Columns.numeric] = num_imputer.transform(X[Columns.numeric])
    X[Columns.categorical] = cat_imputer.transform(X[Columns.categorical])
    X["age_group"] = X["age_group"].astype("category")

    # Add "Unknown" to categories if not already present
    if "Unknown" not in X["age_group"].cat.categories:
        X["age_group"] = X["age_group"].cat.add_categories(["Unknown"])

    # Fill NaNs with "Unknown"
    X["age_group"] = X["age_group"].fillna("Unknown")

    X_encoded = encoder.transform(X[Columns.categorical])
    X_scaled = scaler.transform(X[Columns.numeric])

    # Combine all parts
    X_final = np.hstack([
        X_scaled,          # numeric
        X_encoded,         # categorical
        X[Columns.binary].values  # binary flags
    ])

    if y is not None:
        y = label_encoder.transform(y)
    X_final = pd.DataFrame(X_final, columns=feature_names)
    return (X_final, y)
