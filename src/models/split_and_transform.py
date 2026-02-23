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

def split_and_transform(acc: pd.DataFrame):
    # Drop rows with missing target:
    acc = acc.dropna(subset=[Columns.target])

    X = acc.drop(columns=[Columns.target])
    y = acc[Columns.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.test_size, random_state=Config.random_state, stratify=y
    )

    num_imputer = SimpleImputer(strategy="median")
    X_train[Columns.numeric] = num_imputer.fit_transform(X_train[Columns.numeric])
    X_test[Columns.numeric] = num_imputer.transform(X_test[Columns.numeric])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[Columns.categorical] = cat_imputer.fit_transform(X_train[Columns.categorical])
    X_test[Columns.categorical] = cat_imputer.transform(X_test[Columns.categorical])

    # Check again for NaNs
    print("Remaining NaNs in train:", X_train.isna().sum().sum())
    print("Remaining NaNs in test:", X_test.isna().sum().sum())
    # Show which columns still have NaNs
    print("Train NaNs per column:\n", X_train.isna().sum()[X_train.isna().sum() > 0])
    print("\nTest NaNs per column:\n", X_test.isna().sum()[X_test.isna().sum() > 0])
    # Ensure dtype is categorical
    X_train["age_group"] = X_train["age_group"].astype("category")
    X_test["age_group"] = X_test["age_group"].astype("category")

    # Add "Unknown" to categories if not already present
    if "Unknown" not in X_train["age_group"].cat.categories:
        X_train["age_group"] = X_train["age_group"].cat.add_categories(["Unknown"])
        X_test["age_group"] = X_test["age_group"].cat.add_categories(["Unknown"])

    # Fill NaNs with "Unknown"
    X_train["age_group"] = X_train["age_group"].fillna("Unknown")
    X_test["age_group"] = X_test["age_group"].fillna("Unknown")

    # Final check
    print("Remaining NaNs in train:", X_train.isna().sum().sum())
    print("Remaining NaNs in test:", X_test.isna().sum().sum())
    #### Encoding Categorical Features

    # Initialize encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Fit on train, transform both train and test
    X_train_encoded = encoder.fit_transform(X_train[Columns.categorical])
    X_test_encoded = encoder.transform(X_test[Columns.categorical])

    print("Encoded train shape:", X_train_encoded.shape)
    print("Encoded test shape:", X_test_encoded.shape)
    #### Scale Numeric Features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train[Columns.numeric])
    X_test_scaled = scaler.transform(X_test[Columns.numeric])

    print("Scaled train shape:", X_train_scaled.shape)
    print("Scaled test shape:", X_test_scaled.shape)
    #### Combine All Features

    # Combine all parts
    X_train_final = np.hstack([
        X_train_scaled,          # numeric
        X_train_encoded,         # categorical
        X_train[Columns.binary].values  # binary flags
    ])

    X_test_final = np.hstack([
        X_test_scaled,
        X_test_encoded,
        X_test[Columns.binary].values
    ])

    print("Final train shape:", X_train_final.shape)
    print("Final test shape:", X_test_final.shape)
    #### Encoding Target Labels 
    # Encoding Target Labels (y_train & y_test)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    return (
        X_train_final, X_test_final,
        y_train_enc, y_test_enc,
        label_encoder, encoder, num_imputer, cat_imputer
    )
