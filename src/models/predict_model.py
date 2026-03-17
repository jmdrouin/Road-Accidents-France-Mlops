from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.util import last_file_in_folder
from src.models.split_and_transform import transform
from src.models.accident import build_accident_model


def load_artifact(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)

    required_keys = {
        "model",
        "num_imputer",
        "cat_imputer",
        "onehot_encoder",
        "label_encoder",
        "feature_names",
        "column_info",
        "sample"
    }
    missing = required_keys - set(artifact.keys())
    if missing:
        raise KeyError(f"Artifact missing required keys: {sorted(missing)}")

    return artifact

def prepare_features(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    label_encoder = artifact["label_encoder"]
    encoder = artifact["onehot_encoder"]
    num_imputer = artifact["num_imputer"]
    cat_imputer = artifact["cat_imputer"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    X, y = transform(df, None, label_encoder, encoder, num_imputer, cat_imputer, scaler, feature_names)
    return X

def predict_dataframe(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]

    X = prepare_features(df, artifact)

    y_pred_encoded = model.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    result = df.copy()
    result["prediction"] = y_pred

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        class_labels = label_encoder.inverse_transform(range(proba.shape[1]))

        for i, class_name in enumerate(class_labels):
            result[f"proba_{class_name}"] = proba[:, i]

    return result

def latest_model_artifact():
    file = last_file_in_folder("models", "model_*.pkl")
    return load_artifact(file)

# TODO: This should be a unit test (checking if the data was retransformed correctly)
def main() -> None:
    artifact = latest_model_artifact()

    print("\n======================\nSAMPLE:")
    print(artifact["sample"]["X"])

    print("\n======================\nCOLS:")
    print(artifact["sample"]["X"].columns)

    print("\n======================\nFEATURES:")
    print(artifact["column_info"])


    preds = predict_dataframe(artifact["sample"]["X"], artifact)

    proba_cols = [f"proba_{c}" for c in artifact["label_encoder"].classes_]
    pred_y = preds[proba_cols]
    true_y = artifact["sample"]["y"]

    print("\n-------------------------\nPreview:")
    print(pred_y)

    print("\n-------------------------\nVERIFICATION:")
    print(true_y)


PREDICT_COLUMNS = [
    'timestamp', 'collision_label', 'is_weekend', 'season',
    'surface_condition_label', 'manoeuvre_label', 'sex_label',
    'user_category_label', 'seat_position_label', 'journey_purpose_label',
    'is_holiday', 'age', 'age_group', 'seatbelt_used', 'helmet_used',
    'any_protection_used', 'protection_effective', 'vehicle_group',
    'impact_group', 'motorcycle_side_impact', 'is_night', 'is_urban',
    'lane_width', 'road_group', 'weather_group', 'day_of_week',
    'hour_group'
]

def predict_accident(accident):
    data = accident.model_dump()
    data["timestamp"] = None
    df = pd.DataFrame([data])[PREDICT_COLUMNS]
    pred = predict_dataframe(df, latest_model_artifact())
    print(pred)

if __name__ == "__main__":
    accident = build_accident_model()(
        manoeuvre_label="Turning",
        vehicle_group="Car",
        age=45,
        lane_width=3.5
    )
    predict_accident(accident)
    #main()