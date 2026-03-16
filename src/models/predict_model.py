from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.util import last_file_in_folder
from src.models.split_and_transform import transform


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

def main() -> None:
    file = last_file_in_folder("models", "model_*.pkl")
    artifact = load_artifact(file)

    print("\nSAMPLE:")
    print(artifact["sample"])

    preds = predict_dataframe(artifact["sample"]["X"], artifact)

    proba_cols = [f"proba_{c}" for c in artifact["label_encoder"].classes_]
    pred_y = preds[proba_cols]
    true_y = artifact["sample"]["y"]

    print("\n-------------------------\nPreview:")
    print(pred_y)

    print("\n-------------------------\nVERIFICATION:")
    print(true_y)


if __name__ == "__main__":
    main()