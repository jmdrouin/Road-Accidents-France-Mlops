import joblib
from pathlib import Path
from src.models.split_and_transform import Columns
import datetime

def export(model, feature_names, label_encoder, encoder, cat_imputer, num_imputer, info, metrics):
    # Create models directory
    Path("models").mkdir(exist_ok=True)

    print(f"✓ Feature names reconstructed: {len(feature_names)} features")
    print(f"  First 10: {feature_names[:10]}")
    print(f"  Sample age_group features: {[f for f in feature_names if 'age_group' in f][:5]}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "type": "multiclass lgbm model",
        "timestamp": timestamp,
        "info": info,
        "metrics": metrics,
        "model": model,
        "booster": model.booster_,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "onehot_encoder": encoder,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "column_info": {
            'numeric_cols': Columns.numeric,
            'categorical_cols': Columns.categorical,
            'binary_cols': Columns.binary
        }
    }

    destination = f"models/model_{timestamp}.pkl"
    joblib.dump(artifact, destination)

    print("\n Multiclass model and preprocessors exported successfully!")
    print("   Model: lgbm_bs85_final")
    print(f"   Location: {destination}")