import joblib
import pickle
from pathlib import Path
from src.models.split_and_transform import Columns

def export(model, feature_names, label_encoder, encoder, cat_imputer, num_imputer):
    # Create models directory
    Path("models").mkdir(exist_ok=True)

    print(f"✓ Feature names reconstructed: {len(feature_names)} features")
    print(f"  First 10: {feature_names[:10]}")
    print(f"  Sample age_group features: {[f for f in feature_names if 'age_group' in f][:5]}")

    # Save the final multiclass model
    joblib.dump(model, 'models/multiclass_lgbm_model.pkl')

    # Save the preprocessors
    joblib.dump(num_imputer, 'models/num_imputer.pkl')
    joblib.dump(cat_imputer, 'models/cat_imputer.pkl')
    joblib.dump(encoder, 'models/onehot_encoder.pkl')
    #joblib.dump(scaler, 'models/standard_scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    # Save feature names
    with open('models/feature_names_multiclass.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Save column lists
    with open('models/column_info.pkl', 'wb') as f:
        pickle.dump({
            'numeric_cols': Columns.numeric,
            'categorical_cols': Columns.categorical,
            'binary_cols': Columns.binary
        }, f)

    print("\n Multiclass model and preprocessors exported successfully!")
    print("   Model: lgbm_bs85_final")
    print("   Location: models/")
    print(f"   Files created: {len(list(Path('models').glob('*.pkl')))} PKL files")
