"""
Export trained models and preprocessing artifacts for Streamlit deployment
Run this script after training models in the notebooks to save them for the app
"""
import joblib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Create models directory
Path("models").mkdir(exist_ok=True)

print("=" * 60)
print("MODEL EXPORT SCRIPT")
print("=" * 60)
print("\nThis script will help you export your trained models.")
print("Make sure you have run the modeling notebooks first!\n")

# Instructions for multiclass model
print("📊 MULTICLASS MODEL (Step-3_Modeling_FINAL.ipynb)")
print("-" * 60)
print("Add this code to the end of your notebook:")
print("""
# After training best_lgbm or lgbm_final model:
import joblib
import pickle

# Save the final multiclass model
joblib.dump(lgbm_final, 'models/multiclass_lgbm_model.pkl')

# Save the preprocessors
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')
joblib.dump(encoder, 'models/onehot_encoder.pkl')
joblib.dump(scaler, 'models/standard_scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Save feature names
with open('models/feature_names_multiclass.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Save column lists
with open('models/column_info.pkl', 'wb') as f:
    pickle.dump({
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'binary_cols': binary_cols
    }, f)

print("✅ Multiclass model exported successfully!")
""")

print("\n⚖️ BINARY MODEL (Bin_Modeling.ipynb)")
print("-" * 60)
print("Add this code to the end of your notebook:")
print("""
# After training model_bsm_top40 (final binary model):
import joblib

# Save the final binary model
joblib.dump(model_bsm_top40, 'models/binary_lgbm_model.pkl')

# Save top 40 features
with open('models/top40_features_binary.pkl', 'wb') as f:
    pickle.dump(top40_features, f)

# The preprocessors are shared with multiclass, so no need to save again

print("✅ Binary model exported successfully!")
""")

print("\n" + "=" * 60)
print("After running the code above in your notebooks, you can use")
print("the models in streamlit-app.py for real predictions!")
print("=" * 60)
