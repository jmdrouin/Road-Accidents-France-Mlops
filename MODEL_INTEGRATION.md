# Implementing Real Model Predictions in Streamlit

## Overview
The Streamlit app now supports **real trained model predictions** for both multiclass and binary severity classification.

## Setup Instructions

### Step 1: Export Models from Notebooks

#### For Multiclass Model (`Step-3_Modeling_FINAL.ipynb`)

Add this code **at the end** of your notebook after training `lgbm_final` or `best_lgbm`:

```python
import joblib
import pickle
from pathlib import Path

# Create models directory
Path("models").mkdir(exist_ok=True)

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
```

#### For Binary Model (`Bin_Modeling.ipynb`)

Add this code **at the end** of your notebook after training `model_bsm_top40`:

```python
import joblib
import pickle

# Save the final binary model
joblib.dump(model_bsm_top40, 'models/binary_lgbm_model.pkl')

# Save top 40 features
with open('models/top40_features_binary.pkl', 'wb') as f:
    pickle.dump(top40_features, f)

print("✅ Binary model exported successfully!")
```

### Step 2: Verify Model Files

After running the code above, check that you have these files in the `models/` directory:

```
models/
├── multiclass_lgbm_model.pkl
├── binary_lgbm_model.pkl
├── num_imputer.pkl
├── cat_imputer.pkl
├── onehot_encoder.pkl
├── standard_scaler.pkl
├── label_encoder.pkl
├── feature_names_multiclass.pkl
├── column_info.pkl
└── top40_features_binary.pkl
```

### Step 3: Run Streamlit

```bash
streamlit run streamlit-app.py
```

Navigate to the **Modelling** page and scroll to the prediction tools. You should see:
- ✅ "Trained model loaded - Real predictions enabled"

## Features

### Multiclass Prediction Tool
- **Input**: Vehicle type, impact point, age group, weather, road type, time of day, safety features
- **Output**: Predicted severity (Hospitalized/Unhospitalized/Killed/Unharmed) with probabilities
- **Model**: LightGBM + Borderline SMOTE + Class Weights (70.2% accuracy)

### Binary Prediction Tool
- **Input**: All multiclass inputs + collision type, lighting conditions
- **Output**: Severe vs Not Severe with probability and risk breakdown
- **Model**: LightGBM + Borderline SMOTE + Top 40 SHAP Features (89.5% ROC-AUC)

## Fallback Mode

If models are not found, the app automatically falls back to **demonstration mode** using rule-based predictions. This ensures the app always works, even without trained models.

## Troubleshooting

### Models not loading
- Make sure you ran the export code in both notebooks
- Check that the `models/` directory exists and contains all required files
- Verify file paths are correct

### Import errors
- Install required packages: `pip install joblib scikit-learn lightgbm pandas numpy`

### Prediction errors
- Check that input features match the training data format
- Verify categorical values match training categories
- Review error messages in the Streamlit interface

## Notes

- Models are cached using `@st.cache_resource` for fast reloading
- Preprocessing pipeline matches the exact steps from the notebooks
- Feature engineering is automatically applied to user inputs
- Default values are used for features not exposed in the UI
