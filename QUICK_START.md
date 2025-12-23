# 🚀 Quick Start: Enable Real Model Predictions

Your Streamlit app is ready to use trained models! Follow these steps to enable real predictions.

## ✅ Step 1: Export Models from Notebooks

### Multiclass Model (Step-3_Modeling_FINAL.ipynb)

1. **Open** `Step-3_Modeling_FINAL.ipynb`
2. **Scroll to the bottom** - you'll see the new cell: `### 🎯 Export Models for Streamlit App`
3. **Run the cell** - this will:
   - Create the `models/` directory
   - Save `lgbm_bs85_final` model
   - Save all preprocessors (imputers, encoders, scaler)
   - Save feature names and column info

### Binary Model (Bin_Modeling.ipynb)

1. **Open** `Bin_Modeling.ipynb`
2. **Scroll to the bottom** - you'll see the new cell: `### ⚖️ Export Binary Model for Streamlit App`
3. **Run the cell** - this will:
   - Save `model_bsm_top40` (binary model with 40 features)
   - Save the list of top 40 SHAP features

## ✅ Step 2: Verify Models Exported

Check that the `models/` folder now contains:

```
models/
├── multiclass_lgbm_model.pkl          ✓ Multiclass LightGBM model
├── binary_lgbm_model.pkl              ✓ Binary LightGBM model
├── num_imputer.pkl                    ✓ Numeric imputer
├── cat_imputer.pkl                    ✓ Categorical imputer
├── onehot_encoder.pkl                 ✓ OneHot encoder
├── standard_scaler.pkl                ✓ Standard scaler
├── label_encoder.pkl                  ✓ Label encoder
├── feature_names_multiclass.pkl       ✓ Feature names
├── column_info.pkl                    ✓ Column lists
└── top40_features_binary.pkl          ✓ Top 40 features for binary
```

**Expected:** 10 PKL files

## ✅ Step 3: Run Streamlit App

```powershell
streamlit run streamlit-app.py
```

Navigate to **Modelling** page and scroll to the prediction tools.

You should now see:
- ✅ **"Trained model loaded - Real predictions enabled"**

Instead of:
- ⚠️ "Trained models not found - Using demonstration mode"

## 🎯 Test the Predictions

### Multiclass Prediction Tool
1. Select features (vehicle type, impact point, etc.)
2. Click **"🔮 Predict Severity"**
3. See real predictions with probabilities for all 4 classes
4. View probability chart and factor analysis

### Binary Prediction Tool
1. Select features (includes collision type, lighting)
2. Click **"🔮 Predict Severity (Severe vs Not)"**
3. See severe/not severe prediction with probability
4. View risk factor breakdown

## 🔧 Troubleshooting

### Models not loading?
```powershell
# Check if models directory exists
ls models/

# Count PKL files (should be 10)
(ls models/*.pkl).Count
```

### Import errors?
```powershell
# Install required packages
pip install joblib scikit-learn lightgbm pandas numpy
```

### Need to re-export?
Just run the export cells again in the notebooks. They will overwrite the existing models.

## 📊 Model Performance

Once loaded, you'll get:

**Multiclass Model:**
- 70.2% accuracy
- 69.1% F1 macro
- 4-class prediction (Hospitalized/Unhospitalized/Killed/Unharmed)

**Binary Model:**
- 84.1% accuracy  
- 89.5% ROC-AUC
- 73.5% F1
- 2-class prediction (Severe vs Not Severe)

## 🎉 Success!

Your app now uses real trained models for predictions instead of the demonstration mode!
