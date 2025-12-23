# 🚗 Road Accidents in France - Data Science Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data science project analyzing road accident data in France (2005-2016) to predict accident severity using machine learning.

## 📊 Project Overview

This project demonstrates an end-to-end machine learning pipeline for predicting road accident severity, featuring:
- **Data Mining & Visualization**: Temporal, geographical, and feature distribution analysis
- **Feature Engineering**: 27 engineered features from 51 raw features
- **Machine Learning**: Binary and multiclass classification with LightGBM
- **Interactive Web App**: Streamlit dashboard with real-time predictions

### Key Results
- 🎯 **Accuracy**: 73.9% (Binary model)
- 📈 **ROC-AUC**: 83.4%
- 🔍 **Recall (Severe)**: 79%
- ⚡ **Inference**: <10ms per prediction

---

## 🚀 Quick Start

### Option 1: Run with Sample Data (Demonstrative)
Use the included sample datasets (10,000 rows each) for quick testing:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Road-Accidents-France.git
cd Road-Accidents-France

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app with sample data
streamlit run streamlit-app.py
```

The app will use the sample CSV files located in the `sample_data/` folder.

### Option 2: Run with Full Dataset
For complete analysis with 1.4M+ records:

```bash
# 1. Install kagglehub
pip install kagglehub

# 2. Download the full dataset
python download_data.py

# 3. Run the Streamlit app
streamlit run streamlit-app.py
```

---

## 📥 Dataset Download

The full dataset contains **1,412,032 accident records** from France (2005-2016).

### Automatic Download (Recommended)
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ahmedlahlou/accidents-in-france-from-2005-to-2016")
print("Path to dataset files:", path)
```

### Manual Download
1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)
2. Download the following files:
   - `caracteristics.csv` - Accident characteristics (839,985 rows)
   - `places.csv` - Road and location details
   - `users.csv` - Person information (1,876,005 rows)
   - `vehicles.csv` - Vehicle information (1,433,389 rows)
   - `holidays.csv` - French public holidays
3. Place them in the project root directory

---

## 📁 Project Structure

```
Road-Accidents-France/
│
├── 📓 Notebooks (Analysis Pipeline)
│   ├── Step_1_Data mining_DataViz.ipynb       # Exploratory Data Analysis
│   ├── Step 2_Pre-processing_feature-eng.ipynb # Feature Engineering
│   ├── Step-3_Modeling_FINAL.ipynb            # Model Training & Evaluation
│   └── Bin_Modeling.ipynb                     # Binary Classification Model
│
├── 📊 Data Files
│   ├── sample_data/                           # Sample datasets (10K rows)
│   │   ├── caracteristics_sample.csv
│   │   ├── places_sample.csv
│   │   ├── users_sample.csv
│   │   └── vehicles_sample.csv
│   ├── holidays.csv                           # French holidays
│   ├── acc.csv                                # Merged dataset (generated)
│   └── master_acc.csv                         # Final ML-ready dataset (generated)
│
├── 🤖 Models
│   └── models/                                # Trained ML models (exported)
│       ├── binary_lgbm_classweight.pkl
│       ├── num_imputer.pkl
│       ├── cat_imputer.pkl
│       ├── onehot_encoder.pkl
│       └── standard_scaler.pkl
│
├── 🌐 Web Application
│   └── streamlit-app.py                       # Interactive dashboard
│
├── 📜 Documentation
│   ├── README.md                              # This file
│   ├── MODEL_INTEGRATION.md                   # Model deployment guide
│   └── RESTRUCTURING_SUMMARY.md               # Project structure notes
│
└── 🛠️ Utilities
    ├── download_data.py                       # Dataset download script
    ├── create_samples.py                      # Create sample datasets
    ├── requirements.txt                       # Python dependencies
    └── .gitignore                             # Git ignore rules
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended for full dataset)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Road-Accidents-France.git
cd Road-Accidents-France
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download data** (choose one):
   - **Quick test**: Use included sample data (already in `sample_data/`)
   - **Full analysis**: Run `python download_data.py`

5. **Run notebooks** (optional):
```bash
jupyter notebook
```
Execute notebooks in order: Step_1 → Step 2 → Step-3_Modeling_FINAL → Bin_Modeling

6. **Launch Streamlit app**
```bash
streamlit run streamlit-app.py
```

The app will open automatically at `http://localhost:8501`

---

## 💻 Usage

### Streamlit Web Application

The interactive dashboard contains three main pages:

#### 1. 📊 Data Mining & Visualization
- Temporal patterns (yearly, monthly, hourly trends)
- Geographical distribution with interactive maps
- Feature distributions and correlations
- Severity analysis by various factors

#### 2. 🔧 Pre-processing & Feature Engineering
- Data cleaning pipeline visualization
- Feature transformation flowcharts
- Final dataset statistics
- Engineering logic explanations

#### 3. 🚀 Modeling & Optimization
- Model selection journey
- Performance comparison (multiclass vs binary)
- SHAP explainability analysis
- **Interactive prediction tool** - Test the model with custom inputs!

### Running Notebooks

Execute notebooks sequentially for full analysis:

1. **Step_1_Data mining_DataViz.ipynb**
   - Load and explore raw data
   - Generate visualizations
   - Identify patterns and anomalies

2. **Step 2_Pre-processing_feature-eng.ipynb**
   - Merge datasets
   - Clean and transform features
   - Create engineered features
   - Export `acc.csv` and `master_acc.csv`

3. **Step-3_Modeling_FINAL.ipynb**
   - Multiclass classification (4 severity levels)
   - Compare sampling strategies (SMOTE, Borderline SMOTE)
   - Hyperparameter optimization

4. **Bin_Modeling.ipynb**
   - Binary classification (Severe vs Not Severe)
   - Model comparison and evaluation
   - Export trained models for deployment
   - **Run this to enable predictions in Streamlit app!**

---

## 🎯 Machine Learning Pipeline

### Problem Statement
Predict injury severity in road accidents using:
- **Target (Multiclass)**: Hospitalized, Slight Injury, Killed, Uninjured
- **Target (Binary)**: Severe (Hospitalized + Killed) vs Not Severe
- **Features**: 85 (after encoding)
- **Algorithm**: LightGBM with class weighting

### Feature Engineering Highlights
- **Safety Equipment**: `seatbelt_used`, `helmet_used`, `protection_effective`
- **Temporal**: `hour_group`, `day_of_week`, `is_weekend`, `is_holiday`
- **Contextual**: `is_night`, `is_urban`, `lane_width`, `weather_group`
- **Vehicle**: `vehicle_group`, `impact_group`, `motorcycle_side_impact`

### Model Performance

| Model | Accuracy | F1 (Severe) | Recall (Severe) | ROC-AUC |
|-------|----------|-------------|-----------------|---------|
| **Binary + Class Weights** ⭐ | **73.9%** | **56.6%** | **79%** | **83.4%** |
| Binary + SMOTE | 81.6% | 51.9% | 46% | 82.6% |
| Multiclass + Borderline SMOTE | 64.3% | 50.1% | 22% | N/A |

**Recommended**: Binary model with class weights for production deployment.

---

## 📦 Dependencies

Core libraries:
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computing
- `scikit-learn >= 1.3.0` - ML algorithms
- `lightgbm >= 4.0.0` - Gradient boosting
- `imblearn >= 0.11.0` - Handling imbalanced data
- `streamlit >= 1.28.0` - Web dashboard
- `plotly >= 5.17.0` - Interactive visualizations
- `pyproj >= 3.6.0` - Coordinate transformations

See `requirements.txt` for complete list.

---

## 🌐 Streamlit App Features

### 🎨 Interactive Visualizations
- 📅 Temporal trends with holiday analysis
- 🗺️ Geographic heatmaps (mainland France)
- 📊 Feature distributions with smart sampling
- 🔗 Correlation matrices

### 🔮 Prediction Tool
Enter accident details and get:
- Severity prediction (Severe / Not Severe)
- Confidence score
- Risk factor analysis
- Protective factor identification

### 📈 Model Insights
- LazyPredict screening results
- Cross-validation metrics
- SHAP feature importance
- Deployment recommendations

---

## 🧹 Cleaning Up (Development)

To remove generated files and start fresh:

```bash
# Remove generated data files (keep originals)
rm acc.csv master_acc.csv

# Remove model files
rm -rf models/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +
```

Or use the cleanup utility:
```bash
python cleanup.py --all
```

---

## 📊 Sample Data Information

The `sample_data/` folder contains reduced datasets for quick testing:
- **10,000 rows** from each CSV file
- **Randomly sampled** to maintain distribution
- **Sufficient** for demonstrating all app features
- **Fast loading** (<1 second)

To regenerate sample data:
```bash
python create_samples.py --rows 10000
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Z. Malik** - Data Engineering & ML Pipeline
- **M. Peuyn** - Feature Engineering & Visualization

---

## 🙏 Acknowledgments

- Dataset: [Ahmed Lahlou on Kaggle](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)
- Original data source: [French Government Open Data](https://www.data.gouv.fr/)
- Inspired by road safety research and emergency response optimization

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [Your Email]

---

## 🔄 Version History

- **v1.0.0** (2025-12) - Initial release
  - Complete data pipeline
  - Binary and multiclass models
  - Interactive Streamlit dashboard
  - Prediction API

---

**⭐ If you find this project helpful, please star the repository!**
