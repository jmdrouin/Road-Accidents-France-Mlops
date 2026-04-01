'''
import os
from dotenv import load_dotenv
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
env_path = root_path / '.env'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f".env not found in root")

repo_owner = os.environ['MLFLOW_TRACKING_USERNAME']
repo_name = os.environ['MLFLOW_TRACKING_REPO']
token = os.environ['MLFLOW_TRACKING_PASSWORD']
repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}.s3"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = repo_url
os.environ['AWS_ACCESS_KEY_ID'] = token
os.environ['AWS_SECRET_ACCESS_KEY'] = token
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
'''

import streamlit as st
from pathlib import Path

from mlflow_config_loader import setup_mlflow
setup_mlflow(1)

import joblib
import mlflow
from mlflow.artifacts import download_artifacts

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# functions
# ----------------------------------------------

def get_best_model():
    base_path = Path(__file__).resolve().parents[1] / "models" / "best_model_bundle"
    model_path_str = str(base_path)
    model_path = base_path / "best_model_bundle.pkl"

    if model_path.exists():
        model_bundle = joblib.load(model_path)
        print("Model load succeeded!")
        return model_bundle
    else:
        print(f"File not found: {model_path}")
        raise FileNotFoundError(f"File not found: {model_path}")

def get_model_metrics(model_bundle):
    f1_macro = model_bundle["metrics"]["f1_macro"]
    accuracy = model_bundle["metrics"]["accuracy"]
    balanced_accuracy = model_bundle["metrics"]["balanced_accuracy"]
    return f1_macro, accuracy, balanced_accuracy

def kpi_card(title, value, color="#2E86C1"):
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white;">
            <h5 style="margin:0;">{title}</h5>
            <p style="font-size:24px; margin:0;"><b>{value}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def model_info(bundle):
    run_id = bundle["run_id"]
    params = bundle["params"]
    #boosting_type = bundle["params"].get("boosting_type", "Not found")
    
    return run_id, params

def mlflow_link_markdown(mlflow_ui_url):
    st.markdown("""
    <style>
    div.stLinkButton > a {
        background-color: #28a745 !important;
        color: white !important;
        border-radius: 5px;
        border: none;
    }
    div.stLinkButton > a:hover {
        background-color: #218838 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.link_button("MLflow Dashboard", mlflow_ui_url, type="primary")
    st.markdown(f'<p style="font-size: 12px;">{mlflow_ui_url}</p>', unsafe_allow_html=True)

def mlflow_link(mlflow_ui_url):

    st.link_button("Dagshub Dashboard", mlflow_ui_url, type="primary")
    st.markdown(f'<p style="font-size: 12px;">{mlflow_ui_url}</p>', unsafe_allow_html=True)

def display_confusion_matrix(run_id):

    client = mlflow.tracking.MlflowClient()
    
    # rel path in artifacts
    artifact_path = "plots/confusion_matrix.png"
    
    try:
        # load pic from MLflow into temp folder
        local_path = client.download_artifacts(run_id, artifact_path)
        
        # show in Streamlit
        st.image(local_path, caption=f"Confusion Matrix (Run: {run_id})")
    except Exception as e:
        st.error(f"Matrix couldnt be loaded: {e}")
    
# ----------------------------------------------
# Page: MLflow
# ----------------------------------------------

def streamlit_mlflow():
    
    # ------------------------------------------
    # page header, tabs
    # ------------------------------------------

    st.title("MLflow Dashboard")
    st.info("KPIs of best run-through: "
    "**F1-Score** is used as criteria for the selected best model.")
    st.write(' ')

    bundle = get_best_model()
    if bundle:
        f1_score, accuracy, balanced_accuracy = get_model_metrics(bundle)
        run_id = bundle["run_id"]
        model_name = bundle["model"].__class__.__name__
        #run_id, params = model_info(bundle)

    # KPI Cards 
    col1_kpi, col2_kpi, col3_kpi = st.columns(3)

    # F1 Score → green
    with col1_kpi: kpi_card("Best F1-Score", f"{f1_score:.4f}", "#28a745") #27AE60
    # Accuracy → blue
    with col2_kpi: kpi_card("Accuracy", f"{accuracy:.4f}", "#2980B9")
    with col3_kpi: kpi_card("Balanced Accuracy", f"{balanced_accuracy:.4f}", "#2980B9")

    st.divider()
    
    # Model Info
    col1_info, col2_info = st.columns(2)

    with col1_info:
        # Run ID
        st.write(f'Run ID: {run_id}')
        st.write(f'Model: {model_name}')
    with col2_info:
        # MLflow UI Url
        mlflow_ui_url = "https://dagshub.com/ASi-DS/Road-Accidents-France-Mlops.mlflow/#/experiments"
        mlflow_link(mlflow_ui_url)
    
    #tabs
    tab1, tab2 = st.tabs(["Model Parameters", "Confusion Matrix"])

    # ------------------------------------------
    # Tab: Model Paramters
    # ------------------------------------------
    with tab1:
        st.subheader("Model Parameters")
        st.dataframe(bundle["params"])
    
    # ------------------------------------------
    # Tab: Confusion Matrix
    # ------------------------------------------
    with tab2:
        st.subheader("Confusion Matrix")
        display_confusion_matrix(run_id)
    
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
