import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import joblib
from pathlib import Path

import sys
import io
import os
import tempfile

from scripts.run_pipeline import run_pipeline

def get_latest_model_path():
    model_dir = Path("models")

    # search for all files beginnen with 'model_' and ending with '.pkl'
    model_files = list(model_dir.glob("model_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError("No model files found in folder 'models'.")
    
    # sort alphabetically/chronologically and return path
    return str(max(model_files))

def load_model(file_path: str):
    """load model bundle incl preprocessors."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No model found under path: {file_path}")
    
    # Load artifact dictionary
    artifact = joblib.load(path)

    print(f"[OK] Model loaded: {artifact['type']}")
    print(f"  Created:: {artifact['timestamps']['created']}")
    print(f"  Features: {len(artifact['feature_names'])}")

    # return artifact dictionary
    return artifact

    # ...extract and return components, eg
    # model = artifact['model']
    # encoder = artifact['onehot_encoder']
    # return model, encoder

def track_results(artifact):

    base_path = Path(__file__).resolve().parents[2] / "data" / "mlflow"
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    db_uri = f"sqlite:///{base_path.as_posix()}/mlflow.db"

    mlflow.set_tracking_uri(db_uri) # register tracking
    mlflow.set_registry_uri(db_uri) # register model
    
    mlflow.set_experiment("Road_Accidents_France")

    with mlflow.start_run(run_name=f"LGBM_{artifact['timestamps']['created']}"):

        # 1. Filter metrics: allow only numbers
        all_metrics = artifact.get("metrics", {})
        
        # create dict with scalar values (float/int)
        scalar_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar_metrics)

        # 2. Handle Confusion Matrix
        if "confusion_matrix" in all_metrics:
            
            cm = np.array(all_metrics["confusion_matrix"])

            # Store as text file, using temp file
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = os.path.join(tmp_dir, "confusion_matrix.txt")
                
                # 1. write to temp file
                np.savetxt(temp_path, cm, fmt='%d')
                
                # 2. log as artifact
                mlflow.log_artifact(temp_path)
            
            # Create plot and store as pic
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Wahre Klasse')
            plt.xlabel('Vorhergesagte Klasse')

            mlflow.log_figure(plt.gcf(), "plots/confusion_matrix.png")
            plt.close()
        
        # 3. track info
        if "info" in artifact:
            mlflow.log_params(artifact["info"])
        
        # 4. Log model as an MLflow model (LGBM)
        mlflow.lightgbm.log_model(
            #lgb_model=artifact["model"], 
            lgb_model=artifact["model"].booster_, # use booster for better compatibility (LGBM)
            artifact_path="mlflow_model",
            registered_model_name="Road_Accident_LGBM" # reigstered model name
        )
        
        # add params
        lgb_model = artifact["model"]
        lgbm_params = lgb_model.get_params()
        cleaned_params = {
            k: (str(v) if isinstance(v, (list, dict)) else v) 
            for k, v in lgbm_params.items() if v is not None
        }
        mlflow.log_params(cleaned_params)

        # 5. Load the ready .pkl file as additional artifact
        latest_model_path = get_latest_model_path()
        mlflow.log_artifact(latest_model_path, artifact_path="tracked_models")

        #mlflow.lightgbm.log_model(lgb_model=artifact["model"], artifact_path="mlflow_model")
        #mlflow.log_artifact(get_latest_model_path(), artifact_path="tracked_models")

        print(f"OK: Run finished: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":

    # 0. Run pipeline
    run_pipeline()

    # 1. Find path to newest model
    path = get_latest_model_path() 
    
    # 2. Load model bundle (artifact)
    artifact = load_model(path)

    #check: print model accuracy
    accuracy = artifact["metrics"].get("accuracy", "N/A")
    print(f"Model accuracy: {accuracy}")

    # 3. Call tracking function with loaded bundle
    print(f"START: Start MLflow tracking for: {path}")
    track_results(artifact)
    print("OK: Tracking has concluded successfully!")
