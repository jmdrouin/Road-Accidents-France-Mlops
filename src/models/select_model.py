import mlflow
import os
import shutil
import joblib
from pathlib import Path

import logging
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

def select_and_store_best_model(experiment_name="Road_Accidents_France", metric="f1_macro"):
    """
    Find best run in MLflow, create bundle, store locally as .pkl
    """
    # 1. Get tracking URI of MLflow SQLite DB
    base_path = Path(__file__).resolve().parents[2] / "data" / "mlflow"
    db_uri = f"sqlite:///{base_path.as_posix()}/mlflow.db"
    mlflow.set_tracking_uri(db_uri)
    
    # 2. Load experiment, find best run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return
    
    print(f"Search for best rund for experiment {experiment_name} and {metric}...")
    # todo: different metrics may need DESC / ASC; eg, for f1_macro it is desc
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if runs_df.empty:
        print("No runs found.")
        return
    
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']
    current_f1 = best_run[f'metrics.{metric}']
    print(f"Best Run ID: {best_run_id} ({metric}: {current_f1:.4f})")

    # 3. Define paths
    ARTIFACT_NAME = "mlflow_model" 
    local_path_bundle = os.path.join("models", "best_model_bundle")
    bundle_file = os.path.join(local_path_bundle, "best_model_bundle.pkl")

    # 4. Cleanup, create folder
    if os.path.exists(local_path_bundle):
        shutil.rmtree(local_path_bundle)
        print(f"Old folder removed : {local_path_bundle}")
    
    os.makedirs(local_path_bundle, exist_ok=True)

    # 5. Load model, create bundle
    model_uri = f"runs:/{best_run_id}/{ARTIFACT_NAME}"

    try:
        best_model_obj = mlflow.pyfunc.load_model(model_uri)
        
        bundle = {
            "model": best_model_obj,
            "run_id": best_run_id,
            "metrics": {k.replace("metrics.", ""): v for k, v in best_run.items() if k.startswith("metrics.")},
            "params": {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
        }

        # 6. Store with joblib (as .pkl)
        joblib.dump(bundle, bundle_file, compress=3)
        print(f"Success: Bundle stored under: {bundle_file}")

        # 7. Test: load bundle
        test_bundle = joblib.load(bundle_file)
        f1_score = test_bundle["metrics"]["f1_macro"]

        print(f"Test-Check: Model successfully loaded. Run: {test_bundle['run_id']} .")
        print(f"F1-Score: {f1_score:.4f}")
        
    except Exception as e:
        print(f"Error when storing/loading model: {e}")

if __name__ == "__main__":
    select_and_store_best_model(
        experiment_name="Road_Accidents_France", 
        metric="f1_macro"
    )
