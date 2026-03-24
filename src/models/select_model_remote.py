import os
from dotenv import load_dotenv
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
env_path = root_path / '.env'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f".env not found in root")

repo_owner = os.environ['MLFLOW_TRACKING_REPO_OWNER']
user = os.environ['MLFLOW_TRACKING_USERNAME']
repo_name = os.environ['MLFLOW_TRACKING_REPO']
token = os.environ['MLFLOW_TRACKING_PASSWORD']
repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}.s3"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = repo_url
os.environ['AWS_ACCESS_KEY_ID'] = token
os.environ['AWS_SECRET_ACCESS_KEY'] = token
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

print(f"S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

import mlflow
import shutil
import joblib

import logging
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)


def select_and_store_best_model(experiment_name="Road_Accidents_France", metric="f1_macro"):
    """
    Find best run in MLflow, create bundle, store locally as .pkl
    """
    try:
        # 1. Set tracking URL
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        # 2. Load experiment, find best run
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Error: Experiment '{experiment_name}' not found.")
            return
        
        print(f"Search for best rund for experiment {experiment_name} and {metric}...")
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
        ARTIFACT_NAME = "tracked_models"
        local_path_bundle = os.path.join("models", "best_model_bundle")
        bundle_file = os.path.join(local_path_bundle, "best_model_bundle.pkl")

        # 4. Cleanup, create folder
        if os.path.exists(local_path_bundle):
            shutil.rmtree(local_path_bundle)
            print(f"Old folder removed : {local_path_bundle}")
        
        os.makedirs(local_path_bundle, exist_ok=True)
        
        # 5. Load model, create bundle
        print(f"Downloading artifacts for run {best_run_id}...")
        
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=best_run_id, 
            artifact_path=ARTIFACT_NAME
        )

        # find .pkl-file in temp folder
        pkl_files = list(Path(local_dir).glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"Keine .pkl Datei gefunden in {local_dir}")
        actual_pkl_path = pkl_files[0]

        # load model
        downloaded_bundle = joblib.load(actual_pkl_path)

        # extract
        if isinstance(downloaded_bundle, dict) and "model" in downloaded_bundle:
            model_obj = downloaded_bundle["model"]
        else:
            model_obj = downloaded_bundle
        
        # new bundle (with remote metrics from runs_df)
        # best_run from mlflow.search_runs call
        bundle = {
            "model": model_obj,
            "run_id": best_run_id,
            # use metrics from dataframe delivered by mlflow
            "metrics": {k.replace("metrics.", ""): v for k, v in best_run.items() if k.startswith("metrics.")},
            "params": {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")},
            "artifact_info": {
                "downloaded_from": ARTIFACT_NAME,
                "remote_run_id": best_run_id
            }
        }

        # store locally as best model
        joblib.dump(bundle, bundle_file) #, compress=3
        print(f"Success: Remote Model & Metrics stored under: {bundle_file}")
        
        # 6. Register model as 'best'
        try:
            # registered model name
            original_model_name = "Road_Accident_LGBM"
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

            # search for version belonging to best run id
            filter_string = f"run_id = '{best_run_id}'"
            versions = client.search_model_versions(filter_string)
            
            if versions:
                best_version = versions[0].version
                print(f"Found: Version {best_version} belongs to run id {best_run_id}.")
                
                # 2. set alias 'best' for this version
                client.set_registered_model_alias(original_model_name, "best", best_version)
                print(f"Sucsess: Model '{original_model_name}' Version {best_version} is now marked as 'best'.")
            else:
                print(f"Warning: No registered Version found for run {best_run_id}.")
        
        except Exception as e:
            print(f"Error when setting alias: {e}")
        
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
