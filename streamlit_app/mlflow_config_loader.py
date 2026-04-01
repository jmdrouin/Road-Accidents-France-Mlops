import os
from dotenv import load_dotenv
from pathlib import Path
import mlflow

def setup_mlflow(parent_level):

    # path to .env file
    root_path = Path(__file__).resolve().parents[parent_level] #eg 1
    env_path = root_path / '.env'

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        raise FileNotFoundError(f".env nicht in {root_path} gefunden")

    # environment variables
    repo_owner = os.environ['MLFLOW_TRACKING_USERNAME']
    repo_name = os.environ['MLFLOW_TRACKING_REPO']
    token = os.environ['MLFLOW_TRACKING_PASSWORD']
    
    # DagsHub url
    repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}.s3"
    tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = repo_url
    os.environ['AWS_ACCESS_KEY_ID'] = token
    os.environ['AWS_SECRET_ACCESS_KEY'] = token
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    
    # MLflow tracking url
    mlflow.set_tracking_uri(tracking_uri)

    #print(f"S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
