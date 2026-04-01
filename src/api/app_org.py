# uvicorn src.api.app:app --reload
# $env:MODEL_DIR = ".\models"; uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.models.accident import build_accident_model

import matplotlib
matplotlib.use('Agg')  # force non-interactive backend

import subprocess
import sys

Accident = build_accident_model()

app = FastAPI()

class DatasetRequest(BaseModel):
    cutoff_date: str

@app.get("/health")
def health():
    return {"status": "ok"}

# curl -X POST http://127.0.0.1:8000/update_data -H "Content-Type: application/json"
@app.post("/update_data")
def update_data():
    from src.data.fetch_data import fetch_data
    def n_years_ago(n):
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        result = datetime.now() - relativedelta(years=n)
        return result.strftime("%Y-%m-%d %H:%M:%S")

    # Fetch data until 20 years ago (simulated)
    # TODO: let the caller of the api pick the cutoff?
    file_written, num_rows = fetch_data(n_years_ago(20))

    return {
        "status": "ok",
        "file_written": file_written,
        "num_rows": num_rows
    }

# curl -X POST http://127.0.0.1:8000/build_features -H "Content-Type: application/json"
@app.post("/build_features")
def build_features_from_latest_data():
    from src.features.build_features import build_features
    file_written = build_features()
    return {
        "status": "ok",
        "file_written": file_written,
    }

class TrainRequest(BaseModel):
    max_rows: Optional[int] = None

# curl -X POST http://127.0.0.1:8000/train_model -H "Content-Type: application/json"
# OR with a subset of the data: ... -d '{"max_rows": 50000}'
@app.post("/train_model")
def train_new_model(request: TrainRequest):
    from src.models.train_model import train_model
    file_written = train_model(nrows=request.max_rows, info={"nrows": request.max_rows})
    return {
        "status": "ok",
        "file_written": file_written,
    }

PREDICT_COLUMNS = [
    "timestamp", "collision_label", "is_weekend", "season",
    "surface_condition_label", "manoeuvre_label", "sex_label",
    "user_category_label", "seat_position_label", "journey_purpose_label",
    "is_holiday", "age", "age_group", "seatbelt_used", "helmet_used",
    "any_protection_used", "protection_effective", "vehicle_group",
    "impact_group", "motorcycle_side_impact", "is_night", "is_urban",
    "lane_width", "road_group", "weather_group", "day_of_week",
    "hour_group"
]

@app.post("/predict")
def predict_accident(accident: Accident):
    from src.util.files import last_file_in_folder
    from src.models.predict_model import predict_dataframe, load_artifact
    import pandas as pd
    try:
        #model_file = last_file_in_folder("models/best_model_bundle", "*.pkl")
        model_file = last_file_in_folder("models", "model_*.pkl") #todo: best, workaround
        if not model_file:
            raise HTTPException(status_code=404, detail="No trained model found")

        artifact = load_artifact(model_file)

        row = accident.model_dump()
        row["timestamp"] = None

        df = pd.DataFrame([row])[PREDICT_COLUMNS]
        result = predict_dataframe(df, artifact).copy()

        # convert numpy / arrow / pandas values into plain JSON-safe Python values
        result_row = result.iloc[0].to_dict()

        probas = {
            k.replace("proba_", ""): float(v)
            for k, v in result_row.items()
            if k.startswith("proba_")
        }

        if not probas:
            raise HTTPException(status_code=500, detail="Prediction returned no probabilities")

        best_label = max(probas, key=probas.get)
        best_proba = probas[best_label]

        return {
            "status": "ok",
            "model_file": model_file,
            "prediction": best_label,
            "confidence": float(best_proba),
            "probabilities": probas,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/run_pipeline")
def run_pipeline(request: TrainRequest):
    from src.data.fetch_data import fetch_data
    from src.features.build_features import build_features
    from src.models.train_model import train_model
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    def n_years_ago(n):
        result = datetime.now() - relativedelta(years=n)
        return result.strftime("%Y-%m-%d %H:%M:%S")

    data_file, num_rows = fetch_data(n_years_ago(20))
    features_file = build_features()
    model_file = train_model(
        nrows=request.max_rows,
        info={"nrows": request.max_rows},
    )

    return {
        "status": "ok",
        "data_file": data_file,
        "num_rows": num_rows,
        "features_file": features_file,
        "model_file": model_file,
    }

# import track_model_remote
from src.models.track_model_remote import (
     run_pipeline, 
     get_latest_model_path, 
     load_model, 
     track_results, 
     select_and_store_best_model
)

# curl -X POST http://127.0.0.1:8001/run_tracking_remote -H "Content-Type: application/json"
# Invoke-RestMethod -Method Post -Uri "http://localhost:8001/run_tracking_remote"
@app.post("/run_tracking_remote")
def run_tracking_remote():
    try:
        # 0. run pre-pipeline
        run_pipeline()

        # 1. find path
        path = get_latest_model_path()
        if not path:
            raise HTTPException(status_code=404, detail="No model found")

        # 2. load bundle & check metrics
        artifact = load_model(path)
        accuracy = artifact["metrics"].get("accuracy", "N/A")

        # 3. start tracking
        track_results(artifact)

        # 4. choose best model
        select_and_store_best_model(
            experiment_name="Road_Accidents_France", 
            metric="f1_macro"
        )

        return {
            "status": "success",
            "model_path": path,
            "accuracy": accuracy,
            "message": "Tracking successfully concluded"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    