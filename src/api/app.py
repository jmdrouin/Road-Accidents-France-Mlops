# uvicorn src.api.app:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.models.accident import build_accident_model
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
    from src.util import last_file_in_folder
    from src.models.predict_model import predict_dataframe, load_artifact
    import pandas as pd
    try:
        model_file = last_file_in_folder("models/best_model_bundle", "*.pkl")
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