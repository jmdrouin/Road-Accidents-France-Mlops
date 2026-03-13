# uvicorn src.api.app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

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