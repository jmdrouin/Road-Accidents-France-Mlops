# uvicorn src.api.app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from src.data.fetch_data import fetch_data

app = FastAPI()

class DatasetRequest(BaseModel):
    cutoff_date: str

@app.get("/health")
def health():
    return {"status": "ok"}

# curl -X POST http://127.0.0.1:8000/update_data -H "Content-Type: application/json"
@app.post("/update_data")
def build_dataset():

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