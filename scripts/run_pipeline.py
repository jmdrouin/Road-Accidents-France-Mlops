from src.data.fetch_data import fetch_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from datetime import datetime
from dateutil.relativedelta import relativedelta

def n_years_ago(n):
    result = datetime.now() - relativedelta(years=n)
    return result.strftime("%Y-%m-%d %H:%M:%S")

def run_pipeline(nrows: int | None = None):
    print("Fetching Data")
    data_file, num_rows = fetch_data(n_years_ago(20))

    print("Building features")
    features_file = build_features()

    print("Training Model")
    model_file = train_model(
        nrows=nrows,
        info={"nrows": nrows},
    )

if __name__ == "__main__":
    run_pipeline(nrows=100000)