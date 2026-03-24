from src.data.fetch_data import fetch_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.util.time_machine import simulated_time

import sys, os
is_background = not sys.stdout.isatty()
if is_background:
    os.environ["TQDM_DISABLE"] = "1"

def run_pipeline(nrows: int | float | None = None):
    print("Fetching Data")
    data_file, num_rows = fetch_data(simulated_time())

    print("Building features")
    features_file = build_features()

    print("Training Model")
    model_file = train_model(
        nrows=nrows,
        info={"nrows": nrows},
    )

if __name__ == "__main__":
    from src.util.config import CONFIG
    run_pipeline(nrows=CONFIG["data"]["nrows"])