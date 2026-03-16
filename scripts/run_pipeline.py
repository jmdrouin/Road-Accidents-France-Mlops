from src.data.fetch_data import fetch_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from datetime import datetime
from dateutil.relativedelta import relativedelta

nrows = 100000

def n_years_ago(n):
    result = datetime.now() - relativedelta(years=n)
    return result.strftime("%Y-%m-%d %H:%M:%S")

data_file, num_rows = fetch_data(n_years_ago(20))
features_file = build_features()
model_file = train_model(
    nrows=nrows,
    info={"nrows": nrows},
)
