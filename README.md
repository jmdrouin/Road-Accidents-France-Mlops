# Dev notes

## Development setup (uv)

### Activate:
source .venv/bin/activate   # macOS/Linux
uv sync

### Add package:
uv add <package>

### Add package for a single group (e.g. `dev`)
uv add --group dev <package>

## Run unit tests
uv run pytest

## (for now) Download and convert data to sql:
uv run scripts/download_data.py
uv run scripts/csv_to_sqlite.py

==============================
## Docker
==============================

# Start docker for both the API and the streamlit:
docker compose up --build

# Prerequisites:
- Make sure that docker is running
- Make sure that you have at least one trained model (run python -m scripts.run_pipeline to make one)

# URLs:
Streamlit -> http://localhost:8501
API docs -> http://localhost:8000/docs

==============================
## MLflow
==============================

## Run model tracking script:

uv sync
$env:PYTHONPATH = "."; uv run python -m src.models.track_model


## Stop server on port 5000:

$p = (Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue).OwningProcess
if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }


## Run server in background:

Start-Job -Name "mlflow" -ScriptBlock {
    Set-Location "[local path to Road-Accidents-France-Mlops]"
    uv run mlflow ui --backend-store-uri sqlite:///[local path to Road-Accidents-France-Mlops]/data/mlflow/mlflow.db --host 127.0.0.1 --port 5000
}

==============================
## Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>