import pandas as pd
from sqlalchemy import create_engine

def read_some_table():
    engine = create_engine("sqlite:///data/sql/accidents.db")
    return pd.read_sql("SELECT * FROM caracteristics LIMIT 10", engine)