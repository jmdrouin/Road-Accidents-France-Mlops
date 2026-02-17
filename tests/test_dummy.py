from src.data import make_dataset as md
import pandas as pd

def test_dummy():
    df = md.read_some_table()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] > 0