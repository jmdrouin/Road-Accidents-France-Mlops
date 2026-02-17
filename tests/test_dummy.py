from src.data import make_dataset as md
import pandas as pd

def test_read_dataframe():
    for table_name in 
    df = md.get_dataframe("caracteristics", 10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert df.shape[1] > 0