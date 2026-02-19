from sqlalchemy import create_engine
from tqdm import tqdm
from typing import List
import pandas as pd

def read_as_dataframes(
    file: str,
    tables: List[str]
) -> dict[str, pd.DataFrame]:
    engine = create_engine(f"sqlite:///{file}")
    result = {}
    for table in tables:
        chunks = pd.read_sql_table(table, con=engine, chunksize=50_000)
        dfs = []
        for chunk in tqdm(chunks, desc=f"Loading {table}"):
            dfs.append(chunk)
        result[table] = pd.concat(dfs, ignore_index=True)
    return result

def write_dataframe(table_name: str, df: pd.DataFrame, to_file: str):
    engine = create_engine(f"sqlite:///{to_file}")
    chunksize = 10_000
    total = len(df)
    with tqdm(total=total, desc=f"Writing SQL to {to_file}") as pbar:
        for i in range(0, total, chunksize):
            if_exists = "replace" if i==0 else "append"
            chunk = df.iloc[i:i+chunksize]
            chunk.to_sql(table_name, engine, if_exists=if_exists, index=False)
            pbar.update(min(chunksize, total - i))

