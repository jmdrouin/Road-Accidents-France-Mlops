from sqlalchemy import create_engine, text
from tqdm import tqdm
from typing import List
import pandas as pd
import math

def read_as_dataframes(
    file: str,
    tables: List[str]
) -> dict[str, pd.DataFrame]:
    engine = create_engine(f"sqlite:///{file}")
    result = {}
    chunksize = 10_000
    with engine.connect() as conn:
        for table in tables:
            n_rows = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()
            n_chunks = math.ceil(n_rows / chunksize)
            chunks = pd.read_sql_table(table, con=conn, chunksize=chunksize)
            dfs = []
            for chunk in tqdm(chunks, total=n_chunks, desc=f"Loading {table}", unit="chunk"):
                dfs.append(chunk)
            result[table] = pd.concat(dfs, ignore_index=True)
    return result

def read_accidents(
    nrows: int | None,
    cutoff_date: str | None,
) -> pd.DataFrame:
    file = "data/processed/accidents.db"
    engine = create_engine(f"sqlite:///{file}")
    table = "accidents"
    chunksize = 10_000
    params = {}

    where_clause = ""
    if cutoff_date is not None:
        where_clause = " WHERE date <= :cutoff_date"
        params["cutoff_date"] = cutoff_date

    limit_clause = ""
    if not nrows is None:
        limit_clause = " ORDER BY RANDOM() LIMIT :nrows"
        params["nrows"] = nrows

    count_query = text(f"SELECT COUNT(*) FROM (SELECT 1 FROM {table}{where_clause}{limit_clause})")
    data_query = text(f"SELECT * FROM {table}{where_clause}{limit_clause}")

    with engine.connect() as conn:
        n_rows = conn.execute(count_query, params).scalar()
        n_chunks = math.ceil(n_rows / chunksize)
        chunks = pd.read_sql(data_query, conn, params=params, chunksize=chunksize)
        dfs = []
        for chunk in tqdm(chunks, total=n_chunks, desc=f"Loading {table}", unit="chunk"):
            dfs.append(chunk)
        return pd.concat(dfs, ignore_index=True)    

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

def write_dataframes( to_file: str, tables: dict[str, pd.DataFrame]):
    print(f"Writing SQL to {to_file}")

    engine = create_engine(f"sqlite:///{to_file}")
    chunksize = 10_000

    for table_name, df in tables.items():
        total = len(df)
        with tqdm(total=total, desc=f"Writing {table_name}") as pbar:
            for i in range(0, total, chunksize):
                if_exists = "replace" if i==0 else "append"
                chunk = df.iloc[i:i+chunksize]
                chunk.to_sql(table_name, engine, if_exists=if_exists, index=False)
                pbar.update(min(chunksize, total - i))