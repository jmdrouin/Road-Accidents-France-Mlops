from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import sqlite3

def load_folder_to_sqlite(
        input_folder: Path,
        db_path: Path,
        delimiter: str,
        encoding: str
) -> None:
    print("Convert csv to sqlite.")
    print("--from--", input_folder, "--to--", db_path)

    csv_files = sorted(input_folder.glob("*.csv"))
    print("files:")
    for f in csv_files: print("-", f)

    print("Creating engine.")
    engine = create_engine(f"sqlite:///{db_path}")

    for csv_file in csv_files:
        table_name = csv_file.stem.lower()
        print(f"Loading {csv_file.name} -> table {table_name}")

        print(csv_file)
        df = pd.read_csv(csv_file, encoding=encoding, sep=delimiter, low_memory=False)

        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            chunksize=10_000,
        )

    print(f"Database written to: {db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="data/csv")
    parser.add_argument("--output", type=Path, default="data/sql/accidents.db")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--encoding", default="latin-1")
    args = parser.parse_args()

    load_folder_to_sqlite(args.input, args.output, args.delimiter, args.encoding)


    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("data/sql/accidents.db")

    df = pd.read_sql_query("""
    SELECT strftime('%Y-%m', timestamp) AS month, COUNT(*) as n
    FROM accidents
    WHERE timestamp IS NOT NULL
    GROUP BY month
    ORDER BY month
    """, conn)

    print(df.head())
    print(df.tail())