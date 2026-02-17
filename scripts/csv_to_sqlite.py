from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

def pretty_tracebacks():
    # Pretty tracebacks:
    import os, re, sys, traceback
    from pathlib import Path
    home_raw = os.path.expanduser("~")
    home_resolved = str(Path(home_raw).resolve())
    user_pat = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    def redact(text: str) -> str:
        for h in {home_raw, home_resolved}:
            if h and h != "/":
                text = text.replace(h, "~")
        if user_pat:
            text = re.sub(rf"(/Users/){user_pat}\b", r"\1~", text)
            text = re.sub(rf"(/home/){user_pat}\b", r"\1~", text)
        return text
    def redacting_excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(redact(text))
    sys.excepthook = redacting_excepthook
pretty_tracebacks()


from pathlib import Path
import os

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
        print(f"Loading {csv_file.name} → table {table_name}")

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