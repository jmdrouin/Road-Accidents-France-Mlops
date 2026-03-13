import shutil
import sqlite3
import glob
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def fetch_data(cutoff: str):
    source = "data/raw/accidents.db"
    dest_folder = "data/raw/latest"

    # clean old db files
    for f in glob.glob(os.path.join(dest_folder, "*.db")):
        os.remove(f)

    destination = dest_folder + "/accidents_" + pd.Timestamp(cutoff).strftime("%Y%m%d%H%M%S") + ".db"

    shutil.copy(source, destination)

    con = sqlite3.connect(destination)

    orig_count = con.execute("SELECT COUNT(*) FROM caract").fetchone()[0]

    con.execute("DELETE FROM caract WHERE timestamp >= ?", (cutoff,))
    con.commit()

    filtered_count = con.execute("SELECT COUNT(*) FROM caract").fetchone()[0]
    print("Original rows:", orig_count)
    print("Filtered rows:", filtered_count)

    con.close()

# Fetch data until 20 years ago (simulated)
ten_years_ago = datetime.now() - relativedelta(years=20)
fetch_data(ten_years_ago.strftime("%Y-%m-%d %H:%M:%S"))