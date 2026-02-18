import pandas as pd

def cleanup_holidays(holidays: pd.DataFrame):
    # Step 1: rename columns
    holidays = holidays.rename(columns={"ds": "date", "holiday": "holiday_name"})

    # Step 2: convert to datetime
    holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce")

    # Step 3: add holiday flag
    holidays["is_holiday"] = 1

    # Step 4: drop duplicates
    holidays = holidays.drop_duplicates(subset="date")

    # Step 5: clean holiday_name
    holidays["holiday_name"] = holidays["holiday_name"].str.strip().fillna("None")

    return holidays
