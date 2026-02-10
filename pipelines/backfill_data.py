
#Backfill raw data in csv just for understanding data and visualization purpose only.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.data_fetcher import AQIDataFetcher
import pandas as pd

def backfill_historical_data(days: int = 365):
    print(f"🔄 Starting backfill for last {days} days...")

    # 1️⃣ Fetch data
    fetcher = AQIDataFetcher()
    df = fetcher.fetch_historical_data(days=days)

    if df is None or df.empty:
        print("❌ No data fetched. Backfill failed.")
        return None

    # 2️⃣ Save RAW data
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/aq_weather_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved: {raw_path}")
    print(f"📊 Rows: {len(df)} | Columns: {len(df.columns)}")

    # 3️⃣ Basic sanity checks
    print("\n🔍 Sanity Check:")
    print("Date range:")
    print(" From:", df["timestamp"].min())
    print(" To  :", df["timestamp"].max())

    print("\nMissing values per column:")
    print(df.isna().sum())

    return df


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    backfill_historical_data(days=365)
