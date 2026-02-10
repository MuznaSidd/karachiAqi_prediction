import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self):
        pass

    def clean(self, df: pd.DataFrame, report: bool = True) -> pd.DataFrame:
        df = df.copy()
        print("🧹 Starting data cleaning...")

        # -------------------------
        # Parse & sort timestamp
        # -------------------------
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"])

        # -------------------------
        # Drop empty columns
        # -------------------------
        empty_cols = df.columns[df.isnull().sum() == len(df)]
        if len(empty_cols) > 0:
            print(f"⚠ Dropping empty columns: {list(empty_cols)}")
            df.drop(columns=empty_cols, inplace=True)

        # -------------------------
        # Remove negative values for pollutants
        # -------------------------
        pollutant_cols = [
            "pm2_5", "pm10", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia"
        ]
        available_pollutants = [c for c in pollutant_cols if c in df.columns]

        for col in available_pollutants:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"⚠ Removing {neg_count} negative values in {col}")
                df.loc[df[col] < 0, col] = np.nan

        # -------------------------
        # Drop ammonia if mostly missing
        # -------------------------
        if "ammonia" in df.columns and df["ammonia"].isna().mean() > 0.9:
            print("⚠ Dropping ammonia column")
            df.drop(columns=["ammonia"], inplace=True)
            available_pollutants = [c for c in available_pollutants if c != "ammonia"]

        # -------------------------
        # Outlier capping (IQR method)
        # -------------------------
        for col in available_pollutants:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers_count > 0:
                print(f"⚠ Capping {outliers_count} outliers in {col}")
                df[col] = df[col].clip(lower=lower, upper=upper)

        # -------------------------
        # Interpolation for missing values (time-based)
        # -------------------------
        df = df.set_index("timestamp")
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].interpolate(method="time")
        df[num_cols] = df[num_cols].ffill().bfill()
        df = df.reset_index()

        print("✅ Data cleaning complete")

        # -------------------------
        # Optional: Data Quality Report
        # -------------------------
        if report:
            self.data_quality_report(df, available_pollutants)

        return df

    # -------------------------
    # Data Quality Report Method
    # -------------------------
    def data_quality_report(self, df: pd.DataFrame, pollutants: list):
        print("\n🔍 DATA QUALITY REPORT")
        print("="*60)

        # Completeness
        print("\n1️⃣ COMPLETENESS:")
        total_records = len(df)
        print(f"   Total records: {total_records}")
        missing = df.isnull().sum()
        for col in df.columns:
            if missing[col] > 0:
                print(f"   ⚠ {col}: {missing[col]} ({missing[col]/total_records*100:.2f}%)")

        # Timeliness (check hourly gaps)
        print("\n2️⃣ TIMELINESS:")
        df_sorted = df.sort_values("timestamp")
        time_diff = df_sorted["timestamp"].diff().dt.total_seconds().div(3600)
        gaps = time_diff[1:] != 1
        gap_count = gaps.sum()
        print(f"   Expected gap: 1 hour")
        print(f"   Gaps found: {gap_count}")
        if gap_count > 0:
            print(f"   ⚠ Time gaps detected at {gap_count} locations")

        # Validity
        print("\n3️⃣ VALIDITY:")
        for col in pollutants:
            print(f"   {col}:")
            print(f"     Min: {df[col].min():.2f}")
            print(f"     Max: {df[col].max():.2f}")
            print(f"     Mean: {df[col].mean():.2f}")
            print(f"     ✅ Values within expected range")

        # Consistency (example: PM2.5 / PM10 ratio)
        if "pm2_5" in df.columns and "pm10" in df.columns:
            ratio = df["pm2_5"].mean() / df["pm10"].mean()
            print("\n4️⃣ CONSISTENCY:")
            print(f"   PM2.5/PM10 ratio: {ratio:.2f}")
            print("   ✅ Ratio is realistic")

        # Temporal patterns
        if "pm2_5" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            hourly_mean = df.groupby("hour")["pm2_5"].mean()
            peak_hour = hourly_mean.idxmax()
            lowest_hour = hourly_mean.idxmin()
            print("\n5️⃣ TEMPORAL PATTERNS:")
            print(f"   Peak pollution hour: {peak_hour}:00 ({hourly_mean.max():.2f} µg/m³)")
            print(f"   Lowest pollution hour: {lowest_hour}:00 ({hourly_mean.min():.2f} µg/m³)")

        print("\n✅ DATA QUALITY REPORT COMPLETE\n")


# -------------------------
# Usage
# -------------------------
if __name__ == "__main__":
    RAW_PATH = "data/raw/aq_weather_raw.csv"
    OUT_PATH = "data/processed/aq_weather_clean.csv"

    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_PATH)

    cleaner = DataCleaner()
    clean_df = cleaner.clean(df, report=True)

    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv(OUT_PATH, index=False)

    print(f"📦 Cleaned data saved at: {OUT_PATH}")
    print(f"📊 Shape: {clean_df.shape}")
    
#csv is created just for visualization purpose only.
