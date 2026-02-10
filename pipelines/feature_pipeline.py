# =========================================
# AQI Feature Pipeline
# API → Clean → Features → Feature Store 
# =========================================

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.data_fetcher import AQIDataFetcher
from src.features.data_cleaning import DataCleaner
from src.utils.aqi_calculator import EPAAQICalculator

load_dotenv()

FEATURE_GROUP_NAME = "aqi_karachi_features_final"
FEATURE_GROUP_VERSION = 3
POLLUTANTS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide"]

# =========================
# FEATURE ENGINEER CLASS
# =========================
class FeatureEngineer:
    def __init__(self):
        self.calc = EPAAQICalculator()

    def time_features(self, df):
        
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_night"] = df["hour"].isin([0,1,2,3,4,5,22,23]).astype(int)
        return df

    def weather_features(self, df):
        
        df['is_hot'] = (df['temperature_2m'] > 35).astype(int)
        df['is_cold'] = (df['temperature_2m'] < 20).astype(int)
        df['low_wind'] = (df['wind_speed_10m'] < 5).astype(int)
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m'] / 100
        return df

    def lag_features(self, df):
        cols = POLLUTANTS + ['aqi']
        for col in cols:
            for lag in [1, 3, 6, 24, 72]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def rolling_features(self, df):
        cols = POLLUTANTS + ['aqi']
        for col in cols:
            for w in [3, 6, 24, 72]:
                df[f"{col}_roll_mean_{w}"] = df[col].shift(1).rolling(w).mean()
                df[f"{col}_roll_std_{w}"] = df[col].shift(1).rolling(w).std()
        return df

    def create_aqi(self, df):
        df['aqi'] = df.apply(lambda r: self.calc.calculate_aqi(
            pm25=r.get('pm2_5', 0),
            pm10=r.get('pm10', 0),
            o3=r.get('ozone', 0),
            no2=r.get('nitrogen_dioxide', 0)
        )[0], axis=1)
        return df

    def fill_missing(self, df):
        for col in df.columns:
            if 'roll_std' in col:
                df[col] = df[col].fillna(0)
            elif 'lag' in col or 'roll_mean' in col:
                df[col] = df[col].ffill()
        return df

    def run(self, df):
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = self.create_aqi(df)
        df = self.time_features(df)
        df = self.weather_features(df)
        df = self.lag_features(df)
        df = self.rolling_features(df)
        df = self.fill_missing(df)
        df.dropna(inplace=True)
        return df.reset_index(drop=True)

# =========================
# MAIN PIPELINE - INCREMENTAL
# =========================
def run_feature_pipeline(use_api=True, days_to_fetch=7):
    #At first run, it was set to 365, then after first run it was set to 7 just for incremental runs
    """
    use_api: True = fetch from API
    days_to_fetch: how many days to fetch (for daily update)
    """
    print("="*70)
    print(f"🚀 Feature Pipeline - {datetime.now()}")
    print("="*70)
    
    # ✅ Fetch from API (for automation)
    if use_api:
        print(f"\n📥 Fetching {days_to_fetch} days from API...")
        fetcher = AQIDataFetcher()
        df_raw = fetcher.fetch_historical_data(days=days_to_fetch)
        
        if df_raw is None or df_raw.empty:
            raise ValueError("❌ API data unavailable!")
        
        print(f"   ✅ Fetched: {len(df_raw)} rows")
        
        # Clean
        print("\n🧹 Cleaning data...")
        cleaner = DataCleaner()
        df = cleaner.clean(df_raw, report=False)
        print(f"   ✅ Clean: {len(df)} rows")
    
    else:
        df = pd.read_csv("data/processed/aq_weather_clean.csv", parse_dates=['timestamp'])
        print(f"   ✅ Loaded: {len(df)} rows")
    
    # ✅ Engineer features
    print("\n⚙️  Engineering features...")
    fe = FeatureEngineer()
    df_feat = fe.run(df)
    
    # ✅ Add target columns 
    df_feat['target_day1'] = np.nan
    df_feat['target_day2'] = np.nan
    df_feat['target_day3'] = np.nan
    
    # ✅ Convert all numeric columns to proper type (Hopsworks compatibility)
    for col in df_feat.select_dtypes(include=[np.float64, np.float32]).columns:
        df_feat[col] = df_feat[col].astype(np.float64)  
    
    print(f"   ✅ Features: {df_feat.shape[1]} columns, {len(df_feat)} rows")
    
    # ✅ Connect to Hopsworks
    print("\n📤 Connecting to Hopsworks...")
    import hopsworks
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs = project.get_feature_store()
     
    try:
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        print(f"   ✅ Found Feature Group v{FEATURE_GROUP_VERSION}")
    except:
        print(f"   ⚠️  Feature Group v{FEATURE_GROUP_VERSION} not found, creating...")
        fg = fs.create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            description="Merged leakage_safe AQI features with weather & historical patterns",
            online_enabled=False
        )
        print(f"   ✅ Created Feature Group v{FEATURE_GROUP_VERSION}")
    
    # CHECK latest timestamp in Feature Store
    print("\n🔍 Checking for duplicates...")
    try:
        existing_df = fg.read()  
        latest_fs_timestamp = pd.to_datetime(existing_df["timestamp"].max()).tz_localize(None)
        print(f"   📅 Feature Store latest: {latest_fs_timestamp}")
        
        # Filter ONLY new data
        df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
        # Proper timezone check
        if hasattr(df_feat['timestamp'].dtype, 'tz') and df_feat['timestamp'].dtype.tz is not None:
            df_feat['timestamp'] = df_feat['timestamp'].dt.tz_localize(None)
        
        new_data = df_feat[df_feat['timestamp'] > latest_fs_timestamp].copy()
        
        if len(new_data) > 0:
            print(f"   ✨ NEW rows to insert: {len(new_data)}")
            print(f"   Date range: {new_data['timestamp'].min()} → {new_data['timestamp'].max()}")
            
            # Insert ONLY new data
            fg.insert(new_data, write_options={"wait_for_job": True})
            print(f"   ✅ Inserted {len(new_data)} rows")
        else:
            print(f"   ℹ️  No new data (Feature Store up-to-date)")
    
    except Exception as e:
        # First time insert (Feature Group empty)
        print(f"   ⚠️  Feature Store empty or error: {str(e)}")
        print(f"   ✨ Inserting ALL {len(df_feat)} rows...")
        fg.insert(df_feat, write_options={"wait_for_job": True})
        print(f"   ✅ Inserted {len(df_feat)} rows")
    
    print("\n" + "="*70)
    print("✅ FEATURE PIPELINE COMPLETE")
    print("="*70)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,
                       help="Days to fetch from API (default: 7)")
    
    args = parser.parse_args()
    
    # Simple: Just run with API
    run_feature_pipeline(use_api=True, days_to_fetch=args.days)