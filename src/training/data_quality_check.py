#To understand the raw data
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_data_quality():
    print("🔍 DATA QUALITY REPORT\n" + "="*60)
    
    df = pd.read_csv('data/raw/aq_weather_raw.csv', parse_dates=['timestamp'])
    
    # 1. Completeness
    print("\n1️⃣ COMPLETENESS:")
    total = len(df)
    missing = df.isnull().sum()
    print(f"   Total records: {total}")
    if missing.sum() > 0:
        print(f"   Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"     {col}: {count} ({count/total*100:.1f}%)")
    else:
        print("   ✅ No missing values!")
    
    # 2. Timeliness
    print("\n2️⃣ TIMELINESS:")
    time_gaps = df['timestamp'].diff()
    expected_gap = pd.Timedelta(hours=1)
    gaps = time_gaps[time_gaps != expected_gap]
    print(f"   Expected gap: 1 hour")
    print(f"   Gaps found: {len(gaps)}")
    if len(gaps) > 0:
        print(f"   ⚠️ Time gaps detected at {len(gaps)} locations")
    else:
        print("   ✅ Perfect hourly sequence!")
    
    # 3. Validity
    print("\n3️⃣ VALIDITY:")
    for col in ['pm2_5', 'pm10', 'temperature_2m', 'relative_humidity_2m']:
        if col in df.columns:
            print(f"   {col}:")
            print(f"     Min: {df[col].min():.2f}")
            print(f"     Max: {df[col].max():.2f}")
            print(f"     Mean: {df[col].mean():.2f}")
            
            # Check for impossible values
            if col == 'pm2_5' and (df[col].min() < 0 or df[col].max() > 1000):
                print(f"     ⚠️ Suspicious values detected!")
            elif col == 'temperature_2m' and (df[col].min() < -50 or df[col].max() > 60):
                print(f"     ⚠️ Suspicious values detected!")
            elif col == 'relative_humidity_2m' and (df[col].min() < 0 or df[col].max() > 100):
                print(f"     ⚠️ Suspicious values detected!")
            else:
                print(f"     ✅ Values within expected range")
    
    # 4. Consistency
    print("\n4️⃣ CONSISTENCY:")
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        ratio = (df['pm2_5'] / df['pm10']).mean()
        print(f"   PM2.5/PM10 ratio: {ratio:.2f}")
        if 0.3 < ratio < 0.8:
            print(f"   ✅ Ratio is realistic")
        else:
            print(f"   ⚠️ Unusual ratio!")
    
    # 5. Temporal patterns
    print("\n5️⃣ TEMPORAL PATTERNS:")
    if 'pm2_5' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby('hour')['pm2_5'].mean()
        peak_hour = hourly.idxmax()
        low_hour = hourly.idxmin()
        print(f"   Peak pollution hour: {peak_hour}:00 ({hourly[peak_hour]:.1f} µg/m³)")
        print(f"   Lowest pollution hour: {low_hour}:00 ({hourly[low_hour]:.1f} µg/m³)")
        
        if 6 <= peak_hour <= 10 or 17 <= peak_hour <= 20:
            print(f"   ✅ Peak aligns with rush hours (realistic!)")
        else:
            print(f"   ⚠️ Unusual peak timing")
    
    print("\n" + "="*60)
    print("✅ DATA QUALITY CHECK COMPLETE!")

if __name__ == "__main__":
    check_data_quality()