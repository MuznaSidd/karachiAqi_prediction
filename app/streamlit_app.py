# ========================================
#  Karachi AQI Predictor 
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from hopsworks import login
from dotenv import load_dotenv
from datetime import datetime, timedelta

# ---------------------- ENV ----------------------
load_dotenv()
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
st.sidebar.title("1🌍 Pearls AQI Predictor")
page = st.sidebar.radio("Navigation", ["Forecast Dashboard", "EDA Dashboard"])

# ---------------------- AQI color & hazard ----------------------
def aqi_color_hazard(aqi):
    if aqi < 50:
        return "green", "Good"
    elif aqi < 100:
        return "yellow", "Moderate"
    elif aqi < 150:
        return "orange", "Unhealthy for Sensitive Groups"
    elif aqi < 200:
        return "red", "Unhealthy"
    elif aqi < 300:
        return "purple", "Very Unhealthy"
    else:
        return "maroon", "Hazardous"

# ---------------------- Connect Hopsworks ----------------------
@st.cache_resource
def connect_hopsworks():
    project = login(api_key_value=HOPSWORKS_API_KEY, host=HOPSWORKS_HOST)
    fs = project.get_feature_store()
    return project, fs

project, fs = connect_hopsworks()

# ============================================================== 
# ======================= FORECAST DASHBOARD =================== 
# ============================================================== 
if page == "Forecast Dashboard":

    FEATURE_GROUP_NAME = "aqi_karachi_features_final"
    FEATURE_GROUP_VERSION = 3
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    st.sidebar.subheader("Forecast Settings")
    day_choice = st.sidebar.slider("Select Forecast Day", 1, 3, value=1)

    today = datetime.today()
    st.sidebar.markdown(
        f"📅 Forecast reference date: {today.strftime('%A, %d %b %Y')}"
    )

    st.title("🌫️ Karachi AQI – 3 Day Forecast")
    st.info("Fetching latest AQI data from Hopsworks Feature Store...")

    try:
        df_recent = fg.read().sort_values("timestamp").tail(3).reset_index(drop=True)
        if len(df_recent) < 3:
            st.warning("Less than 3 rows available. Using latest row only.")
            X_latest = df_recent.tail(1)
        else:
            if day_choice == 1:
                X_latest = df_recent.iloc[-1:]
            elif day_choice == 2:
                X_latest = df_recent.iloc[-2:-1]
            else:
                X_latest = df_recent.iloc[-3:-2]
    except Exception as e:
        st.warning(f"Feature Store unavailable: {e}. Using safe dummy data.")
        cols = [
            "pm2_5","pm10","ozone",
            "carbon_monoxide","nitrogen_dioxide","sulphur_dioxide"
        ]
        X_latest = pd.DataFrame([{c: 0 for c in cols}])

    X_latest = X_latest.fillna(0)

    # ---------------------- Load model (FIXED - Direct name approach) ----------------------
    st.info("Loading best production model from Model Registry...")

    mr = project.get_model_registry()
    
    try:
        day_tag = f"day{day_choice}"
        
        #  Direct model name approach 
        model_names_to_try = [
            f"aqi_ridge_{day_tag}",
            f"aqi_randomforest_{day_tag}",
            f"aqi_xgboost_{day_tag}",
            # Fallback to timestamp versions
            f"aqi_ridge_{day_tag}",
            f"aqi_randomforest_{day_tag}",
            f"aqi_xgboost_{day_tag}",
        ]
        
        day_models = []
        
        for model_name in model_names_to_try:
            try:
                model = mr.get_model(model_name, version=None)
                day_models.append(model)
            except:
                continue
        
        if not day_models:
            st.error(f"❌ No models found for day {day_choice}.")
            st.warning("Expected model names like: aqi_ridge_day1_v2, aqi_randomforest_day1_v2")
            st.info("Please check Model Registry for exact model names.")
            st.stop()
        
        #  Safe metrics access with fallback
        def get_rmse(model):
            try:
                if hasattr(model, 'training_metrics') and model.training_metrics:
                    return model.training_metrics.get("rmse", float('inf'))
                return float('inf')
            except:
                return float('inf')
        
        model_obj = min(day_models, key=get_rmse)
        
        # Get metrics safely
        try:
            metrics = model_obj.training_metrics if model_obj.training_metrics else {}
            rmse = metrics.get("rmse", "N/A")
            mae = metrics.get("mae", "N/A")
            r2 = metrics.get("r2", "N/A")
        except:
            rmse = mae = r2 = "N/A"

        st.success(f"✅ **{model_obj.name}**")
        st.info(f"📊 RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")

        # Load model & scaler
        model_dir = model_obj.download()
        model = joblib.load(f"{model_dir}/model.pkl")

        scaler = None
        scaler_path = f"{model_dir}/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

        features = json.load(open(f"{model_dir}/features.json"))["columns"]

        for f in features:
            if f not in X_latest.columns:
                X_latest[f] = 0

        X_input = X_latest[features]
        if scaler:
            X_input = scaler.transform(X_input)

        # ---------------------- Prediction ----------------------
        pred = float(np.clip(model.predict(X_input)[0], 0, 500))
        color, hazard = aqi_color_hazard(pred)

        # ✅AQI NUMBER + Health Impact
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                label=f"🌫️ Predicted AQI — Day {day_choice}",
                value=f"{pred:.1f}",
                delta=hazard,
                delta_color="off"
            )
        
        with col2:
            st.subheader("🚦 Health Impact (Alerts)")
            st.info(f"**{hazard}**")

        # ✅ TABULAR DATA: Pollutants + Weather
        st.subheader("📊 Current Measurements ")
        
        # Prepare data for table
        pollutants = ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]
        weather = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]
        
        data_dict = {"Parameter": [], "Value": [], "Unit": []}
        
        
        # Pollutants
        for p in pollutants:
            if p in X_latest.columns:
                data_dict["Parameter"].append(p.replace("_", " ").title())
                data_dict["Value"].append(f"{X_latest[p].values[0]:.2f}")
                data_dict["Unit"].append("μg/m³")
        
        # Weather
        for w in weather:
            if w in X_latest.columns:
                unit = "°C" if "temperature" in w else "%" if "humidity" in w else "m/s" if "wind" in w else "hPa"
                data_dict["Parameter"].append(w.replace("_", " ").title())
                data_dict["Value"].append(f"{X_latest[w].values[0]:.2f}")
                data_dict["Unit"].append(unit)
        
        # Display table
        df_measurements = pd.DataFrame(data_dict)
        st.dataframe(df_measurements, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)

# ============================================================== 
# =========================== EDA ============================== 
# ============================================================== 
else:
    FEATURE_GROUP_NAME = "aqi_karachi_features_final"
    FEATURE_GROUP_VERSION = 3
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.info("Loading data from Feature Store...")

    try:
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        st.warning("Using dummy data for EDA.")
        cols = [
            "pm2_5","pm10","ozone","carbon_monoxide",
            "nitrogen_dioxide","sulphur_dioxide",
            "temperature","humidity","windspeed","aqi"
        ]
        df = pd.DataFrame([{c: 0 for c in cols} for _ in range(24)])
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=24, freq="H")

    # AQI trend
    st.subheader("📈 AQI Trend")
    if "aqi" in df.columns:
        st.line_chart(df.set_index("timestamp")["aqi"])

    # Correlation
    st.subheader("🔥 Pollutants + Weather Correlation")
    numeric = df.select_dtypes(include=[np.number])
    st.write(numeric.corr().style.background_gradient(cmap="coolwarm"))

    # Distributions
    st.subheader("📊 Feature Distributions")
    features = [
        "pm2_5","pm10","ozone",
        "carbon_monoxide","nitrogen_dioxide",
        "temperature","humidity","windspeed"
    ]
    for f in features:
        if f in df.columns:
            st.write(f)
            st.bar_chart(df[f])

    st.caption("Developed by Muzna Siddiqui | 10Pearls | AQI Predictor (Hopsworks Integrated)")