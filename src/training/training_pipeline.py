# ==========================================
# AQI Forecast Training
# Leakage-free, Time-series correct
# CI/CD compatible - versioning mode
# ==========================================

import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import hopsworks
import joblib
import tempfile
import json
import shutil
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# CONNECT
# -----------------------------
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)

fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_karachi_features_final", version=3)

print("=" * 70)
print("TRAINING PIPELINE - " + str(datetime.now()))
print("=" * 70)

# -----------------------------
# LOAD FEATURES
# -----------------------------
df = fg.read().sort_values("timestamp").reset_index(drop=True)
print("Loaded " + str(len(df)) + " rows from Feature Store")

# -----------------------------
# CREATE TARGETS
# -----------------------------
df["target_day1"] = df["aqi"].shift(-24)
df["target_day2"] = df["aqi"].shift(-48)
df["target_day3"] = df["aqi"].shift(-72)
df = df.dropna(subset=["target_day1", "target_day2", "target_day3"]).reset_index(drop=True)

# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

print("Train samples: " + str(len(train_df)))
print("Test samples:  " + str(len(test_df)))

# -----------------------------
# PREPARE FEATURES
# -----------------------------
DROP_COLS = ["timestamp", "aqi", "target_day1", "target_day2", "target_day3"]

X_train = train_df.drop(columns=DROP_COLS)
X_test  = test_df.drop(columns=DROP_COLS)

y_train = {
    "day1": train_df["target_day1"],
    "day2": train_df["target_day2"],
    "day3": train_df["target_day3"],
}

y_test = {
    "day1": test_df["target_day1"],
    "day2": test_df["target_day2"],
    "day3": test_df["target_day3"],
}

# Fill NaNs - train stats only
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test  = X_test.fillna(train_median)

# -----------------------------
# SCALE (Ridge only)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# MODELS 
# -----------------------------
models = {
    "Ridge": Ridge(alpha=50),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=40,
        min_samples_split=80,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=10,
        reg_lambda=10,
        random_state=42
    )
}

# -----------------------------
# TRAIN AND EVALUATE
# -----------------------------
print("\nTRAINING\n")

results = {}

for day in ["day1", "day2", "day3"]:
    print("\nTARGET: " + day.upper())
    results[day] = {}

    for name, model in models.items():
        if name == "Ridge":
            model.fit(X_train_scaled, y_train[day])
            preds_train = model.predict(X_train_scaled)
            preds_test  = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train[day])
            preds_train = model.predict(X_train)
            preds_test  = model.predict(X_test)

        rmse_train = mean_squared_error(y_train[day], preds_train, squared=False)
        rmse_test  = mean_squared_error(y_test[day], preds_test, squared=False)
        mae        = mean_absolute_error(y_test[day], preds_test)
        r2         = r2_score(y_test[day], preds_test)

        ratio = rmse_test / rmse_train
        status = "OK" if ratio <= 2.0 else "OVERFITTING"

        print(
            f"{name:<12} | RMSE: {rmse_test:6.2f} | "
            f"MAE: {mae:6.2f} | R2: {r2:5.3f} | {status}"
        )

        results[day][name] = rmse_test

# -----------------------------
# BEST MODEL PER DAY
# -----------------------------
print("\nBEST MODEL PER HORIZON\n")
for day in results:
    best = min(results[day], key=results[day].get)
    print(f"{day.upper()} -> {best} (RMSE = {results[day][best]:.2f})")

print("\nTRAINING DONE")

# ==========================================
# REGISTRY UPLOAD - VERSIONING MODE
# Hopsworks automatically creates version 1, 2, 3...
# No deletion - versions accumulate
# ==========================================
print("\n" + "=" * 70)
print("UPLOADING MODELS (VERSIONING MODE)")
print("=" * 70)

mr = project.get_model_registry()
feature_columns = X_train.columns.tolist()

# ---- STEP 1: Remove best_model tag from ALL old models ----
# Purane models pe se tag hatao taake naye best models clear ho jaye
print("\nRemoving best_model tags from previous versions...")

MODEL_NAMES = [
    "aqi_ridge_day1",
    "aqi_ridge_day2",
    "aqi_ridge_day3",
    "aqi_randomforest_day1",
    "aqi_randomforest_day2",
    "aqi_randomforest_day3",
    "aqi_xgboost_day1",
    "aqi_xgboost_day2",
    "aqi_xgboost_day3",
]

for name in MODEL_NAMES:
    try:
        # Get ALL versions of this model
        model_list = mr.get_models(name=name)
        if model_list:
            if not isinstance(model_list, list):
                model_list = [model_list]
            for m in model_list:
                try:
                    tags = m.get_tags()
                    if tags.get("best_model") == "true":
                        print("  Removing tag from: " + m.name + " v" + str(m.version))
                        m.delete_tag("best_model")
                except:
                    pass
    except:
        pass

print("  Tag cleanup done")

# ---- STEP 2: Upload new models (Hopsworks auto-versions) ----
print("\nUploading new model versions...\n")

for day in ["day1", "day2", "day3"]:
    best_model_name = min(results[day], key=results[day].get)

    for model_name, trained_model in models.items():
        X_eval = X_test_scaled if model_name == "Ridge" else X_test
        preds = trained_model.predict(X_eval)

        model_metrics = {
            "rmse": float(results[day][model_name]),
            "mae": float(mean_absolute_error(y_test[day], preds)),
            "r2": float(r2_score(y_test[day], preds)),
        }

        is_best = (model_name == best_model_name)
        if is_best:
            model_metrics["selected_for_deployment"] = 1

        # Same name every time - Hopsworks creates version 1, 2, 3...
        unique_name = f"aqi_{model_name.lower()}_{day}"

        # Plain ASCII description
        status_text = "BEST - PRODUCTION" if is_best else "Alternative"
        hours = 24 if day == "day1" else (48 if day == "day2" else 72)
        description = (
            f"AQI | {status_text} | "
            f"{day.upper()} {hours}h | {model_name} | "
            f"RMSE {model_metrics['rmse']:.2f} | "
            f"MAE {model_metrics['mae']:.2f} | "
            f"R2 {model_metrics['r2']:.3f} | "
            f"Trained {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        model_obj = mr.python.create_model(
            name=unique_name,
            description=description,
            metrics=model_metrics
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            joblib.dump(trained_model, os.path.join(tmp_dir, "model.pkl"))

            if model_name == "Ridge":
                joblib.dump(scaler, os.path.join(tmp_dir, "scaler.pkl"))

            with open(os.path.join(tmp_dir, "features.json"), "w") as f:
                json.dump({
                    "columns": feature_columns,
                    "requires_scaling": (model_name == "Ridge"),
                }, f)

            model_obj.save(tmp_dir)

            # Tags - ONLY best models get best_model tag
            try:
                model_obj.set_tag("model_type", model_name)
                model_obj.set_tag("forecast_day", day)
                if is_best:
                    model_obj.set_tag("best_model", "true")
            except:
                pass

            label = "[BEST]" if is_best else "[ALT] "
            print(f"  {label} {unique_name} (new version created)")

        finally:
            shutil.rmtree(tmp_dir)

print("\n" + "=" * 70)
print("ALL 9 MODELS UPLOADED ")
print("=" * 70)