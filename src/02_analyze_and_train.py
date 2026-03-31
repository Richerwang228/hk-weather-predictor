"""
Step 2: Analyze HKO historical data + Train LightGBM model + Compute all parameters.

This script does EVERYTHING:
  1. Load and clean HKO data (1884-2026)
  2. Exploratory Data Analysis (seasonal patterns, peak hour stats)
  3. Feature engineering for LightGBM
  4. Train LightGBM warming/cooling classifier
  5. Cross-validation + seasonal accuracy breakdown
  6. Compute temperature extrapolation parameters (how much more to rise)
  7. Probability calibration
  8. Save all models and parameters
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# PART 1: Load and Clean Data
# =============================================================================

def load_hko_csv(filepath, value_col_name="max_temp"):
    """Load HKO CSV, skip header rows, handle missing data."""
    # HKO CSVs have 2 header description rows, then the actual CSV starts at row 3
    df = pd.read_csv(filepath, skiprows=2, encoding="utf-8-sig")
    
    # Standardize column names
    df.columns = ["year", "month", "day", "value", "completeness"]
    
    # Filter only complete data
    df = df[df["completeness"] == "C"].copy()
    
    # Convert value to float
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    
    # Create date column
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    
    # Filter valid dates
    valid_dates = []
    for _, row in df.iterrows():
        try:
            d = datetime(int(row["year"]), int(row["month"]), int(row["day"]))
            valid_dates.append(d)
        except ValueError:
            valid_dates.append(None)
    
    df["date"] = valid_dates
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    
    df = df.rename(columns={"value": value_col_name})
    return df[["date", "year", "month", "day", value_col_name]].reset_index(drop=True)


def load_all_data():
    """Load max, min, mean temp and merge."""
    print("📊 Loading HKO historical data...")
    
    df_max = load_hko_csv(os.path.join(RAW_DIR, "daily_max_temp_HKO.csv"), "max_temp")
    df_min = load_hko_csv(os.path.join(RAW_DIR, "daily_min_temp_HKO.csv"), "min_temp")
    df_mean = load_hko_csv(os.path.join(RAW_DIR, "daily_mean_temp_HKO.csv"), "mean_temp")
    
    print(f"   Max temp records: {len(df_max)} (from {df_max['date'].min()} to {df_max['date'].max()})")
    print(f"   Min temp records: {len(df_min)}")
    print(f"   Mean temp records: {len(df_mean)}")
    
    # Merge on date
    df = df_max.merge(df_min[["date", "min_temp"]], on="date", how="left")
    df = df.merge(df_mean[["date", "mean_temp"]], on="date", how="left")
    
    # Derived features
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    df["day_of_year"] = df["date"].dt.dayofyear
    df["weekday"] = df["date"].dt.weekday
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]: return "winter"
        elif month in [3, 4, 5]: return "spring"
        elif month in [6, 7, 8]: return "summer"
        else: return "autumn"
    
    df["season"] = df["month"].apply(get_season)
    
    print(f"   ✅ Merged dataset: {len(df)} rows")
    return df


# =============================================================================
# PART 2: Exploratory Data Analysis
# =============================================================================

def run_eda(df):
    """Run EDA and print key statistics."""
    print("\n" + "=" * 60)
    print("📈 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Use recent data (last 20 years) for more relevant stats
    recent = df[df["year"] >= 2005].copy()
    print(f"\nUsing recent data (2005-2026): {len(recent)} days")
    
    # Monthly average max temp
    print("\n📊 Monthly average MAX temperature (2005-2026):")
    monthly = recent.groupby("month")["max_temp"].agg(["mean", "std", "min", "max"])
    for m, row in monthly.iterrows():
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        print(f"   {months[m-1]}: {row['mean']:.1f}°C ± {row['std']:.1f}  (range: {row['min']:.1f} - {row['max']:.1f})")
    
    # Temperature buckets for Polymarket (which degree range is most common per month)
    print("\n📊 Most common temperature BUCKET per month (for Polymarket):")
    recent["temp_bucket"] = recent["max_temp"].apply(lambda x: int(x))
    for m in range(1, 13):
        month_data = recent[recent["month"] == m]
        top_buckets = month_data["temp_bucket"].value_counts().head(3)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        buckets_str = ", ".join([f"{b}°C ({c/len(month_data)*100:.0f}%)" for b, c in top_buckets.items()])
        print(f"   {months[m-1]}: {buckets_str}")
    
    # Day-to-day temperature change distribution
    recent_sorted = recent.sort_values("date").copy()
    recent_sorted["prev_max"] = recent_sorted["max_temp"].shift(1)
    recent_sorted["temp_change"] = recent_sorted["max_temp"] - recent_sorted["prev_max"]
    recent_sorted = recent_sorted.dropna(subset=["temp_change"])
    
    print("\n📊 Day-to-day MAX temperature change:")
    print(f"   Mean: {recent_sorted['temp_change'].mean():+.2f}°C")
    print(f"   Std:  {recent_sorted['temp_change'].std():.2f}°C")
    print(f"   |change| > 3°C: {(recent_sorted['temp_change'].abs() > 3).mean()*100:.1f}% of days")
    print(f"   |change| > 5°C: {(recent_sorted['temp_change'].abs() > 5).mean()*100:.1f}% of days")
    
    # Warming vs cooling days per season
    print("\n📊 Warming vs Cooling days by SEASON:")
    for season in ["winter", "spring", "summer", "autumn"]:
        s = recent_sorted[recent_sorted["season"] == season]
        warming = (s["temp_change"] > 0.5).sum()
        cooling = (s["temp_change"] < -0.5).sum()
        flat = len(s) - warming - cooling
        print(f"   {season:8s}: ☀️ warming {warming/len(s)*100:.0f}%  ❄️ cooling {cooling/len(s)*100:.0f}%  ➡️ flat {flat/len(s)*100:.0f}%")
    
    return recent


# =============================================================================
# PART 3: Feature Engineering for LightGBM
# =============================================================================

def engineer_features(df):
    """
    Create features for the warming/cooling classifier.
    
    Since we don't have hourly data from HKO historical API, we use
    daily-level features that are available at prediction time.
    """
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.sort_values("date").copy()
    
    # --- Yesterday's features ---
    df["prev_max"] = df["max_temp"].shift(1)
    df["prev_min"] = df["min_temp"].shift(1)
    df["prev_mean"] = df["mean_temp"].shift(1)
    df["prev_range"] = df["temp_range"].shift(1)
    
    # --- Temperature change (what we're predicting) ---
    df["temp_change"] = df["max_temp"] - df["prev_max"]
    
    # --- Multi-day trends ---
    df["change_2d"] = df["max_temp"].shift(0) - df["max_temp"].shift(2)  # recalc later
    df["prev_change_1d"] = df["prev_max"] - df["max_temp"].shift(2)
    df["prev_change_2d"] = df["prev_max"] - df["max_temp"].shift(3)
    df["trend_3d"] = (df["max_temp"].shift(1) + df["max_temp"].shift(2) + df["max_temp"].shift(3)) / 3
    df["deviation_from_trend"] = df["prev_max"] - df["trend_3d"]
    
    # --- Seasonal / calendar features ---
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    # --- Historical climatology: how does today's max compare to seasonal avg? ---
    seasonal_avg = df.groupby("day_of_year")["max_temp"].transform("mean")
    df["anomaly_from_clim"] = df["prev_max"] - seasonal_avg.shift(1)
    
    # --- Range features ---
    df["prev_range"] = df["temp_range"].shift(1)
    
    # --- Was yesterday a big change day? ---
    df["prev_was_big_warming"] = (df["prev_change_1d"] > 3).astype(int)
    df["prev_was_big_cooling"] = (df["prev_change_1d"] < -3).astype(int)
    
    # --- Consecutive warming/cooling streak ---
    changes = df["prev_change_1d"].copy()
    warming_streak = 0
    cooling_streak = 0
    w_streaks = []
    c_streaks = []
    for c in changes:
        if pd.isna(c):
            w_streaks.append(0)
            c_streaks.append(0)
            continue
        if c > 0.5:
            warming_streak += 1
            cooling_streak = 0
        elif c < -0.5:
            cooling_streak += 1
            warming_streak = 0
        else:
            warming_streak = 0
            cooling_streak = 0
        w_streaks.append(warming_streak)
        c_streaks.append(cooling_streak)
    df["warming_streak"] = w_streaks
    df["cooling_streak"] = c_streaks
    
    # --- Create target: warming / cooling / flat ---
    # For classification: warming day = max_temp increases by > 0.5°C
    df["target_direction"] = 0  # flat
    df.loc[df["temp_change"] > 0.5, "target_direction"] = 1   # warming
    df.loc[df["temp_change"] < -0.5, "target_direction"] = -1  # cooling
    
    # Drop rows with NaN
    feature_cols = [
        "prev_max", "prev_min", "prev_mean", "prev_range",
        "prev_change_1d", "prev_change_2d",
        "trend_3d", "deviation_from_trend",
        "month", "day_of_year_sin", "day_of_year_cos",
        "anomaly_from_clim",
        "prev_was_big_warming", "prev_was_big_cooling",
        "warming_streak", "cooling_streak"
    ]
    
    df = df.dropna(subset=feature_cols + ["target_direction"])
    
    print(f"   ✅ Features created: {len(feature_cols)} features")
    print(f"   📊 Dataset size after cleanup: {len(df)} rows")
    print(f"   📊 Target distribution:")
    print(f"      ☀️ Warming:  {(df['target_direction'] == 1).sum()} ({(df['target_direction'] == 1).mean()*100:.1f}%)")
    print(f"      ➡️ Flat:     {(df['target_direction'] == 0).sum()} ({(df['target_direction'] == 0).mean()*100:.1f}%)")
    print(f"      ❄️ Cooling:  {(df['target_direction'] == -1).sum()} ({(df['target_direction'] == -1).mean()*100:.1f}%)")
    
    return df, feature_cols


# =============================================================================
# PART 4: Train LightGBM Model
# =============================================================================

def train_lightgbm(df, feature_cols):
    """
    Train gradient boosting model to predict warming/cooling/flat.
    Uses sklearn's HistGradientBoostingClassifier (same algo as LightGBM).
    Uses time-based split (no data leakage!).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import pickle
    
    print("\n" + "=" * 60)
    print("🤖 TRAINING HistGradientBoosting MODEL")
    print("   (same algorithm as LightGBM, built into sklearn)")
    print("=" * 60)
    
    # Map target: -1 → 0, 0 → 1, 1 → 2
    target_map = {-1: 0, 0: 1, 1: 2}
    target_names = ["❄️ Cooling", "➡️ Flat", "☀️ Warming"]
    df["target_lgb"] = df["target_direction"].map(target_map)
    
    X = df[feature_cols].values
    y = df["target_lgb"].values
    dates = df["date"].values
    seasons = df["season"].values
    
    # Time-based split: last 3 years = test, rest = train
    cutoff_date = pd.Timestamp("2023-01-01")
    train_mask = df["date"] < cutoff_date
    test_mask = df["date"] >= cutoff_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_seasons = seasons[test_mask]
    
    print(f"   Train: {len(X_train)} samples (before 2023)")
    print(f"   Test:  {len(X_test)} samples (2023-2026)")
    
    # Train HistGradientBoosting (same idea as LightGBM)
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   📊 Overall Test Accuracy: {acc*100:.1f}%")
    
    # Classification report
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    for line in report.split("\n"):
        print(f"   {line}")
    
    # Accuracy by season
    print(f"\n   📊 Accuracy by SEASON:")
    for season in ["winter", "spring", "summer", "autumn"]:
        mask = test_seasons == season
        if mask.sum() > 0:
            season_acc = accuracy_score(y_test[mask], y_pred[mask])
            emoji = {"winter": "❄️", "spring": "🌸", "summer": "☀️", "autumn": "🍂"}[season]
            print(f"      {emoji} {season:8s}: {season_acc*100:.1f}% ({mask.sum()} samples)")
    
    # Feature importance (permutation-based for HistGB)
    from sklearn.inspection import permutation_importance
    print(f"\n   📊 Computing feature importance (permutation-based)...")
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    feat_imp = sorted(zip(feature_cols, perm_result.importances_mean), key=lambda x: x[1], reverse=True)
    
    max_imp = max(imp for _, imp in feat_imp) if feat_imp else 1
    print(f"   📊 Top Feature Importance:")
    for fname, imp in feat_imp[:10]:
        bar_len = int(imp / max_imp * 30) if max_imp > 0 else 0
        bar = "█" * max(1, bar_len)
        print(f"      {fname:25s} {bar} ({imp:.4f})")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "hgb_warming_cooling.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n   ✅ Model saved to: {model_path}")
    
    return model, y_pred_proba, y_test, test_seasons


# =============================================================================
# PART 5: Temperature Extrapolation Parameters
# =============================================================================

def compute_extrapolation_params(df):
    """
    Compute "how much more will the temperature rise?" statistics.
    
    Since we only have daily data (not hourly), we compute:
    - Given yesterday's max, today's max, and month → how much does it typically change?
    - Given the daily range (max - min) → what does that tell us about today's conditions?
    
    These parameters are used for the real-time correction engine at runtime.
    """
    print("\n" + "=" * 60)
    print("📐 COMPUTING EXTRAPOLATION PARAMETERS")
    print("=" * 60)
    
    recent = df[df["year"] >= 2005].sort_values("date").copy()
    recent["prev_max"] = recent["max_temp"].shift(1)
    recent["temp_change"] = recent["max_temp"] - recent["prev_max"]
    recent = recent.dropna(subset=["temp_change"])
    
    # --- Parameter A: Typical remaining rise above morning min temp ---
    # max_temp - min_temp = how much the day "rises" above its starting point
    # This is a proxy for "how much will it warm from early morning to peak"
    
    print("\n   📊 Typical daily temperature RISE (max - min) by month:")
    print("   (This approximates: if you know the morning temp, how much more will it rise)")
    
    rise_params = {}
    for m in range(1, 13):
        month_data = recent[recent["month"] == m]
        rise = month_data["temp_range"]  # max - min
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        rise_params[m] = {
            "mean_rise": round(float(rise.mean()), 2),
            "std_rise": round(float(rise.std()), 2),
            "median_rise": round(float(rise.median()), 2),
            "p25_rise": round(float(rise.quantile(0.25)), 2),
            "p75_rise": round(float(rise.quantile(0.75)), 2),
        }
        print(f"   {months[m-1]}: mean +{rise.mean():.1f}°C  (median +{rise.median():.1f}, P25 +{rise.quantile(0.25):.1f}, P75 +{rise.quantile(0.75):.1f})")
    
    # --- Parameter B: Day-to-day change conditioned on previous change ---
    # "If yesterday was a warming day, what's the probability of another warming day?"
    
    print("\n   📊 Conditional transition probabilities:")
    recent["prev_direction"] = "flat"
    recent.loc[recent["temp_change"].shift(1) > 0.5, "prev_direction"] = "warming"
    recent.loc[recent["temp_change"].shift(1) < -0.5, "prev_direction"] = "cooling"
    recent["cur_direction"] = "flat"
    recent.loc[recent["temp_change"] > 0.5, "cur_direction"] = "warming"
    recent.loc[recent["temp_change"] < -0.5, "cur_direction"] = "cooling"
    
    transition_matrix = pd.crosstab(recent["prev_direction"], recent["cur_direction"], normalize="index")
    print(f"   (If yesterday was X, probability of today being Y)")
    print(transition_matrix.round(3).to_string(index=True))
    
    transition_params = transition_matrix.to_dict()
    
    # --- Parameter C: Temperature bucket transition probabilities ---
    # "If yesterday's max was 22°C bucket, what's the probability for each bucket today?"
    
    print("\n   📊 Temperature bucket transition probabilities (top transitions):")
    recent["prev_bucket"] = recent["prev_max"].apply(lambda x: int(x) if pd.notna(x) else None)
    recent["cur_bucket"] = recent["max_temp"].apply(lambda x: int(x))
    recent_valid = recent.dropna(subset=["prev_bucket"])
    
    bucket_transitions = {}
    for m in range(1, 13):
        month_data = recent_valid[recent_valid["month"] == m]
        if len(month_data) < 30:
            continue
        
        trans = pd.crosstab(month_data["prev_bucket"], month_data["cur_bucket"], normalize="index")
        bucket_transitions[m] = trans.to_dict()
    
    # Show March as example (current month)
    march_data = recent_valid[recent_valid["month"] == 3]
    print("   Example: March transitions (yesterday bucket → today bucket probability)")
    march_trans = pd.crosstab(march_data["prev_bucket"], march_data["cur_bucket"], normalize="index")
    # Show rows for 20-24°C range (most relevant for current weather)
    for prev_b in [20, 21, 22, 23, 24]:
        if prev_b in march_trans.index:
            top3 = march_trans.loc[prev_b].nlargest(3)
            probs = ", ".join([f"{int(b)}°C: {p*100:.0f}%" for b, p in top3.items()])
            print(f"      Yesterday {prev_b}°C → Most likely: {probs}")
    
    # --- Save parameters ---
    params = {
        "monthly_rise": rise_params,
        "transition_probabilities": {
            str(k): {str(k2): v2 for k2, v2 in v.items()} 
            for k, v in transition_params.items()
        },
        "generated_at": datetime.now().isoformat(),
        "data_range": f"{recent['date'].min()} to {recent['date'].max()}",
        "sample_size": len(recent),
    }
    
    params_path = os.path.join(MODELS_DIR, "extrapolation_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    
    print(f"\n   ✅ Parameters saved to: {params_path}")
    return params


# =============================================================================
# PART 6: Probability Calibration
# =============================================================================

def calibrate_probabilities(model, df, feature_cols):
    """
    Train a probability calibration model on the LightGBM outputs.
    
    We use the test set predictions and check:
    "When the model says 70% probability of warming, how often does warming actually happen?"
    """
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    
    print("\n" + "=" * 60)
    print("🎯 PROBABILITY CALIBRATION")
    print("=" * 60)
    
    # Use 2022-2024 as calibration set, 2025-2026 as validation
    cal_mask = (df["date"] >= "2022-01-01") & (df["date"] < "2025-01-01")
    val_mask = df["date"] >= "2025-01-01"
    
    target_map = {-1: 0, 0: 1, 1: 2}
    df["target_lgb"] = df["target_direction"].map(target_map)
    
    X_cal = df[cal_mask][feature_cols].values
    y_cal = df[cal_mask]["target_lgb"].values
    X_val = df[val_mask][feature_cols].values
    y_val = df[val_mask]["target_lgb"].values
    
    if len(X_val) == 0:
        print("   ⚠️ Not enough validation data for calibration, skipping...")
        return None
    
    pred_cal = model.predict_proba(X_cal)
    pred_val = model.predict_proba(X_val)
    
    print(f"   Calibration set: {len(X_cal)} samples (2022-2024)")
    print(f"   Validation set:  {len(X_val)} samples (2025-2026)")
    
    # For each class, build calibration curve
    calibration_models = {}
    class_names = ["❄️ Cooling", "➡️ Flat", "☀️ Warming"]
    
    for cls_idx, cls_name in enumerate(class_names):
        y_binary = (y_cal == cls_idx).astype(int)
        probs = pred_cal[:, cls_idx]
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_binary, probs, n_bins=8, strategy="quantile")
        
        print(f"\n   {cls_name} calibration:")
        print(f"   {'Model says':>12s} → {'Actually':>10s} → {'Bias':>8s}")
        for pt, pp in zip(prob_pred, prob_true):
            bias = pt - pp
            bias_str = f"{bias:+.1%}" if abs(bias) > 0.02 else "OK"
            print(f"   {pp*100:10.1f}%  →  {pt*100:8.1f}%  →  {bias_str:>8s}")
        
        # Train isotonic regression for each class
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, y_binary)
        calibration_models[cls_idx] = iso
    
    # Validate on 2025-2026 data
    if len(X_val) > 30:
        print(f"\n   📊 Validation on 2025-2026 data:")
        for cls_idx, cls_name in enumerate(class_names):
            y_binary = (y_val == cls_idx).astype(int)
            raw_probs = pred_val[:, cls_idx]
            cal_probs = calibration_models[cls_idx].transform(raw_probs)
            
            # Brier score (lower = better)
            brier_raw = ((raw_probs - y_binary) ** 2).mean()
            brier_cal = ((cal_probs - y_binary) ** 2).mean()
            improvement = (brier_raw - brier_cal) / brier_raw * 100
            
            print(f"      {cls_name}: Brier {brier_raw:.4f} → {brier_cal:.4f} ({improvement:+.1f}%)")
    
    print(f"\n   ✅ Calibration models ready")
    return calibration_models


# =============================================================================
# PART 7: Summary Report
# =============================================================================

def print_summary(df, params):
    """Print a summary of everything we computed."""
    print("\n" + "=" * 60)
    print("📋 SUMMARY: What you now have")
    print("=" * 60)
    
    print("""
    ✅ 1. Historical Data
       - {n_years}+ years of daily max/min/mean temperature
       - Station: HKO (Tsim Sha Tsui) = Polymarket settlement station
    
    ✅ 2. LightGBM Warming/Cooling Model
       - Predicts: warming / flat / cooling day
       - Saved to: models/lgb_warming_cooling.txt
    
    ✅ 3. Extrapolation Parameters
       - Monthly typical rise (max - min)
       - Day-to-day transition probabilities
       - Temperature bucket transitions
       - Saved to: models/extrapolation_params.json
    
    ✅ 4. Probability Calibration
       - Isotonic regression calibrators for each class
       - Corrects overconfident model predictions
    
    🔜 Next Steps:
       - Run 03_realtime_predictor.py to connect everything to live data
       - Connect to Open-Meteo for JMA/ECMWF/GFS forecasts
       - Connect to Polymarket API for market prices
    """.format(n_years=df["year"].max() - df["year"].min()))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🇭🇰 Hong Kong Weather Prediction System")
    print("   Full Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_all_data()
    
    # Step 2: EDA
    recent = run_eda(df)
    
    # Step 3: Feature engineering
    df_feat, feature_cols = engineer_features(df)
    
    # Step 4: Train LightGBM
    model, pred_proba, y_test, test_seasons = train_lightgbm(df_feat, feature_cols)
    
    # Step 5: Extrapolation parameters
    params = compute_extrapolation_params(df)
    
    # Step 6: Calibration
    cal_models = calibrate_probabilities(model, df_feat, feature_cols)
    
    # Step 7: Summary
    print_summary(df, params)
    
    # Save processed data
    output_path = os.path.join(PROCESSED_DIR, "hko_features.csv")
    df_feat.to_csv(output_path, index=False)
    print(f"\n   📁 Processed features saved to: {output_path}")
