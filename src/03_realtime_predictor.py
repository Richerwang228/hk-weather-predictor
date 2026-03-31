"""
Step 3: Real-time Hong Kong Weather Prediction Engine

This is the MAIN SCRIPT that runs during market hours. It:
  1. Fetches forecasts from JMA, ECMWF, GFS (via Open-Meteo)
  2. Fetches real-time observations from HKO
  3. Runs Kalman-gain correction (blend forecast with observations)
  4. Computes temperature bucket probabilities
  5. Compares with Polymarket prices to find Edge
  6. Outputs trading signals

Usage:
  python src/03_realtime_predictor.py              # Run full prediction for today
  python src/03_realtime_predictor.py --tomorrow    # Run prediction for tomorrow
"""

import requests
import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import Counter

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HK_LAT = 22.30
HK_LON = 114.17
HK_TZ = timezone(timedelta(hours=8))

# Ensemble weights
MODEL_WEIGHTS = {
    "jma": 0.35,
    "ecmwf": 0.30,
    "gfs_mean": 0.20,
    "historical": 0.15,  # from bucket transition probabilities
}

# Edge threshold for trading
MIN_EDGE = 0.08       # 8% minimum edge before considering a bet
KELLY_FRACTION = 0.15  # use 15% Kelly (conservative)

# =============================================================================
# MODULE 1: Fetch Forecasts from Open-Meteo
# =============================================================================

def fetch_jma_forecast(target_date):
    """Fetch JMA model forecast for Hong Kong."""
    url = "https://api.open-meteo.com/v1/jma"
    params = {
        "latitude": HK_LAT, "longitude": HK_LON,
        "hourly": "temperature_2m",
        "forecast_days": 3,
        "timezone": "Asia/Hong_Kong"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Extract temperatures for target date
        temps = []
        for t, v in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]):
            if v is not None and t.startswith(target_date):
                temps.append(v)
        
        if not temps:
            return None
        
        return {
            "model": "JMA",
            "max_temp": max(temps),
            "min_temp": min(temps),
            "hourly": temps,
            "peak_hour": temps.index(max(temps)),
        }
    except Exception as e:
        print(f"   ⚠️ JMA fetch failed: {e}")
        return None


def fetch_ecmwf_forecast(target_date):
    """Fetch ECMWF model forecast for Hong Kong."""
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": HK_LAT, "longitude": HK_LON,
        "hourly": "temperature_2m",
        "forecast_days": 3,
        "timezone": "Asia/Hong_Kong"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        temps = []
        for t, v in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]):
            if v is not None and t.startswith(target_date):
                temps.append(v)
        
        if not temps:
            return None
        
        return {
            "model": "ECMWF",
            "max_temp": max(temps),
            "min_temp": min(temps),
            "hourly": temps,
            "peak_hour": temps.index(max(temps)),
        }
    except Exception as e:
        print(f"   ⚠️ ECMWF fetch failed: {e}")
        return None


def fetch_gfs_ensemble(target_date):
    """Fetch GFS ensemble (30 members) for Hong Kong."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": HK_LAT, "longitude": HK_LON,
        "hourly": "temperature_2m",
        "models": "gfs_seamless",
        "forecast_days": 3,
        "timezone": "Asia/Hong_Kong"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # Find all ensemble member keys
        member_keys = [k for k in data["hourly"].keys() if k.startswith("temperature_2m_member")]
        
        # Get times for target date
        times = data["hourly"]["time"]
        date_indices = [i for i, t in enumerate(times) if t.startswith(target_date)]
        
        if not date_indices:
            return None
        
        # Get max temp from each member for target date
        member_maxes = []
        for key in member_keys:
            vals = [data["hourly"][key][i] for i in date_indices]
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                member_maxes.append(max(valid_vals))
        
        member_maxes.sort()
        
        return {
            "model": "GFS_Ensemble",
            "n_members": len(member_maxes),
            "member_maxes": member_maxes,
            "mean": np.mean(member_maxes),
            "std": np.std(member_maxes),
            "min": min(member_maxes),
            "max": max(member_maxes),
            "p10": np.percentile(member_maxes, 10),
            "p50": np.percentile(member_maxes, 50),
            "p90": np.percentile(member_maxes, 90),
        }
    except Exception as e:
        print(f"   ⚠️ GFS ensemble fetch failed: {e}")
        return None


def fetch_all_forecasts(target_date):
    """Fetch all model forecasts."""
    print(f"\n📡 Fetching forecasts for {target_date}...")
    
    jma = fetch_jma_forecast(target_date)
    ecmwf = fetch_ecmwf_forecast(target_date)
    gfs = fetch_gfs_ensemble(target_date)
    
    print(f"   🇯🇵 JMA:    max = {jma['max_temp']:.1f}°C" if jma else "   🇯🇵 JMA:    ❌ failed")
    print(f"   🇪🇺 ECMWF:  max = {ecmwf['max_temp']:.1f}°C" if ecmwf else "   🇪🇺 ECMWF:  ❌ failed")
    if gfs:
        print(f"   🇺🇸 GFS:    mean = {gfs['mean']:.1f}°C (±{gfs['std']:.1f}, range {gfs['min']:.1f}-{gfs['max']:.1f}, {gfs['n_members']} members)")
    else:
        print(f"   🇺🇸 GFS:    ❌ failed")
    
    return {"jma": jma, "ecmwf": ecmwf, "gfs": gfs}


# =============================================================================
# MODULE 2: Fetch Real-time HKO Observations
# =============================================================================

def fetch_hko_realtime():
    """Fetch current temperature from HKO Observatory station."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    params = {"dataType": "rhrread", "lang": "en"}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Get HKO station temp
        hko_temp = None
        all_temps = {}
        for t in data.get("temperature", {}).get("data", []):
            all_temps[t["place"]] = t["value"]
            if t["place"] == "Hong Kong Observatory":
                hko_temp = t["value"]
        
        update_time = data.get("updateTime", "")
        humidity = data.get("humidity", {}).get("data", [{}])[0].get("value")
        
        # Check for warning signals
        warning = data.get("warningMessage", "")
        
        return {
            "hko_temp": hko_temp,
            "all_temps": all_temps,
            "update_time": update_time,
            "humidity": humidity,
            "warning": warning if warning else None,
        }
    except Exception as e:
        print(f"   ⚠️ HKO real-time fetch failed: {e}")
        return None


def fetch_hko_warnings():
    """Fetch any active HKO warning signals."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    params = {"dataType": "warningInfo", "lang": "en"}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        warnings_list = []
        for detail in data.get("details", []):
            warnings_list.append({
                "name": detail.get("warningStatementCode", ""),
                "contents": detail.get("contents", []),
            })
        return warnings_list
    except Exception:
        return []


# =============================================================================
# MODULE 3: Compute Bucket Probabilities
# =============================================================================

def compute_bucket_probabilities(forecasts, historical_params, current_obs=None, current_hour=None):
    """
    Compute probability for each temperature bucket (e.g., 20°C, 21°C, ...).
    
    Combines:
    1. JMA point forecast → Gaussian spread around predicted max
    2. ECMWF point forecast → Gaussian spread
    3. GFS ensemble → direct member counting
    4. Historical bucket transitions → prior
    5. Real-time Kalman correction (if observations available)
    """
    
    # Determine bucket range (typically 5-degree window centered on forecasts)
    forecast_maxes = []
    if forecasts.get("jma"):
        forecast_maxes.append(forecasts["jma"]["max_temp"])
    if forecasts.get("ecmwf"):
        forecast_maxes.append(forecasts["ecmwf"]["max_temp"])
    if forecasts.get("gfs"):
        forecast_maxes.append(forecasts["gfs"]["mean"])
    
    if not forecast_maxes:
        return None
    
    center = np.mean(forecast_maxes)
    buckets = list(range(int(center) - 5, int(center) + 6))
    
    probs = {b: 0.0 for b in buckets}
    
    # --- Source 1: JMA (Gaussian spread, σ = 1.5°C typical NWP uncertainty) ---
    if forecasts.get("jma"):
        jma_max = forecasts["jma"]["max_temp"]
        sigma = 1.5
        for b in buckets:
            # P(temp in [b, b+1)) using Gaussian CDF
            from scipy.stats import norm
            p = norm.cdf(b + 1, jma_max, sigma) - norm.cdf(b, jma_max, sigma)
            probs[b] += MODEL_WEIGHTS["jma"] * p
    
    # --- Source 2: ECMWF (Gaussian spread, σ = 1.3°C) ---
    if forecasts.get("ecmwf"):
        ecmwf_max = forecasts["ecmwf"]["max_temp"]
        sigma = 1.3
        for b in buckets:
            from scipy.stats import norm
            p = norm.cdf(b + 1, ecmwf_max, sigma) - norm.cdf(b, ecmwf_max, sigma)
            probs[b] += MODEL_WEIGHTS["ecmwf"] * p
    
    # --- Source 3: GFS Ensemble (direct member counting) ---
    if forecasts.get("gfs"):
        member_maxes = forecasts["gfs"]["member_maxes"]
        n = len(member_maxes)
        for b in buckets:
            count = sum(1 for m in member_maxes if b <= m < b + 1)
            probs[b] += MODEL_WEIGHTS["gfs_mean"] * (count / n)
    
    # --- Source 4: Historical prior ---
    if historical_params:
        # Use a flat prior with slight seasonal adjustment
        # This gets overridden by other sources but prevents zero probabilities
        for b in buckets:
            probs[b] += MODEL_WEIGHTS["historical"] * (1 / len(buckets))
    
    # --- Kalman Correction (if real-time data available) ---
    if current_obs and current_hour is not None and current_hour >= 8:
        probs = apply_kalman_correction(probs, buckets, forecasts, current_obs, current_hour)
    
    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {b: p / total for b, p in probs.items()}
    
    return probs


# =============================================================================
# MODULE 4: Kalman-Gain Real-time Correction
# =============================================================================

def apply_kalman_correction(probs, buckets, forecasts, current_obs, current_hour):
    """
    Apply Kalman-gain style correction based on real-time observations.
    
    Key idea: As the day progresses, we trust observations more than forecasts.
    The Kalman gain K increases from 0 (midnight) to 1 (after peak).
    
    If current temperature is already 22°C at 10am and typical remaining rise
    in March is +2-3°C, we can estimate max around 24-25°C.
    """
    
    # Load extrapolation params
    params_path = os.path.join(MODELS_DIR, "extrapolation_params.json")
    if not os.path.exists(params_path):
        return probs
    
    with open(params_path) as f:
        extrap = json.load(f)
    
    hko_temp = current_obs.get("hko_temp")
    if hko_temp is None:
        return probs
    
    # Kalman gain schedule: how much to trust observations vs forecast
    # Increases through the day as we have more data
    kalman_schedule = {
        6: 0.10, 7: 0.15, 8: 0.20, 9: 0.30,
        10: 0.40, 11: 0.50, 12: 0.65, 13: 0.75,
        14: 0.85, 15: 0.90, 16: 0.95, 17: 1.0,
    }
    K = kalman_schedule.get(current_hour, 0.0 if current_hour < 6 else 1.0)
    
    # Estimate max from current observation
    month = datetime.now(HK_TZ).month
    month_str = str(month)
    
    if month_str in extrap.get("monthly_rise", {}):
        rise_stats = extrap["monthly_rise"][month_str]
        mean_rise = rise_stats["mean_rise"]
        std_rise = rise_stats["std_rise"]
    else:
        mean_rise = 4.5  # fallback
        std_rise = 1.5
    
    # After peak (typ. 13-15h), max is approximately current max observed
    if current_hour >= 14:
        # The current temp IS approximately the max (or max already occurred)
        estimated_max = hko_temp  # Already past peak
        sigma_obs = 0.5  # Very confident
    elif current_hour >= 11:
        # Approaching peak: small additional rise expected
        remaining_fraction = max(0, (14 - current_hour) / 14)  # fraction of rise remaining
        additional_rise = mean_rise * remaining_fraction * 0.3  # conservative
        estimated_max = hko_temp + additional_rise
        sigma_obs = 1.0
    else:
        # Morning: significant rise still expected
        # Use hour-based fraction of daily rise
        # At 8am, roughly 30% of daily rise completed
        hour_fractions = {6: 0.0, 7: 0.1, 8: 0.25, 9: 0.40, 10: 0.60}
        fraction_done = hour_fractions.get(current_hour, 0.3)
        
        # current_temp ≈ min_temp + fraction_done * daily_rise
        # max_temp ≈ current_temp + (1 - fraction_done) * daily_rise  
        remaining_rise = mean_rise * (1 - fraction_done)
        estimated_max = hko_temp + remaining_rise
        sigma_obs = std_rise * (1 - fraction_done) + 0.5
    
    print(f"   🔄 Kalman correction (K={K:.2f}, hour={current_hour})")
    print(f"      Current HKO temp: {hko_temp}°C")
    print(f"      Estimated max:    {estimated_max:.1f}°C (σ={sigma_obs:.1f})")
    
    # Compute observation-based probabilities
    from scipy.stats import norm
    obs_probs = {}
    for b in buckets:
        p = norm.cdf(b + 1, estimated_max, sigma_obs) - norm.cdf(b, estimated_max, sigma_obs)
        obs_probs[b] = p
    
    # Blend: corrected = (1-K) * forecast + K * observation
    corrected = {}
    for b in buckets:
        corrected[b] = (1 - K) * probs.get(b, 0) + K * obs_probs.get(b, 0)
    
    return corrected


# =============================================================================
# MODULE 5: Edge Calculation + Trading Signal
# =============================================================================

def compute_edge(our_probs, market_prices=None):
    """
    Compute edge for each bucket.
    
    Edge = Our probability - Market implied probability.
    If edge > MIN_EDGE, it's a trading opportunity.
    """
    
    if market_prices is None:
        # If no market data, just show our probabilities
        return None
    
    edges = {}
    for bucket, our_p in our_probs.items():
        market_p = market_prices.get(bucket, None)
        if market_p is not None and market_p > 0:
            edge = our_p - market_p
            kelly = 0
            if edge > 0:
                odds = (1 / market_p) - 1
                if odds > 0:
                    kelly = KELLY_FRACTION * (edge / odds)
            
            edges[bucket] = {
                "our_prob": our_p,
                "market_prob": market_p,
                "edge": edge,
                "kelly_fraction": kelly,
                "signal": "BUY" if edge >= MIN_EDGE else ("WATCH" if edge >= 0.04 else "SKIP"),
            }
    
    return edges


# =============================================================================
# MODULE 6: Main Prediction Runner
# =============================================================================

def run_prediction(target_date=None, market_prices=None):
    """
    Run the full prediction pipeline for a given date.
    
    Args:
        target_date: str "YYYY-MM-DD" (default: today)
        market_prices: dict {bucket_int: probability} from Polymarket
    """
    
    now = datetime.now(HK_TZ)
    
    if target_date is None:
        target_date = now.strftime("%Y-%m-%d")
    
    is_today = target_date == now.strftime("%Y-%m-%d")
    current_hour = now.hour if is_today else None
    
    print("=" * 65)
    print(f"🇭🇰 Hong Kong Weather Prediction Engine")
    print(f"   Target date: {target_date}")
    print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} HKT")
    if is_today:
        print(f"   Mode: LIVE (real-time correction active)")
    else:
        print(f"   Mode: FORECAST ONLY (no real-time data)")
    print("=" * 65)
    
    # --- Step 1: Load historical parameters ---
    params_path = os.path.join(MODELS_DIR, "extrapolation_params.json")
    historical_params = None
    if os.path.exists(params_path):
        with open(params_path) as f:
            historical_params = json.load(f)
    
    # --- Step 2: Fetch all forecasts ---
    forecasts = fetch_all_forecasts(target_date)
    
    # --- Step 3: Fetch real-time observations (if today) ---
    current_obs = None
    if is_today:
        print(f"\n📡 Fetching real-time HKO observations...")
        current_obs = fetch_hko_realtime()
        if current_obs:
            print(f"   🌡️  HKO station: {current_obs['hko_temp']}°C")
            print(f"   💧 Humidity: {current_obs['humidity']}%")
            print(f"   🕐 Update: {current_obs['update_time']}")
            if current_obs.get("warning"):
                print(f"   ⚠️  WARNING: {current_obs['warning']}")
        
        # Check for active warnings
        warnings = fetch_hko_warnings()
        if warnings:
            print(f"   🚨 Active warnings: {[w['name'] for w in warnings]}")
    
    # --- Step 4: Compute bucket probabilities ---
    print(f"\n📊 Computing bucket probabilities...")
    probs = compute_bucket_probabilities(
        forecasts, historical_params, current_obs, current_hour
    )
    
    if probs is None:
        print("   ❌ Could not compute probabilities (no forecast data)")
        return
    
    # Sort and display
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'─' * 65}")
    print(f"   {'Bucket':>8s}  {'Probability':>12s}  {'Bar':30s}")
    print(f"{'─' * 65}")
    
    for bucket, prob in sorted_probs:
        if prob >= 0.01:  # Only show buckets with ≥1% probability
            bar = "█" * int(prob * 50)
            marker = " ← MOST LIKELY" if prob == max(probs.values()) else ""
            print(f"   {bucket:>5d}°C  {prob:>10.1%}    {bar}{marker}")
    
    # --- Step 5: Edge calculation ---
    if market_prices:
        print(f"\n📈 Edge Analysis vs Polymarket:")
        edges = compute_edge(probs, market_prices)
        
        print(f"{'─' * 65}")
        print(f"   {'Bucket':>8s}  {'Our':>6s}  {'Market':>8s}  {'Edge':>8s}  {'Kelly':>7s}  {'Signal':>6s}")
        print(f"{'─' * 65}")
        
        for bucket, info in sorted(edges.items()):
            signal_emoji = {"BUY": "✅", "WATCH": "👀", "SKIP": "  "}.get(info["signal"], "")
            print(f"   {bucket:>5d}°C  {info['our_prob']:>5.1%}  {info['market_prob']:>7.1%}  "
                  f"{info['edge']:>+7.1%}  {info['kelly_fraction']:>6.1%}  {signal_emoji} {info['signal']}")
        
        # Show recommended trades
        buys = {b: v for b, v in edges.items() if v["signal"] == "BUY"}
        if buys:
            print(f"\n🎯 RECOMMENDED TRADES:")
            for bucket, info in sorted(buys.items(), key=lambda x: x[1]["edge"], reverse=True):
                bankroll_example = 500
                bet_size = bankroll_example * info["kelly_fraction"]
                print(f"   BUY {bucket}°C @ {info['market_prob']:.0%} "
                      f"(edge={info['edge']:+.1%}, Kelly bet=${bet_size:.1f} of ${bankroll_example})")
        else:
            print(f"\n   No trades meet the {MIN_EDGE:.0%} edge threshold.")
    else:
        print(f"\n💡 No market prices provided. To see edge analysis, pass market_prices dict.")
        print(f"   Example: run_prediction(market_prices={{23: 0.30, 24: 0.25, 25: 0.20}})")
    
    # --- Step 6: Confidence assessment ---
    print(f"\n📋 Confidence Assessment:")
    top_bucket = sorted_probs[0]
    top_two = sorted_probs[:2]
    concentration = top_bucket[1]
    spread = abs(top_two[0][0] - top_two[1][0]) if len(top_two) >= 2 else 0
    
    if concentration >= 0.40:
        confidence = "HIGH"
        emoji = "🟢"
    elif concentration >= 0.25:
        confidence = "MEDIUM"
        emoji = "🟡"
    else:
        confidence = "LOW"
        emoji = "🔴"
    
    print(f"   {emoji} Overall: {confidence}")
    print(f"   Top bucket: {top_bucket[0]}°C ({top_bucket[1]:.1%})")
    print(f"   Forecast spread: {spread}°C between top-2 buckets")
    
    if is_today and current_hour:
        if current_hour < 8:
            print(f"   ⏰ Time: Still early. Wait for real observations before trading.")
        elif current_hour < 11:
            print(f"   ⏰ Time: Window 1 (first correction). Good entry if edge is large.")
        elif current_hour < 14:
            print(f"   ⏰ Time: Window 2 (peak window). Best accuracy, lower edge.")
        else:
            print(f"   ⏰ Time: Past peak. Temperature likely locked in.")
    
    return {
        "target_date": target_date,
        "probabilities": probs,
        "forecasts": {
            "jma_max": forecasts["jma"]["max_temp"] if forecasts.get("jma") else None,
            "ecmwf_max": forecasts["ecmwf"]["max_temp"] if forecasts.get("ecmwf") else None,
            "gfs_mean": forecasts["gfs"]["mean"] if forecasts.get("gfs") else None,
        },
        "current_obs": current_obs,
        "confidence": confidence,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    target = None
    if "--tomorrow" in sys.argv:
        target = (datetime.now(HK_TZ) + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Example: if you have Polymarket prices, pass them here
    # Format: {bucket_int: market_implied_probability}
    # e.g., from Polymarket: "23°C" is trading at 0.25 = 25%
    example_market_prices = None  # Set to None to skip edge analysis
    
    # Uncomment and fill in actual Polymarket prices to see edge:
    # example_market_prices = {
    #     20: 0.05,
    #     21: 0.10,
    #     22: 0.15,
    #     23: 0.25,
    #     24: 0.20,
    #     25: 0.15,
    #     26: 0.05,
    # }
    
    result = run_prediction(target_date=target, market_prices=example_market_prices)
