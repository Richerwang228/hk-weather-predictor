"""
HK Weather Prediction Dashboard — Web Server v2

Premium web dashboard at http://localhost:8080
- Streaming step-by-step analysis
- Polymarket date auto-detection
- Distance-based confidence decay

Usage:
    python src/server.py
    → Open http://localhost:8080 in browser
"""

import http.server
import json
import os
import re
import sys
import socketserver
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import requests
import numpy as np

HK_TZ = timezone(timedelta(hours=8))
HK_LAT = 22.30
HK_LON = 114.17
PORT = 8080
MODELS_DIR = os.path.join(BASE_DIR, "models")
UHI_OFFSET_CLEAR = 0.5  # UHI on clear/sunny days (strong heat island)
UHI_OFFSET_CLOUDY = 0.2  # UHI on overcast days (weak heat island)


# ──────────────────────────────────────────────
# Polymarket helpers
# ──────────────────────────────────────────────

def search_polymarket_hk_weather():
    """Fetch HK weather markets by constructing known slug patterns.
    
    Polymarket HK weather events use the slug pattern:
      highest-temperature-in-hong-kong-on-{month}-{day}-{year}
    We probe today through +10 days to discover all open markets.
    """
    MONTH_NAMES = {1:"january",2:"february",3:"march",4:"april",
                   5:"may",6:"june",7:"july",8:"august",
                   9:"september",10:"october",11:"november",12:"december"}
    try:
        all_markets = []
        now = datetime.now(HK_TZ)
        # Probe today-1 through today+10
        for delta in range(-1, 11):
            dt = now + timedelta(days=delta)
            month_name = MONTH_NAMES[dt.month]
            slug = f"highest-temperature-in-hong-kong-on-{month_name}-{dt.day}-{dt.year}"
            date_str = dt.strftime("%Y-%m-%d")

            try:
                r = requests.get("https://gamma-api.polymarket.com/events",
                    params={"slug": slug}, timeout=8)
                if r.status_code != 200:
                    continue
                events = r.json()
                if not events:
                    continue
            except:
                continue

            e = events[0]
            for m in e.get("markets", []):
                prices = m.get("outcomePrices", "")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices) if prices else []
                    except:
                        prices = []
                q = m.get("question", "")
                bucket = extract_bucket_from_question(q)
                all_markets.append({
                    "event_title": e.get("title", ""),
                    "event_id": e.get("id"),
                    "question": q,
                    "market_id": m.get("id"),
                    "yes_price": float(prices[0]) if len(prices) > 0 else 0,
                    "no_price": float(prices[1]) if len(prices) > 1 else 0,
                    "slug": slug,
                    "date": date_str,
                    "bucket": bucket,
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity": float(m.get("liquidityNum", 0) or m.get("liquidity", 0) or 0),
                })
        return all_markets
    except Exception as ex:
        return {"error": str(ex)}


def extract_date_from_question(q):
    """Extract date from Polymarket question text."""
    # Patterns: "on March 21", "March 21, 2026", "on 3/21"
    months = {"january":"01","february":"02","march":"03","april":"04",
              "may":"05","june":"06","july":"07","august":"08",
              "september":"09","october":"10","november":"11","december":"12"}
    for mname, mnum in months.items():
        pat = re.search(rf'{mname}\s+(\d{{1,2}})', q, re.IGNORECASE)
        if pat:
            day = int(pat.group(1))
            year = datetime.now(HK_TZ).year
            return f"{year}-{mnum}-{day:02d}"
    return None


def extract_bucket_from_question(q):
    """Extract temperature bucket from question text.
    
    Handles: 'be 23°C', '17°C or below', '27°C or higher'
    Returns (bucket_int, bucket_type) where type is 'exact' / 'below' / 'above'
    or just bucket_int for compatibility.
    """
    # Check "or below" / "or higher" first
    m_below = re.search(r'(\d{1,2})\s*°\s*[Cc]\s+or\s+below', q, re.IGNORECASE)
    m_above = re.search(r'(\d{1,2})\s*°\s*[Cc]\s+or\s+(?:higher|above)', q, re.IGNORECASE)
    if m_below:
        return int(m_below.group(1))  # treat as that bucket (includes all below)
    if m_above:
        return int(m_above.group(1))  # treat as that bucket (includes all above)
    # Standard: "be 23°C"
    m = re.search(r'(\d{1,2})\s*°\s*[CcFf]', q)
    if m:
        return int(m.group(1))
    return None


def get_available_dates():
    """Get unique dates from Polymarket HK weather markets."""
    markets = search_polymarket_hk_weather()
    if isinstance(markets, dict) and "error" in markets:
        return {"error": markets["error"], "dates": [], "markets": []}

    dates = set()
    for m in markets:
        if m.get("date"):
            dates.add(m["date"])
    
    today = datetime.now(HK_TZ).strftime("%Y-%m-%d")
    dates.add(today)  # Always include today

    sorted_dates = sorted(dates)
    now = datetime.now(HK_TZ)
    
    date_infos = []
    for d in sorted_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        days_away = (dt.date() - now.date()).days
        # Distance confidence decay
        if days_away <= 0:
            confidence = "高" if now.hour >= 10 else "中"
            conf_value = 0.9 if now.hour >= 10 else 0.7
        elif days_away == 1:
            confidence = "中高"
            conf_value = 0.65
        elif days_away == 2:
            confidence = "中"
            conf_value = 0.50
        elif days_away == 3:
            confidence = "中低"
            conf_value = 0.35
        else:
            confidence = "低"
            conf_value = max(0.15, 0.35 - 0.05 * (days_away - 3))

        market_count = sum(1 for m in markets if m.get("date") == d)
        date_infos.append({
            "date": d,
            "days_away": days_away,
            "confidence": confidence,
            "confidence_value": round(conf_value, 2),
            "market_count": market_count,
            "label": format_date_label(d, days_away),
        })

    return {"dates": date_infos, "markets": markets}


def format_date_label(date_str, days_away):
    """Format date for display."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    weekdays = ["周一","周二","周三","周四","周五","周六","周日"]
    wd = weekdays[dt.weekday()]
    m = dt.month
    d = dt.day
    if days_away == 0:
        return f"今天 ({m}月{d}日 {wd})"
    elif days_away == 1:
        return f"明天 ({m}月{d}日 {wd})"
    elif days_away == 2:
        return f"后天 ({m}月{d}日 {wd})"
    elif days_away < 0:
        return f"已过 ({m}月{d}日 {wd})"
    else:
        return f"{m}月{d}日 {wd} (+{days_away}天)"


# ──────────────────────────────────────────────
# Forecast fetchers
# ──────────────────────────────────────────────

def fetch_jma(target_date):
    try:
        r = requests.get("https://api.open-meteo.com/v1/jma",
            params={"latitude": HK_LAT, "longitude": HK_LON,
                    "hourly": "temperature_2m,cloudcover", "forecast_days": 7,
                    "timezone": "Asia/Hong_Kong"}, timeout=15)
        data = r.json()
        temps = [v for t, v in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])
                 if v is not None and t.startswith(target_date)]
        # Extract daytime cloud cover (6am-18pm) for DTR adjustment
        cloud_vals = []
        for t, c in zip(data["hourly"]["time"], data["hourly"].get("cloudcover", [])):
            if t.startswith(target_date) and c is not None:
                hr = int(t[11:13])
                if 6 <= hr <= 18:
                    cloud_vals.append(c)
        cloud_pct = round(np.mean(cloud_vals), 1) if cloud_vals else None
        if temps:
            return {"model": "JMA", "max": round(max(temps),1), "min": round(min(temps),1),
                    "hourly": temps, "cloud_pct": cloud_pct}
        return None
    except Exception:
        return None

def fetch_ecmwf_ensemble(target_date):
    """Fetch ECMWF IFS 50-member ensemble (expert fix: no more assumed Gaussian)."""
    try:
        r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble",
            params={"latitude": HK_LAT, "longitude": HK_LON,
                    "hourly": "temperature_2m", "models": "ecmwf_ifs025",
                    "forecast_days": 7, "timezone": "Asia/Hong_Kong"}, timeout=20)
        data = r.json()
        if "error" in data:
            return _fetch_ecmwf_deterministic(target_date)  # fallback
        member_keys = [k for k in data["hourly"] if k.startswith("temperature_2m_member")]
        date_idx = [i for i, t in enumerate(data["hourly"]["time"]) if t.startswith(target_date)]
        if not date_idx or not member_keys:
            return _fetch_ecmwf_deterministic(target_date)
        maxes = []
        hourly_avgs = None
        for k in member_keys:
            vals = [data["hourly"][k][i] for i in date_idx if data["hourly"][k][i] is not None]
            if vals:
                maxes.append(round(max(vals), 1))
        if not maxes:
            return _fetch_ecmwf_deterministic(target_date)
        # Compute average hourly across members for chart
        n_hours = len(date_idx)
        hourly_avgs = []
        for idx in date_idx:
            member_vals = [data["hourly"][k][idx] for k in member_keys
                           if data["hourly"][k][idx] is not None]
            hourly_avgs.append(round(np.mean(member_vals), 1) if member_vals else None)
        maxes.sort()
        return {"model": "ECMWF", "members": len(maxes), "maxes": maxes,
                "mean": round(float(np.mean(maxes)),1), "std": round(float(np.std(maxes)),1),
                "min": round(min(maxes),1), "max": round(max(maxes),1),
                "hourly": hourly_avgs,
                "is_ensemble": True}
    except Exception:
        return _fetch_ecmwf_deterministic(target_date)

def _fetch_ecmwf_deterministic(target_date):
    """Fallback: deterministic ECMWF if ensemble unavailable."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": HK_LAT, "longitude": HK_LON,
                    "hourly": "temperature_2m", "models": "ecmwf_ifs025",
                    "forecast_days": 7, "timezone": "Asia/Hong_Kong"}, timeout=15)
        data = r.json()
        temps = [v for t, v in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])
                 if v is not None and t.startswith(target_date)]
        if temps:
            return {"model": "ECMWF", "max": round(max(temps),1), "min": round(min(temps),1),
                    "hourly": temps, "is_ensemble": False}
        return None
    except Exception:
        return None

def fetch_gfs_ensemble(target_date):
    try:
        r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble",
            params={"latitude": HK_LAT, "longitude": HK_LON,
                    "hourly": "temperature_2m", "models": "gfs_seamless",
                    "forecast_days": 7, "timezone": "Asia/Hong_Kong"}, timeout=15)
        data = r.json()
        member_keys = [k for k in data["hourly"] if k.startswith("temperature_2m_member")]
        date_idx = [i for i, t in enumerate(data["hourly"]["time"]) if t.startswith(target_date)]
        if not date_idx:
            return None
        maxes = []
        for k in member_keys:
            vals = [data["hourly"][k][i] for i in date_idx if data["hourly"][k][i] is not None]
            if vals:
                maxes.append(round(max(vals), 1))
        maxes.sort()
        return {"model": "GFS", "members": len(maxes), "maxes": maxes,
                "mean": round(float(np.mean(maxes)),1), "std": round(float(np.std(maxes)),1),
                "min": round(min(maxes),1), "max": round(max(maxes),1)} if maxes else None
    except:
        return None

def fetch_hko_realtime():
    try:
        r = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php",
            params={"dataType": "rhrread", "lang": "tc"}, timeout=10)
        data = r.json()
        hko_temp = None
        all_temps = {}
        for t in data.get("temperature", {}).get("data", []):
            all_temps[t["place"]] = t["value"]
            if "天文台" in t["place"] or t["place"] == "Hong Kong Observatory":
                hko_temp = t["value"]
        # Fallback
        if hko_temp is None:
            r2 = requests.get("https://data.weather.gov.hk/weatherAPI/opendata/weather.php",
                params={"dataType": "rhrread", "lang": "en"}, timeout=10)
            data2 = r2.json()
            for t in data2.get("temperature", {}).get("data", []):
                if t["place"] == "Hong Kong Observatory":
                    hko_temp = t["value"]
                all_temps[t["place"]] = t["value"]
        
        return {
            "hko_temp": hko_temp,
            "all_temps": all_temps,
            "update_time": data.get("updateTime", ""),
            "humidity": data.get("humidity", {}).get("data", [{}])[0].get("value"),
        }
    except:
        return None


# ──────────────────────────────────────────────
# Bucket probability engine
# ──────────────────────────────────────────────

def compute_bucket_probs(jma, ecmwf, gfs, hko_obs, current_hour, days_away=0):
    """Compute temperature bucket probabilities.
    Expert-reviewed v3: climatological prior, cloud-adjusted DTR,
    midnight-max trap, ECMWF ensemble counting, UHI offset."""
    from scipy.stats import norm

    # ── Dynamic UHI offset based on cloud cover ──
    cloud_pct = jma.get("cloud_pct") if jma else None
    if cloud_pct is not None:
        # Interpolate: 0% cloud → +0.5, 100% cloud → +0.2
        uhi = UHI_OFFSET_CLEAR - (cloud_pct / 100.0) * (UHI_OFFSET_CLEAR - UHI_OFFSET_CLOUDY)
    else:
        uhi = 0.3  # conservative default

    forecast_maxes = []
    if jma: forecast_maxes.append(jma["max"] + uhi)
    if ecmwf:
        if ecmwf.get("is_ensemble"):
            forecast_maxes.append(ecmwf["mean"] + uhi)
        else:
            forecast_maxes.append(ecmwf["max"] + uhi)
    if gfs: forecast_maxes.append(gfs["mean"] + uhi)
    if not forecast_maxes:
        return {}, 0.0

    center = np.mean(forecast_maxes)
    spread = 5 + min(days_away, 3)
    buckets = list(range(int(center) - spread, int(center) + spread + 1))
    probs = {b: 0.0 for b in buckets}

    sigma_boost = 1.0 + 0.3 * min(days_away, 5)

    # ── Midnight max trap detection ──
    # If model hourly data shows temperature DECREASING through the day
    # (cold front passage), the daily max is at midnight, not 14:00.
    midnight_max_detected = False
    hourly_source = jma.get("hourly") if jma else (ecmwf.get("hourly") if ecmwf else None)
    if hourly_source and len(hourly_source) >= 15:
        t_midnight = hourly_source[0]  # 00:00
        t_noon = hourly_source[12] if len(hourly_source) > 12 else None
        t_peak = hourly_source[14] if len(hourly_source) > 14 else None
        if t_midnight is not None and t_noon is not None:
            if t_midnight > t_noon + 0.5:  # midnight hotter than noon by 0.5+
                midnight_max_detected = True

    # ── JMA (weight 35%): Gaussian from max ──
    jma_max_adj = (jma["max"] + uhi) if jma else None
    if jma:
        for b in buckets:
            probs[b] += 0.35 * (norm.cdf(b+1, jma_max_adj, 1.5*sigma_boost) - norm.cdf(b, jma_max_adj, 1.5*sigma_boost))

    # ── ECMWF (weight 30%): ensemble counting if available, else Gaussian ──
    if ecmwf and ecmwf.get("is_ensemble") and ecmwf.get("maxes"):
        n = len(ecmwf["maxes"])
        maxes_adj = [m + uhi for m in ecmwf["maxes"]]
        for b in buckets:
            count = sum(1 for m in maxes_adj if b <= m < b+1)
            probs[b] += 0.30 * (count / n)
    elif ecmwf:
        ecmwf_max_adj = (ecmwf["max"] + uhi)
        for b in buckets:
            probs[b] += 0.30 * (norm.cdf(b+1, ecmwf_max_adj, 1.3*sigma_boost) - norm.cdf(b, ecmwf_max_adj, 1.3*sigma_boost))

    # ── GFS (weight 20%): member counting ──
    if gfs:
        n = len(gfs["maxes"])
        maxes_adj = [m + uhi for m in gfs["maxes"]]
        for b in buckets:
            count = sum(1 for m in maxes_adj if b <= m < b+1)
            probs[b] += 0.20 * (count / n)

    # ── Climatological prior (15%): monthly Gaussian instead of uniform ──
    # Expert fix: uniform distribution gave false edge on extreme buckets.
    # Use historical monthly distribution as informative Bayesian prior.
    params_path = os.path.join(MODELS_DIR, "extrapolation_params.json")
    month_str = str(datetime.now(HK_TZ).month)
    # March HK: historical max ~23°C, σ ~3.5°C (wide enough to not constrain)
    clim_center = center  # fallback
    clim_sigma = 3.5
    # Load monthly mean from data if available
    if os.path.exists(params_path):
        try:
            with open(params_path) as f:
                ep = json.load(f)
        except Exception:
            ep = {}
    else:
        ep = {}
    # Use model center as climatological center (better than fixed)
    for b in buckets:
        clim_p = norm.cdf(b+1, clim_center, clim_sigma) - norm.cdf(b, clim_center, clim_sigma)
        probs[b] += 0.15 * clim_p

    # ── Kalman correction (only for today with real-time data) ──
    kalman_k = 0.0
    if days_away == 0 and hko_obs and hko_obs.get("hko_temp") is not None and current_hour and current_hour >= 6:
        K_sched = {6:.20, 7:.30, 8:.40, 9:.55, 10:.65, 11:.75,
                   12:.85, 13:.90, 14:.95, 15:.97, 16:.98, 17:1.0}
        K = K_sched.get(current_hour, 0.05 if current_hour < 6 else 1.0)
        kalman_k = K
        temp = hko_obs["hko_temp"]

        # Load monthly mean rise
        mean_rise = 4.5
        if month_str in ep.get("monthly_rise", {}):
            mean_rise = ep["monthly_rise"][month_str]["mean_rise"]

        # ── Cloud cover DTR adjustment (expert fix) ──
        # Overcast days have much lower diurnal temperature range.
        cloud_pct = jma.get("cloud_pct") if jma else None
        if cloud_pct is not None:
            # 100% cloud → reduce DTR by 40%; 0% cloud → full DTR
            cloud_factor = (cloud_pct / 100.0) * 0.4
            mean_rise = mean_rise * (1.0 - cloud_factor)

        # ── Midnight max trap (expert fix) ──
        if midnight_max_detected:
            # Cold front: max likely at midnight, not afternoon.
            # Use hourly max directly instead of gain-to-peak extrapolation.
            if hourly_source:
                hourly_max = max(v for v in hourly_source if v is not None)
                est_max = hourly_max + uhi
            else:
                est_max = temp
            sigma = 0.8  # fairly certain
        else:
            # Normal day: gain-to-peak extrapolation
            gain_to_peak = {
                6: (mean_rise * 1.0, 2.0),
                7: (mean_rise * 0.93, 1.8),
                8: (mean_rise * 0.85, 1.6),
                9: (mean_rise * 0.78, 1.5),
                10: (mean_rise * 0.70, 1.4),
                11: (mean_rise * 0.58, 1.2),
                12: (mean_rise * 0.40, 1.0),
                13: (mean_rise * 0.18, 0.7),
                14: (0.2, 0.5),
                15: (0.0, 0.3),
            }
            if current_hour >= 16:
                est_max = temp
                sigma = 0.2
            elif current_hour in gain_to_peak:
                gain, sigma = gain_to_peak[current_hour]
                est_max = temp + gain
            else:
                est_max = temp + mean_rise * 0.5
                sigma = 1.5

        # Don't let est_max fall below model average
        model_avg = center
        est_max = max(est_max, model_avg)

        # Dynamic K boost on divergence
        divergence = abs(est_max - model_avg)
        if divergence > 1.0:
            k_boost = min(0.25, (divergence - 1.0) * 0.15)
            K = min(0.98, K + k_boost)
            kalman_k = K

        obs_probs = {}
        for b in buckets:
            obs_probs[b] = norm.cdf(b+1, est_max, sigma) - norm.cdf(b, est_max, sigma)
        for b in buckets:
            probs[b] = (1-K)*probs[b] + K*obs_probs.get(b, 0)

    total = sum(probs.values())
    if total > 0:
        probs = {b: round(p/total, 4) for b, p in probs.items()}

    probs = {b: p for b, p in probs.items() if p >= 0.005}
    return probs, kalman_k, midnight_max_detected, uhi


# ──────────────────────────────────────────────
# JSON prediction (non-streaming, reliable)
# ──────────────────────────────────────────────

def run_prediction_json(target_date, all_markets):
    """Run prediction and return complete JSON result."""
    now = datetime.now(HK_TZ)
    is_today = target_date == now.strftime("%Y-%m-%d")
    current_hour = now.hour if is_today else None
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    days_away = (dt.date() - now.date()).days

    steps = []
    def log(step_id, status, label, detail=""):
        steps.append({"step": step_id, "status": status, "label": label, "detail": detail})

    log(1, "done", "初始化分析引擎",
        f"目标日期：{target_date} | 模式：{'实时修正' if is_today else '纯预报'}")

    jma = fetch_jma(target_date)
    log(2, "done" if jma else "error", "拉取 JMA 预报",
        (f"最高 {jma['max']}°C  云量 {jma.get('cloud_pct','?')}%" if jma else "获取失败"))

    ecmwf = fetch_ecmwf_ensemble(target_date)
    if ecmwf and ecmwf.get("is_ensemble"):
        log(3, "done", "拉取 ECMWF 集合预报",
            f"{ecmwf['members']}成员 均值 {ecmwf['mean']}°C ±{ecmwf['std']}°C")
    elif ecmwf:
        log(3, "done", "拉取 ECMWF 预报（决定性）",
            f"最高 {ecmwf['max']}°C")
    else:
        log(3, "error", "ECMWF 预报获取失败", "")

    gfs = fetch_gfs_ensemble(target_date)
    log(4, "done" if gfs else "error", "拉取 GFS 集合预报",
        f"{gfs['members']}成员 均值 {gfs['mean']}°C ±{gfs['std']}°C" if gfs else "获取失败")

    hko = None
    if is_today:
        hko = fetch_hko_realtime()
        log(5, "done" if (hko and hko.get("hko_temp")) else "warn",
            "拉取天文台实时数据",
            f"当前 {hko['hko_temp']}°C 湿度 {hko.get('humidity','-')}%" if hko and hko.get("hko_temp") else "暂不可用")
    else:
        log(5, "skip", "跳过实时数据", f"非今天（距今 {days_away} 天）")

    midnight_max = False
    uhi = 0.3
    try:
        probs, kalman_k, midnight_max, uhi = compute_bucket_probs(jma, ecmwf, gfs, hko, current_hour, days_away)
    except Exception:
        probs, kalman_k = {}, 0.0

    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top = sorted_p[0] if sorted_p else (0, 0)
    log(6, "done" if probs else "warn", "计算档位概率",
        f"最可能：{top[0]}°C ({top[1]*100:.1f}%)" if probs else "无预报数据")

    # Confidence
    if days_away <= 0:
        base_conf = 0.9 if (current_hour and current_hour >= 10) else 0.7
    elif days_away == 1:
        base_conf = 0.65
    elif days_away == 2:
        base_conf = 0.50
    elif days_away == 3:
        base_conf = 0.35
    else:
        base_conf = max(0.15, 0.35 - 0.05 * (days_away - 3))
    concentration = top[1] if top[1] else 0
    conf = min(0.95, base_conf * (0.7 + 0.3 * concentration / 0.5)) if concentration else base_conf * 0.5
    if conf >= 0.7: conf_label = "高"
    elif conf >= 0.5: conf_label = "中高"
    elif conf >= 0.35: conf_label = "中"
    elif conf >= 0.2: conf_label = "中低"
    else: conf_label = "低"

    reasoning = []
    if days_away == 0:
        reasoning.append("今天的预测" + ("，有实时数据支撑" if current_hour and current_hour>=8 else ""))
    elif days_away == 1:
        reasoning.append("明天的预测，准确度较高")
    elif days_away <= 3:
        reasoning.append(f"提前 {days_away} 天，准确度中等")
    else:
        reasoning.append(f"提前 {days_away} 天，不确定性较大")
    log(7, "done", f"置信度：{conf_label} ({conf*100:.0f}%)", " | ".join(reasoning))

    # ── Model consistency check ──
    model_maxes = []
    if jma: model_maxes.append(("JMA", jma["max"]))
    if ecmwf: model_maxes.append(("ECMWF", ecmwf["max"]))
    if gfs: model_maxes.append(("GFS", gfs["mean"]))
    model_spread = max(v for _,v in model_maxes) - min(v for _,v in model_maxes) if len(model_maxes) >= 2 else 0
    if model_spread <= 1.0:
        model_agree = "高"
        agree_icon = "✅"
    elif model_spread <= 2.0:
        model_agree = "中"
        agree_icon = "⚠️"
    else:
        model_agree = "低"
        agree_icon = "❌"
        conf *= 0.8  # Penalize confidence when models disagree
        if conf >= 0.7: conf_label = "高"
        elif conf >= 0.5: conf_label = "中高"
        elif conf >= 0.35: conf_label = "中"
        elif conf >= 0.2: conf_label = "中低"
        else: conf_label = "低"

    log(7, "done", f"置信度：{conf_label} ({conf*100:.0f}%)",
        " | ".join(reasoning) + f" | 模型一致性{agree_icon}{model_agree}（偏差{model_spread:.1f}°C）")

    # ── Edge + Liquidity ──
    date_markets = [m for m in all_markets if m.get("date") == target_date]
    edges = {}
    total_liquidity = 0
    for m in date_markets:
        b = m.get("bucket")
        total_liquidity += m.get("liquidity", 0)
        if b is not None and b in probs:
            our_p = probs[b]
            market_p = m["yes_price"]
            if market_p > 0:
                raw_edge = our_p - market_p
                # Expert fix: 2% slippage buffer for thin orderbooks
                edge = raw_edge - 0.02
                kelly = 0
                if edge > 0:
                    odds = (1/market_p) - 1
                    kelly = 0.15 * (edge / odds) if odds > 0 else 0
                liq = m.get("liquidity", 0)
                vol = m.get("volume", 0)
                liq_warn = ""
                if liq < 100:
                    liq_warn = "极低"
                elif liq < 200:
                    liq_warn = "偏低"
                elif liq < 500:
                    liq_warn = "一般"
                else:
                    liq_warn = "充足"
                edges[str(b)] = {
                    "bucket": b, "our_prob": our_p, "market_prob": market_p,
                    "edge": round(edge, 4), "raw_edge": round(raw_edge, 4),
                    "kelly": round(kelly, 4),
                    "signal": "买入" if edge >= 0.06 else ("观望" if edge >= 0.02 else "跳过"),
                    "liquidity": round(liq), "volume": round(vol),
                    "liq_level": liq_warn,
                }
    n_buy = sum(1 for e in edges.values() if e["signal"] == "买入")
    liq_ok = total_liquidity > 2000
    log(8, "done" if date_markets else "warn",
        f"Polymarket：{len(date_markets)} 个市场，{n_buy} 个买入信号" if date_markets else "未找到市场",
        f"总流动性 ${total_liquidity:,.0f}" + ("" if liq_ok else " ⚠️偏低") if date_markets else "")

    # ── Strategy B: Trading Recommendations ──
    DEFAULT_CAPITAL = 10
    now_h = now.hour

    # Entry/exit timing
    if days_away > 0:
        entry_window = "今晚 21:00 - 明天 08:00（夜间价差最大）"
        exit_window = f"{'明天' if days_away==1 else f'{days_away}天后'} 09:00 - 12:00（实测数据推动价格）"
        urgency = "🟢 充裕" if days_away >= 2 else "🟡 建议今晚入场"
    elif is_today:
        if now_h < 9:
            entry_window = "现在 - 09:00（模型已更新，抢先入场）"
            exit_window = "09:00 - 12:00（实测数据进来后卖出）"
            urgency = "🟡 尽快入场"
        elif now_h < 12:
            entry_window = "现在（实测数据已有，精度高）"
            exit_window = "12:00 - 14:00（峰值确认后卖出）"
            urgency = "🟢 当前是最佳窗口"
        elif now_h < 15:
            entry_window = "现在（峰值接近确定）"
            exit_window = "持仓到结算"
            urgency = "🔴 峰值已近，Edge可能已被消化"
        else:
            entry_window = "不建议入场"
            exit_window = "—"
            urgency = "🔴 峰值已过"
    else:
        entry_window = "—"
        exit_window = "—"
        urgency = "⚪ 已过期"

    # Per-bucket recommendations — sorted by OUR probability (highest first)
    # Include any bucket where we have positive edge (our prob > market prob)
    recommendations = []
    # Expert fix: edge threshold now accounts for 2% slippage buffer already applied
    positive_edges = {k:v for k,v in edges.items() if v["edge"] > 0.01}  # > 1% net edge
    sorted_edges = sorted(positive_edges.items(), key=lambda x: x[1]["our_prob"], reverse=True)

    # Expert fix: mutually exclusive Kelly cap.
    # Weather buckets are mutually exclusive — only one can win.
    # Total exposure capped at 20% of capital. Primary gets 60%, rest shares 40%.
    MAX_TOTAL_EXPOSURE = 0.20  # 20% of capital max
    total_budget = DEFAULT_CAPITAL * MAX_TOTAL_EXPOSURE

    for idx, (k, e) in enumerate(sorted_edges):
        entry_p = e["market_prob"]
        tp_mult = min(4.0, max(2.0, 0.30 / entry_p)) if entry_p > 0 else 3.0
        take_profit = min(0.85, round(entry_p * tp_mult, 2))
        stop_loss = max(0.01, round(entry_p * 0.55, 2))

        # Position sizing: mutually exclusive allocation
        if idx == 0:
            position_amt = round(total_budget * 0.60, 2)  # Primary: 60% of budget
        elif idx == 1:
            position_amt = round(total_budget * 0.25, 2)  # Secondary: 25%
        else:
            position_amt = round(total_budget * 0.15, 2)  # Tertiary: 15%
        if position_amt < 0.5:
            position_amt = 0.5
        shares = int(position_amt / entry_p) if entry_p > 0 else 0
        max_loss = round(shares * (entry_p - stop_loss), 2)
        expected_profit = round(shares * (take_profit - entry_p) * e["our_prob"] - shares * (entry_p - stop_loss) * (1 - e["our_prob"]), 2)
        rr = round((take_profit - entry_p) / (entry_p - stop_loss), 1) if entry_p > stop_loss else 0

        # Liquidity check
        can_trade = e["liquidity"] >= 100
        liq_note = ""
        if e["liquidity"] < 100:
            liq_note = "⛔ 流动性极低，不建议交易"
        elif e["liquidity"] < 200:
            liq_note = "⚠️ 流动性偏低，减半仓位"
            position_amt = round(position_amt * 0.5, 2)
            shares = int(position_amt / entry_p) if entry_p > 0 else 0

        recommendations.append({
            "bucket": e["bucket"],
            "our_prob": e["our_prob"],
            "edge": e["edge"],
            "entry_price": entry_p,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "tp_pct": f"+{(tp_mult-1)*100:.0f}%",
            "sl_pct": "-45%",
            "position_amt": position_amt,
            "shares": shares,
            "risk_reward": f"1:{rr}",
            "max_loss": max_loss,
            "expected_profit": expected_profit,
            "liquidity": e["liquidity"],
            "liq_note": liq_note,
            "can_trade": can_trade,
        })

    total_position = sum(r["position_amt"] for r in recommendations)
    total_risk = sum(r["max_loss"] for r in recommendations)
    total_expected = sum(r["expected_profit"] for r in recommendations)

    strategy = {
        "entry_window": entry_window,
        "exit_window": exit_window,
        "urgency": urgency,
        "capital": DEFAULT_CAPITAL,
        "recommendations": recommendations,
        "total_position": round(total_position, 2),
        "total_risk": round(total_risk, 2),
        "total_expected": round(total_expected, 2),
        "total_liquidity": round(total_liquidity),
        "liquidity_ok": liq_ok,
    }

    forecasts = {
        "jma": {"max":jma["max"],"min":jma["min"],"hourly":jma.get("hourly",[]),
                "cloud_pct":jma.get("cloud_pct")} if jma else None,
        "ecmwf": None,
        "gfs": {"mean":gfs["mean"],"std":gfs["std"],"min":gfs["min"],"max":gfs["max"],"members":gfs["members"]} if gfs else None,
    }
    if ecmwf:
        if ecmwf.get("is_ensemble"):
            forecasts["ecmwf"] = {"mean":ecmwf["mean"],"std":ecmwf["std"],
                "min":ecmwf["min"],"max":ecmwf["max"],"members":ecmwf["members"],
                "hourly":ecmwf.get("hourly",[]), "is_ensemble": True}
        else:
            forecasts["ecmwf"] = {"max":ecmwf["max"],"min":ecmwf.get("min",ecmwf["max"]),
                "hourly":ecmwf.get("hourly",[]), "is_ensemble": False}

    return {
        "target_date": target_date, "days_away": days_away, "is_today": is_today,
        "current_hour": current_hour, "forecasts": forecasts, "hko": hko,
        "probabilities": {str(k):v for k,v in probs.items()},
        "kalman_gain": kalman_k, "confidence": conf, "confidence_label": conf_label,
        "midnight_max": midnight_max, "uhi_offset": uhi,
        "edges": edges, "top_bucket": top[0], "top_prob": top[1], "steps": steps,
        "strategy": strategy,
        "model_consistency": {"spread": model_spread, "level": model_agree, "icon": agree_icon},
    }


# ──────────────────────────────────────────────
# Streaming prediction  (step-by-step SSE)
# ──────────────────────────────────────────────

def predict_streaming(handler, target_date, all_markets):
    """Send step-by-step analysis via Server-Sent Events."""
    now = datetime.now(HK_TZ)
    is_today = target_date == now.strftime("%Y-%m-%d")
    current_hour = now.hour if is_today else None
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    days_away = (dt.date() - now.date()).days

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    def send(step_id, status, label, detail=None, data=None):
        msg = {"step": step_id, "status": status, "label": label}
        if detail: msg["detail"] = detail
        if data: msg["data"] = data
        handler.wfile.write(f"data: {json.dumps(msg, ensure_ascii=False)}\n\n".encode())
        handler.wfile.flush()

    # Step 1: Init
    try:
        send(1, "running", "初始化分析引擎",
             f"目标日期：{target_date}，距今 {days_away} 天")
        time.sleep(0.3)
        send(1, "done", "初始化分析引擎",
             f"模式：{'实时修正' if is_today else '纯预报'}",
             {"target_date": target_date, "is_today": is_today, "days_away": days_away})

        # Step 2: JMA
        send(2, "running", "拉取日本气象厅 (JMA) 预报")
        jma = fetch_jma(target_date)
        if jma:
            send(2, "done", "JMA 预报获取成功",
                 f"预测最高 {jma['max']}°C，最低 {jma['min']}°C",
                 {"jma": jma})
        else:
            send(2, "error", "JMA 预报获取失败")

        # Step 3: ECMWF
        send(3, "running", "拉取欧洲中期预报 (ECMWF)")
        ecmwf = fetch_ecmwf_ensemble(target_date)
        if ecmwf and ecmwf.get("is_ensemble"):
            send(3, "done", "ECMWF 50成员集合预报",
                 f"均值 {ecmwf['mean']}°C ±{ecmwf['std']}°C",
                 {"ecmwf": ecmwf})
        elif ecmwf:
            send(3, "done", "ECMWF 预报获取成功",
                 f"预测最高 {ecmwf['max']}°C",
                 {"ecmwf": ecmwf})
        else:
            send(3, "error", "ECMWF 预报获取失败")

        # Step 4: GFS ensemble
        send(4, "running", "拉取美国 GFS 集合预报 (30个成员)")
        gfs = fetch_gfs_ensemble(target_date)
        if gfs:
            send(4, "done", "GFS 集合预报获取成功",
                 f"{gfs['members']} 个模型成员，均值 {gfs['mean']}°C ± {gfs['std']}°C（范围 {gfs['min']}-{gfs['max']}°C）",
                 {"gfs": {"mean":gfs["mean"],"std":gfs["std"],"min":gfs["min"],"max":gfs["max"],"members":gfs["members"]}})
        else:
            send(4, "error", "GFS 集合预报获取失败")

        # Step 5: HKO real-time (only today)
        hko = None
        if is_today:
            send(5, "running", "拉取香港天文台实时数据")
            hko = fetch_hko_realtime()
            if hko and hko.get("hko_temp") is not None:
                send(5, "done", "天文台实时数据获取成功",
                     f"当前温度 {hko['hko_temp']}°C，湿度 {hko.get('humidity','-')}%",
                     {"hko": hko})
            else:
                send(5, "warn", "天文台数据暂不可用", "将仅使用预报模型")
        else:
            send(5, "skip", "跳过实时数据", f"非今天（距今 {days_away} 天），无实时修正")

        # Step 6: Compute probabilities
        send(6, "running", "计算档位概率分布",
             "融合 JMA(35%) + ECMWF(30%) + GFS(20%) + 历史(15%)")
        try:
            probs, kalman_k = compute_bucket_probs(jma, ecmwf, gfs, hko, current_hour, days_away)
        except Exception as ex:
            probs, kalman_k = {}, 0.0
            send(6, "error", "概率计算失败", str(ex))

        if probs:
            sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top = sorted_p[0] if sorted_p else (0, 0)
            send(6, "done", "概率计算完成",
                 f"最可能：{top[0]}°C ({top[1]*100:.1f}%)" + (f" | 卡尔曼增益 K={kalman_k:.2f}" if kalman_k > 0 else ""),
                 {"probabilities": probs, "kalman_gain": kalman_k})
        else:
            sorted_p = []
            top = (0, 0)
            send(6, "warn", "无法计算概率", "所有气象模型均未返回预报数据")

        # Step 7: Confidence assessment
        send(7, "running", "评估预测置信度")
        # Distance-based confidence
        if days_away <= 0:
            base_conf = 0.9 if (current_hour and current_hour >= 10) else 0.7
        elif days_away == 1:
            base_conf = 0.65
        elif days_away == 2:
            base_conf = 0.50
        elif days_away == 3:
            base_conf = 0.35
        else:
            base_conf = max(0.15, 0.35 - 0.05 * (days_away - 3))
        
        concentration = top[1] if top[1] else 0
        conf = min(0.95, base_conf * (0.7 + 0.3 * concentration / 0.5)) if concentration else base_conf * 0.5
        
        if conf >= 0.7: conf_label = "高"
        elif conf >= 0.5: conf_label = "中高"
        elif conf >= 0.35: conf_label = "中"
        elif conf >= 0.2: conf_label = "中低"
        else: conf_label = "低"

        reasoning = []
        if days_away == 0:
            reasoning.append("今天的预测，有实时数据支撑" if current_hour and current_hour >= 8 else "今天的预测，但尚无实时修正")
        elif days_away == 1:
            reasoning.append("明天的预测，1天内预报准确度较高")
        elif days_away <= 3:
            reasoning.append(f"提前 {days_away} 天预测，准确度中等")
        else:
            reasoning.append(f"提前 {days_away} 天预测，不确定性较大")
        
        if concentration >= 0.4:
            reasoning.append(f"概率集中度高（{top[0]}°C 占 {concentration*100:.0f}%）")
        elif concentration >= 0.25:
            reasoning.append("概率分散在多个档位，不确定性中等")
        else:
            reasoning.append("概率非常分散，难以确定目标档位")
        
        send(7, "done", f"置信度：{conf_label} ({conf*100:.0f}%)",
             " | ".join(reasoning),
             {"confidence": conf, "confidence_label": conf_label, "reasoning": reasoning})

        # Step 8: Polymarket edge
        send(8, "running", "对比 Polymarket 市场价格")
        date_markets = [m for m in all_markets if m.get("date") == target_date]
        edges = {}
        if date_markets:
            for m in date_markets:
                b = m.get("bucket")
                if b is not None and b in probs:
                    our_p = probs[b]
                    market_p = m["yes_price"]
                    if market_p > 0:
                        edge = our_p - market_p
                        kelly = 0
                        if edge > 0:
                            odds = (1/market_p) - 1
                            kelly = 0.15 * (edge / odds) if odds > 0 else 0
                        edges[str(b)] = {
                            "bucket": b, "our_prob": our_p, "market_prob": market_p,
                            "edge": round(edge, 4), "kelly": round(kelly, 4),
                            "signal": "买入" if edge >= 0.08 else ("观望" if edge >= 0.04 else "跳过"),
                        }
            n_buy = sum(1 for e in edges.values() if e["signal"] == "买入")
            send(8, "done", f"找到 {len(date_markets)} 个市场，{n_buy} 个买入机会",
                 data={"edges": edges, "market_count": len(date_markets)})
        else:
            send(8, "warn", "未找到该日期的 Polymarket 市场",
                 "Polymarket 可能尚未开放此日期，或搜索无结果")

        # Step 9: Final
        forecasts = {
            "jma": {"max":jma["max"],"min":jma["min"],"hourly":jma.get("hourly",[])} if jma else None,
            "ecmwf": {"max":ecmwf["max"],"min":ecmwf["min"],"hourly":ecmwf.get("hourly",[])} if ecmwf else None,
            "gfs": {"mean":gfs["mean"],"std":gfs["std"],"min":gfs["min"],"max":gfs["max"],"members":gfs["members"]} if gfs else None,
        }
        final = {
            "target_date": target_date,
            "days_away": days_away,
            "is_today": is_today,
            "current_hour": current_hour,
            "forecasts": forecasts,
            "hko": hko,
            "probabilities": probs,
            "kalman_gain": kalman_k,
            "confidence": conf,
            "confidence_label": conf_label,
            "edges": edges,
            "top_bucket": top[0],
            "top_prob": top[1],
        }
        send(9, "complete", "分析完成", data=final)

    except BrokenPipeError:
        pass  # Client disconnected, that's OK
    except Exception as ex:
        import traceback
        traceback.print_exc()
        try:
            send(9, "complete", "分析出错",
                 detail=f"错误：{str(ex)}",
                 data={"error": str(ex), "probabilities": {}, "edges": {},
                       "forecasts": {}, "confidence": 0, "confidence_label": "低",
                       "target_date": target_date, "days_away": days_away,
                       "is_today": is_today, "current_hour": current_hour,
                       "kalman_gain": 0, "top_bucket": 0, "top_prob": 0, "hko": None})
        except:
            pass


# ──────────────────────────────────────────────
# HTTP Server
# ──────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):
    _cached_markets = None
    _cache_time = 0

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(BASE_DIR, "src", "dashboard.html"), "rb") as f:
                self.wfile.write(f.read())

        elif parsed.path == "/api/dates":
            # Cache market scan for 60s
            now_ts = time.time()
            if Handler._cached_markets is None or now_ts - Handler._cache_time > 60:
                Handler._cached_markets = get_available_dates()
                Handler._cache_time = now_ts
            self._json_response(Handler._cached_markets)

        elif parsed.path == "/api/predict":
            params = urllib.parse.parse_qs(parsed.query)
            target = params.get("date", [datetime.now(HK_TZ).strftime("%Y-%m-%d")])[0]
            if Handler._cached_markets and "markets" in Handler._cached_markets:
                markets = Handler._cached_markets["markets"]
            else:
                markets = search_polymarket_hk_weather()
                if isinstance(markets, dict):
                    markets = []
            result = run_prediction_json(target, markets)
            self._json_response(result)

        elif parsed.path == "/api/predict-stream":
            params = urllib.parse.parse_qs(parsed.query)
            target = params.get("date", [datetime.now(HK_TZ).strftime("%Y-%m-%d")])[0]
            if Handler._cached_markets and "markets" in Handler._cached_markets:
                markets = Handler._cached_markets["markets"]
            else:
                markets = search_polymarket_hk_weather()
                if isinstance(markets, dict):
                    markets = []
            predict_streaming(self, target, markets)

        elif parsed.path == "/api/hko":
            self._json_response(fetch_hko_realtime() or {})

        else:
            super().do_GET()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        print(f"  [{datetime.now(HK_TZ).strftime('%H:%M:%S')}] {args[0]}")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    print("=" * 55)
    print("🇭🇰 香港天气预测系统")
    print(f"   打开浏览器：http://localhost:{PORT}")
    print("=" * 55)
    os.chdir(BASE_DIR)
    server = ThreadedHTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
        server.server_close()
