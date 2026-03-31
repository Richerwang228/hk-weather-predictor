"""
Step 1: Download historical daily max temperature data from Hong Kong Observatory.

Data source: HKO Open Data API
- CLMMAXT: Daily Maximum Temperature
- Station: HKO (Hong Kong Observatory headquarters, Tsim Sha Tsui)

This is the EXACT data source Polymarket uses for settlement:
"Absolute Daily Max (deg. C)" from weather.gov.hk/en/cis/climat.htm
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# 1. Download Daily Maximum Temperature (CLMMAXT)
# =============================================================================

def download_daily_max_temp(station="HKO"):
    """
    Download daily max temperature from HKO API.
    
    API: https://data.weather.gov.hk/weatherAPI/opendata/opendata.php
    dataType=CLMMAXT  →  Daily Maximum Temperature
    station=HKO       →  Hong Kong Observatory (Tsim Sha Tsui)
    rformat=csv
    """
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    params = {
        "dataType": "CLMMAXT",
        "station": station,
        "rformat": "csv",
        "lang": "en"
    }
    
    print(f"📥 Downloading daily max temperature for station: {station}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    filepath = os.path.join(DATA_DIR, f"daily_max_temp_{station}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    print(f"   ✅ Saved to {filepath}")
    print(f"   📊 Size: {len(resp.text)} bytes")
    return filepath


# =============================================================================
# 2. Download Daily Mean Temperature (CLMTEMP) for trend analysis
# =============================================================================

def download_daily_mean_temp(station="HKO"):
    """Download daily mean temperature."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    params = {
        "dataType": "CLMTEMP",
        "station": station,
        "rformat": "csv",
        "lang": "en"
    }
    
    print(f"📥 Downloading daily mean temperature for station: {station}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    filepath = os.path.join(DATA_DIR, f"daily_mean_temp_{station}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    print(f"   ✅ Saved to {filepath}")
    return filepath


# =============================================================================
# 3. Download Daily Minimum Temperature (CLMMINT) for range analysis
# =============================================================================

def download_daily_min_temp(station="HKO"):
    """Download daily minimum temperature."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    params = {
        "dataType": "CLMMINT",
        "station": station,
        "rformat": "csv",
        "lang": "en"
    }
    
    print(f"📥 Downloading daily min temperature for station: {station}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    filepath = os.path.join(DATA_DIR, f"daily_min_temp_{station}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    print(f"   ✅ Saved to {filepath}")
    return filepath


# =============================================================================
# 4. Download Daily Rainfall (for feature engineering)
# =============================================================================

def download_daily_rainfall(station="HKO"):
    """Download daily total rainfall."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
    params = {
        "dataType": "CLMRAIN",
        "station": station,
        "rformat": "csv",
        "lang": "en"
    }
    
    print(f"📥 Downloading daily rainfall for station: {station}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    
    filepath = os.path.join(DATA_DIR, f"daily_rainfall_{station}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    print(f"   ✅ Saved to {filepath}")
    return filepath


# =============================================================================
# 5. Test current real-time API (to understand format)
# =============================================================================

def test_realtime_api():
    """Test the HKO real-time weather API to understand its format."""
    url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    params = {"dataType": "rhrread", "lang": "en"}
    
    print("📥 Testing real-time weather API...")
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    # Extract HKO station temperature
    temps = data.get("temperature", {}).get("data", [])
    hko_temp = None
    for t in temps:
        if t.get("place") == "Hong Kong Observatory":
            hko_temp = t.get("value")
            break
    
    update_time = data.get("updateTime", "unknown")
    humidity = data.get("humidity", {}).get("data", [{}])[0].get("value", "N/A")
    
    print(f"   ✅ Real-time API working!")
    print(f"   🕐 Update time: {update_time}")
    print(f"   🌡️  HKO station temp: {hko_temp}°C")
    print(f"   💧 Humidity: {humidity}%")
    print(f"   📊 Total stations reporting: {len(temps)}")
    
    # Print all stations for reference
    print(f"\n   All station temperatures:")
    for t in sorted(temps, key=lambda x: x.get("value", 0), reverse=True):
        print(f"     {t['place']:35s} {t['value']}°C")
    
    return data


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🇭🇰 HKO Historical Data Downloader")
    print("=" * 60)
    print()
    
    # Test real-time API first
    test_realtime_api()
    print()
    
    # Download historical data
    download_daily_max_temp("HKO")
    download_daily_mean_temp("HKO")
    download_daily_min_temp("HKO")
    download_daily_rainfall("HKO")
    
    # Also download for Chek Lap Kok (airport) in case some markets use it
    print()
    print("--- Also downloading airport station (CLK) for comparison ---")
    download_daily_max_temp("CLK")
    
    print()
    print("=" * 60)
    print("✅ All downloads complete!")
    print(f"📁 Data saved to: {DATA_DIR}")
    print("=" * 60)
