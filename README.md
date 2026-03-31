# HK Weather Predictor × Polymarket

> A real-money trading system that predicts Hong Kong's daily maximum temperature using a 4-model meteorological ensemble, then finds mispriced buckets in Polymarket's weather prediction markets.

**The core insight:** Polymarket's HK weather markets are priced by intuition. A systematic ensemble of JMA + ECMWF + GFS forecasts combined with Kalman-filtered real-time HKO observations can find consistent edge — especially in the 9am–noon window when today's temperature trajectory is becoming clear but the market hasn't fully repriced.

---

## How It Works

```
                    ┌──────── Historical Data ────────┐
                    │  HKO 1884–2026 (49,367 days)    │
                    │  HistGradientBoosting model      │
                    │  (warming/cooling predictor)     │
                    └─────────────┬───────────────────┘
                                  │ 15% weight (prior)
                                  ▼
 JMA API ──── 35% ─────►  ┌─────────────────────────┐
 ECMWF API ── 30% ─────►  │  Weighted Ensemble       │
 GFS (30 members) ─ 20% ► │  Probability Engine      │  ──► Per-bucket probabilities
                           └─────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │  Kalman Filter Correction       │ ← HKO real-time (today only)
                    │  K-gain: 0.20 @ 6am → 0.97 @ 2pm│
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │  Edge = Our prob − Market price │
                    │  Kelly position sizing          │
                    │  Wave trading entry/exit windows│
                    └─────────────────────────────────┘
```

### 1. Four-Model Ensemble

Each weather model contributes a Gaussian (or empirical) probability distribution over temperature buckets:

| Model | Source | Weight | Why |
|-------|--------|--------|-----|
| **JMA** | open-meteo JMA API | **35%** | Best accuracy for East Asia subtropical weather |
| **ECMWF** | open-meteo ECMWF | **30%** | World's leading medium-range NWP model |
| **GFS** | 30-member ensemble | **20%** | Provides empirical distribution from 30 members directly |
| **Historical** | HKO 1884–2026 | **15%** | Bayesian smoothing prior; prevents over-concentration |

Forecast uncertainty grows with lead time — `σ` scales by `1 + 0.3 × min(days_away, 5)`.

### 2. Kalman Filter Real-Time Correction (Today Only)

When HKO reports the current temperature, the system extrapolates to today's predicted maximum using historically derived gain-to-peak curves, then blends that observation into the ensemble using a Kalman gain schedule:

```
K-gain schedule:
  06:00 → K = 0.20  (model still dominant)
  09:00 → K = 0.55
  11:00 → K = 0.75  (observation getting important)
  14:00 → K = 0.95  (near peak — observation almost definitive)
  15:00 → K = 0.97
```

If the observation diverges >1°C from the model average, the K-gain is automatically boosted — catching situations where the models are plainly wrong.

### 3. Polymarket Edge & Trading Strategy

```python
edge = our_probability - market_price
# Example: we estimate 28.5% for 23°C bucket; market prices it at 23.4%
# edge = +5.1% → BUY signal
```

**Strategy B — Wave Trading** (don't hold to settlement; trade the repricing):

| Time | Entry window | Exit window |
|------|-------------|-------------|
| Evening before | 9pm – 8am | 9am – noon |
| Today <9am | Now – 9am | 9am – noon |
| Today 9am–noon | Now (prime window) | noon – 2pm |
| Today noon–3pm | Now (near peak) | Hold to settlement |
| Today >3pm | Not recommended | — |

**Position sizing** uses a conservative Kelly formula (15% fractional Kelly, 30% single-bucket cap) with automatic size reduction for low liquidity (<$200 book).

---

## Data

| Dataset | Source | Range | Rows |
|---------|--------|-------|------|
| Daily max temperature (HKO) | HK Observatory | 1884–2026 | 49,367 |
| Daily min temperature | HK Observatory | 1884–2026 | 49,367 |
| Daily mean temperature | HK Observatory | 1884–2026 | 49,367 |
| Daily rainfall | HK Observatory | 1884–2026 | 49,367 |
| Daily max temperature (Chek Lap Kok) | HK Observatory | 1997–2026 | — |

Raw CSV files are included in `data/raw/` (~2.4 MB). Features are pre-computed in `data/processed/hko_features.csv`.

---

## Project Structure

```
hk-weather-predictor/
├── src/
│   ├── server.py                  # Main backend (968 lines) — ThreadingHTTPServer
│   ├── dashboard.html             # Frontend UI (530 lines) — date picker, step-by-step analysis, Strategy B panel
│   ├── 01_download_hko_data.py    # Scrape 142 years of HKO data
│   ├── 02_analyze_and_train.py    # Feature engineering + HistGradientBoosting model training
│   └── 03_realtime_predictor.py   # CLI version for quick predictions
├── data/
│   ├── raw/                       # 5 × HKO CSV files (1884–2026)
│   └── processed/
│       └── hko_features.csv       # Feature-engineered dataset
├── models/
│   ├── extrapolation_params.json  # Monthly gain-to-peak parameters (μ, σ per hour)
│   └── hgb_warming_cooling.pkl    # HistGradientBoosting warming/cooling classifier
└── DOCUMENTATION.md               # Full technical specification (680 lines)
```

---

## Setup & Run

**Requirements:** Python 3.10+, no paid API keys needed (all data sources are free)

```bash
# 1. Clone
git clone https://github.com/Richerwang228/hk-weather-predictor.git
cd hk-weather-predictor

# 2. Install dependencies
pip install numpy scipy scikit-learn requests

# (Optional) Re-download fresh HKO data
python src/01_download_hko_data.py

# (Optional) Retrain the model
python src/02_analyze_and_train.py

# 3. Start the server and dashboard
python src/server.py
# Open http://localhost:8080 in your browser

# Or use the CLI predictor
python src/03_realtime_predictor.py
```

---

## Confidence Levels

The system reports a confidence tier alongside every probability estimate:

| Scenario | Confidence | Range |
|----------|-----------|-------|
| Today, after 2pm | High | 85%+ |
| Today, 8am–2pm | Medium-High | 65–85% |
| Tomorrow | Medium-High | 60–75% |
| 2–3 days out | Medium | 40–60% |
| 4+ days out | Low | 15–35% |

Confidence is further reduced when the three NWP models disagree by >2°C.

---

## Resolution Source

Markets resolve against the **Absolute Daily Max (°C)** from the [HKO Daily Extract](https://www.weather.gov.hk/en/cis/climat.htm), measured at Hong Kong Observatory HQ (King's Park, 22.30°N 114.17°E).

---

## License

MIT
