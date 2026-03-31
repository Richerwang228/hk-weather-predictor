# 香港天气预测 × Polymarket 交易系统 — 完整技术文档

> **项目目标**：通过多源气象数据融合 + 实时修正，预测香港每日最高气温，并与 Polymarket 天气预测市场对比，寻找正期望值（+EV）交易机会。

---

## 1. 项目背景

### 1.1 什么是 Polymarket 天气市场

Polymarket 是一个去中心化预测市场平台，用户可以对现实世界事件进行 YES/NO 下注。其中有一类市场专门预测 **"香港某天的最高气温会落在哪个档位"**：

- 每个市场对应一个温度档位（如 22°C、23°C、24°C）
- 用户可以买入 YES（认为最高温会在这个档位）或 NO
- 市场价格反映了集体对概率的判断（如价格 0.54 = 市场认为 54% 概率）

### 1.2 结算规则

Polymarket 原文：

> *"This market will resolve to the temperature range that contains the highest temperature recorded by the Hong Kong Observatory in degrees Celsius."*
>
> *"The resolution source will be the **Absolute Daily Max (deg. C)** from the Hong Kong Observatory's **Daily Extract**, available at: https://www.weather.gov.hk/en/cis/climat.htm"*

**关键点**：
- 结算数据源：香港天文台（HKO）官方 **Daily Extract** 中的 **Absolute Daily Max**
- 测量站点：香港天文台总部（尖沙咀 King's Park），经纬度 22.30°N, 114.17°E
- 精度：一位小数（如 24.3°C）
- 结算时间：当 Daily Extract 数据发布后结算（通常次日）

### 1.3 我们的优势

| 市场参与者 | 信息来源 | 弱点 |
|-----------|---------|------|
| 普通交易者 | 个人判断、天气 App | 缺乏系统性分析 |
| 我们的系统 | 3 个气象模型 + HKO 实测 + 历史统计 | 模型预报有误差 |

核心策略：**当我们的概率估计与市场价格有显著差异（Edge > 3%）时，说明存在交易机会。**

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   Web Dashboard (前端)                    │
│    dashboard.html · 日期选择 · 逐步分析 · 策略B面板       │
└───────────────┬─────────────────────────────────────────┘
                │ HTTP API (localhost:8080)
┌───────────────▼─────────────────────────────────────────┐
│                 Python Server (后端)                      │
│  server.py · ThreadingHTTPServer · 多线程并发              │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │          数据采集层                           │       │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐  │       │
│  │  │JMA API │ │ECMWF   │ │GFS     │ │HKO   │  │       │
│  │  │Open-   │ │Open-   │ │Ensemble│ │Real- │  │       │
│  │  │Meteo   │ │Meteo   │ │API     │ │time  │  │       │
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └──┬───┘  │       │
│  │      │          │          │         │       │       │
│  │  ┌───▼──────────▼──────────▼─────────▼───┐   │       │
│  │  │     加权集合概率计算引擎                │   │       │
│  │  │  JMA 35% + ECMWF 30% + GFS 20%        │   │       │
│  │  │  + Historical 15%                      │   │       │
│  │  │  + 卡尔曼实时修正 (今天)               │   │       │
│  │  └───────────────┬───────────────────────┘   │       │
│  │                  │                           │       │
│  │  ┌───────────────▼───────────────────────┐   │       │
│  │  │    Polymarket Edge 计算               │   │       │
│  │  │  Edge = 我的概率 - 市场价格            │   │       │
│  │  │  + 策略B (止盈/止损/仓位)              │   │       │
│  │  └───────────────────────────────────────┘   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### 2.1 文件结构

```
hk-weather-predictor/
├── src/
│   ├── server.py              # 主服务器（968行）
│   ├── dashboard.html         # 前端 UI（530行）
│   ├── 01_download_hko_data.py  # 历史数据下载脚本
│   ├── 02_analyze_and_train.py  # 数据分析与模型训练
│   └── 03_realtime_predictor.py # 命令行版预测器
├── data/
│   ├── raw/
│   │   ├── daily_max_temp_HKO.csv   # 日最高温 1884-2026（49,367行）
│   │   ├── daily_min_temp_HKO.csv   # 日最低温
│   │   ├── daily_mean_temp_HKO.csv  # 日平均温
│   │   ├── daily_rainfall_HKO.csv   # 日降雨量
│   │   └── daily_max_temp_CLK.csv   # 赤鱲角站对比数据
│   └── processed/
│       └── hko_features.csv         # 特征工程后数据
├── models/
│   ├── extrapolation_params.json    # 升温外推参数
│   └── hgb_warming_cooling.pkl      # 升降温预判模型
└── DOCUMENTATION.md                 # 本文档
```

---

## 3. 数据源详解

### 3.1 香港天文台历史数据（离线）

| 数据集 | 来源 | 范围 | 用途 |
|--------|------|------|------|
| 日最高温 | HKO 气候资料 | 1884-2026（49,367行）| 月度分布统计、档位概率 |
| 日最低温 | HKO 气候资料 | 1884-2026 | 日升温幅度计算 |
| 日平均温 | HKO 气候资料 | 1884-2026 | 趋势分析 |
| 日降雨量 | HKO 气候资料 | 1884-2026 | 特征工程（阴雨天温度偏低）|

### 3.2 气象模型预报（实时 API）

#### JMA（日本气象厅）— 权重 35%

```
API: https://api.open-meteo.com/v1/jma
参数: hourly=temperature_2m, forecast_days=7
特点: 东亚地区准确度最高，对亚热带天气系统预报优势明显
σ值: 1.5 × sigma_boost（随预报天数递增）
```

#### ECMWF（欧洲中期天气预报中心）— 权重 30%

```
API: https://api.open-meteo.com/v1/ecmwf
备用: https://api.open-meteo.com/v1/forecast?models=ecmwf_ifs025
特点: 全球最强数值天气预报模型，中期预报业界标杆
σ值: 1.3 × sigma_boost
注意: 主 API 偶尔不可用，系统自动切换备用端点
```

#### GFS 集合预报（美国国家气象局）— 权重 20%

```
API: https://ensemble-api.open-meteo.com/v1/ensemble?models=gfs_seamless
特点: 30个集合成员，提供概率分布而非单一预报
用法: 直接统计30个成员的最高温分布，无需假设高斯分布
      某成员预报最高温 24.3°C → 落入 24°C 档位
```

#### 历史统计 — 权重 15%

```
来源: models/extrapolation_params.json
用法: 均匀分布在所有档位上作为先验（贝叶斯平滑）
效果: 防止概率过度集中在某一个档位
```

### 3.3 香港天文台实时观测（今天专用）

```
API 1: https://data.weather.gov.hk/weatherAPI/opendata/weather.php
        ?dataType=rhrread（实时温湿度，27个站点，10分钟更新）
API 2: 同上，fallback 读取 Hong Kong Observatory 站
返回: 当前温度、湿度、更新时间
用途: 卡尔曼滤波修正（仅限今天的预测）
```

### 3.4 Polymarket 市场数据

```
API: https://gamma-api.polymarket.com/events?slug={slug}
Slug 格式: highest-temperature-in-hong-kong-on-{month}-{day}-{year}
探测范围: 前1天 到 后10天
返回: 每个温度档位的 YES/NO 价格、交易量、流动性
```

---

## 4. 概率计算引擎

### 4.1 基础概率（模型预报→档位概率）

核心函数：`compute_bucket_probs(jma, ecmwf, gfs, hko_obs, current_hour, days_away)`

**Step 1: 确定中心温度**

```python
forecast_maxes = [jma["max"], ecmwf["max"], gfs["mean"]]
center = np.mean(forecast_maxes)  # 如 (22.6 + 23.8 + 22.4) / 3 = 22.93
```

**Step 2: 生成档位范围**

```python
spread = 5 + min(days_away, 3)  # 今天: 5°C范围, 3天后: 8°C范围
buckets = range(int(center) - spread, int(center) + spread + 1)
# 例: center=23 → 档位 18°C 到 28°C
```

**Step 3: 每个模型贡献概率**

```python
sigma_boost = 1.0 + 0.3 * min(days_away, 5)  # 距离越远σ越大

# JMA (权重35%): 高斯分布，中心=jma_max, σ=1.5×boost
probs[b] += 0.35 * (norm.cdf(b+1, jma_max, 1.5*σ) - norm.cdf(b, jma_max, 1.5*σ))

# ECMWF (权重30%): 高斯分布，σ=1.3×boost（更集中因为更准）
probs[b] += 0.30 * (norm.cdf(b+1, ecmwf_max, 1.3*σ) - norm.cdf(b, ecmwf_max, 1.3*σ))

# GFS (权重20%): 直接计数（30成员中落在该档位的比例）
probs[b] += 0.20 * (count_in_bucket / n_members)

# Historical (权重15%): 均匀分布（贝叶斯平滑）
probs[b] += 0.15 * (1 / n_buckets)
```

### 4.2 卡尔曼实时修正（仅限今天）

当 `days_away == 0` 且有 HKO 实测温度时，系统执行卡尔曼滤波修正：

**Step 1: 估算今日最高温（从当前温度外推）**

核心方法：**gain-to-peak**（从当前温度估算距离峰值的温升）

```python
# 每个小时到峰值(14:00)的典型温升（基于历史统计）
# mean_rise = 4.68°C（3月份 月均日升温幅度）
gain_to_peak = {
    6:  (4.68, 2.0),   # 6am: 全天升温还在前方
    7:  (4.35, 1.8),
    8:  (3.98, 1.6),
    9:  (3.65, 1.5),
    10: (3.28, 1.4),   # 10am: 还有 ~3.3°C 升温空间
    11: (2.71, 1.2),   # 11am: 还有 ~2.7°C
    12: (1.87, 1.0),   # 12pm: 还有 ~1.9°C
    13: (0.84, 0.7),   # 1pm:  接近峰值
    14: (0.20, 0.5),   # 2pm:  几乎到达峰值
    15: (0.00, 0.3),   # 3pm+: 峰值已过
}
# 例: 11am 实测 21°C → est_max = 21 + 2.71 = 23.71°C (σ=1.2)
```

**Step 2: 卡尔曼增益 K**

K 决定了实测观察 vs 模型预报的权重分配：

```python
K_sched = {6:.20, 7:.30, 8:.40, 9:.55, 10:.65, 11:.75,
           12:.85, 13:.90, 14:.95, 15:.97, 16:.98, 17:1.0}
# 11am: K = 0.75 → 75% 权重给实测外推, 25% 给模型
```

**Step 3: 动态 K 加速（divergence boost）**

当实测外推结果与模型平均偏差 > 1°C 时，进一步提升 K：

```python
divergence = abs(est_max - model_avg)
if divergence > 1.0:
    k_boost = min(0.25, (divergence - 1.0) * 0.15)
    K = min(0.98, K + k_boost)
# 例: est_max=24.4, model_avg=22.9 → 偏差1.5°C → K从0.75增到0.82
```

**Step 4: 概率混合**

```python
# obs_probs = 基于实测外推的高斯概率分布
# probs = 基于模型的原始概率分布
for b in buckets:
    probs[b] = (1-K) * probs[b] + K * obs_probs[b]
```

### 4.3 置信度评估

```python
# 基于数据丰富度和预测距离
if is_today and current_hour >= 14:  # 今天下午
    confidence = "高"     # 85%+
elif is_today and current_hour >= 8: # 今天上午
    confidence = "中高"   # 65-85%
elif days_away == 1:                 # 明天
    confidence = "中高"   # 60-75%
elif days_away <= 3:                 # 2-3天后
    confidence = "中"     # 40-60%
else:                                # 4天+
    confidence = "低"     # 15-35%
```

### 4.4 模型一致性检测

```python
model_maxes = [JMA_max, ECMWF_max, GFS_mean]
spread = max(model_maxes) - min(model_maxes)

if spread <= 1.0:   # 模型一致 → ✅ 高
    ...
elif spread <= 2.0:  # 中等分歧 → ⚠️ 中
    ...
else:                # 严重分歧 → ❌ 低
    confidence *= 0.8  # 降低置信度 20%
```

---

## 5. 交易策略引擎

### 5.1 Edge 计算

```python
edge = our_probability - market_price
# 例: 我们认为 23°C = 28.5%, 市场价 23.4%
#     edge = 28.5% - 23.4% = +5.1%
```

**信号分级**：

| Edge 范围 | 信号 | 含义 |
|-----------|------|------|
| ≥ 8% | 买入 | 强烈正期望值 |
| 4-8% | 观望 | 有 Edge 但风险较高 |
| < 4% | 跳过 | Edge 不够覆盖交易成本 |

### 5.2 策略 B — 波段交易方案

**核心理念**：不是持仓到结算，而是在价格变动中获利。

#### 入场/出场时间窗口

| 场景 | 入场 | 出场 |
|------|------|------|
| 明天的市场 | 今晚 21:00 - 明天 08:00 | 明天 09:00 - 12:00 |
| 今天 < 9am | 现在 - 09:00 | 09:00 - 12:00 |
| 今天 9am-12pm | 现在（最佳窗口）| 12:00 - 14:00 |
| 今天 12-15pm | 现在（峰值接近）| 持仓到结算 |
| 今天 > 15pm | 不建议入场 | — |

#### 止盈/止损

```python
# 止盈: 根据买入价设定 2-4 倍目标
tp_mult = min(4.0, max(2.0, 0.30 / entry_price))
take_profit = min(0.85, entry_price * tp_mult)
# 例: 买入@4¢ → 止盈@16¢ (+300%)

# 止损: 固定 -45%
stop_loss = max(0.01, entry_price * 0.55)
# 例: 买入@4¢ → 止损@2¢
```

#### 仓位管理（Kelly 公式）

```python
# 保守 Kelly（15% Kelly）
if edge > 0:
    odds = (1/market_price) - 1
    kelly = 0.15 * (edge / odds)
    position_pct = min(0.30, kelly)  # 单档上限 30%
    position_amt = capital * position_pct
    position_amt = max(0.50, position_amt)  # 最低 $0.50
```

#### 流动性检查

| 流动性 | 标签 | 操作 |
|--------|------|------|
| ≥ $500 | 充足 | 正常交易 |
| $200-500 | 一般 | 正常交易 |
| $100-200 | 偏低 | 自动减半仓位 |
| < $100 | 极低 | 不建议交易 |

### 5.3 推荐排序逻辑

系统按 **AI 概率最高** 排序推荐（不是按 Edge 最大），门槛为 Edge > 3%：

```python
positive_edges = {k:v for k,v in edges.items() if v["edge"] > 0.03}
sorted by v["our_prob"] descending  # 概率最高的排最前
```

这确保了推荐的第一个档位是系统最有信心的预测结果。

---

## 6. 历史数据分析与模型训练

### 6.1 阶段 1: 数据获取 (`01_download_hko_data.py`)

从 HKO 气候资料下载 1884-2026 年的日最高温、最低温、平均温、降雨量。

### 6.2 阶段 2: 升温外推参数 (`02_analyze_and_train.py`)

从历史数据统计每月的日升温幅度（max - min）：

| 月份 | 平均日升温 | 标准差 | 中位数 |
|------|-----------|--------|--------|
| 1月 | 4.32°C | 1.53 | 4.4°C |
| 2月 | 4.54°C | 1.89 | 4.4°C |
| **3月** | **4.68°C** | **2.05** | **4.6°C** |
| 4月 | 4.68°C | 1.97 | 4.6°C |
| 7月 | 4.82°C | 1.30 | 4.9°C |
| 12月 | 4.32°C | 1.44 | 4.3°C |

还统计了**温度转移概率**：

```
升温→升温: 37.3%   升温→降温: 44.4%   升温→持平: 37.4%
降温→升温: 36.7%   降温→降温: 33.3%   降温→持平: 32.5%
持平→升温: 26.0%   持平→降温: 22.3%   持平→持平: 30.1%
```

保存到 `models/extrapolation_params.json`。

### 6.3 阶段 3: 升降温预判模型

训练 HistGradientBoosting 分类器预判明天是升温/降温/持平：

**16 个特征**：
1. 今天最高温
2. 今天最低温
3. 今天温差
4. 最高温偏差（vs 当月均值）
5. 月份
6. 星期几
7. 3 天滑动平均
8. 7 天滑动平均
9. 温度变化趋势（1天、3天、7天）
10. 湿度（最近可用）
11. 降雨量（最近可用）
12-16. 滞后项

**按季节准确率**：
- 冬季：55.8%
- 春季：48.6%
- 夏季：46.4%
- 秋季：49.8%

使用 Isotonic Regression 进行概率校准。

---

## 7. Web Dashboard 前端

### 7.1 设计系统

基于 **UI/UX Pro Max** 设计框架：

| 属性 | 值 |
|------|-----|
| **字体** | Exo 2（Google Fonts，金融/科技风格）|
| **主色** | 金色 `#F59E0B` + 紫色 `#8B5CF6` |
| **背景** | OLED 深黑 `#0A0E1A` / `#0F172A` |
| **图标** | Lucide SVG（不使用 emoji）|
| **效果** | backdrop-filter blur、glow text-shadow |
| **动画** | 200ms cubic-bezier(0.4,0,0.2,1) |
| **无障碍** | prefers-reduced-motion 支持 |
| **响应式** | 375px / 768px / 1440px |

### 7.2 交互流程

```
1. 页面加载 → 调用 /api/dates → 渲染日期选择器（skeleton 加载态）
2. 用户选择日期 → 点击"开始分析"
3. 调用 /api/predict?date=YYYY-MM-DD → 返回完整 JSON
4. 前端逐步动画展示 9 个分析步骤（250ms 间隔）
5. 渲染完整结果:
   ├── 4 张模型卡片（JMA / ECMWF / GFS / HKO）
   ├── 预测结果 + 置信度仪表盘
   ├── 概率分布条形图（渐变色）
   ├── Edge 交易信号表（含流动性列）
   ├── 策略 B 面板（时间线 + 推荐卡片 + 本金输入）
   └── 逐时温度曲线图（SVG）
```

### 7.3 本金动态更新

用户在"交易本金"输入框输入金额后，`recalcPositions()` 自动按比例更新所有推荐的：
- 建议仓位（$金额和股数）
- 最大亏损
- 期望收益
- 汇总栏

---

## 8. API 接口文档

### `GET /api/dates`

获取可用预测日期列表及市场信息。

**返回**：
```json
{
  "dates": [
    {
      "date": "2026-03-21",
      "days_away": 0,
      "confidence": "中高",
      "confidence_value": 0.72,
      "market_count": 11,
      "label": "今天 (3月21日 周六)"
    }
  ],
  "markets": [...]  // 所有 Polymarket 市场数据（缓存）
}
```

### `GET /api/predict?date=YYYY-MM-DD`

执行完整预测分析。

**返回**（核心字段）：
```json
{
  "target_date": "2026-03-21",
  "days_away": 0,
  "is_today": true,
  "current_hour": 11,
  "forecasts": {
    "jma": {"max": 22.6, "min": 20.1, "hourly": [...]},
    "ecmwf": {"max": 23.8, "min": 20.3, "hourly": [...]},
    "gfs": {"mean": 22.4, "std": 0.3, "min": 22.0, "max": 23.2, "members": 30}
  },
  "hko": {"hko_temp": 21, "humidity": 84},
  "probabilities": {"22": 0.238, "23": 0.285, "24": 0.232, ...},
  "kalman_gain": 0.75,
  "confidence": 0.72,
  "confidence_label": "中高",
  "top_bucket": 23,
  "top_prob": 0.285,
  "edges": {
    "23": {"bucket": 23, "our_prob": 0.285, "market_prob": 0.234, "edge": 0.051, "kelly": 0.023, "signal": "观望", "liquidity": 348, "volume": 1817, "liq_level": "一般"},
    ...
  },
  "strategy": {
    "entry_window": "现在（实测数据已有，精度高）",
    "exit_window": "12:00 - 14:00（峰值确认后卖出）",
    "urgency": "🟢 当前是最佳窗口",
    "capital": 10,
    "recommendations": [
      {
        "bucket": 23,
        "our_prob": 0.285,
        "edge": 0.051,
        "entry_price": 0.234,
        "take_profit": 0.47,
        "stop_loss": 0.13,
        ...
      }
    ],
    "total_position": 1.50,
    "total_risk": -0.28,
    "total_expected": 0.77,
    "total_liquidity": 9489,
    "liquidity_ok": true
  },
  "model_consistency": {"spread": 1.4, "level": "中", "icon": "⚠️"},
  "steps": [
    {"label": "获取 JMA 预报", "status": "done", "detail": "最高温 22.6°C"},
    ...
  ]
}
```

---

## 9. 分析步骤详解

系统按 9 个步骤执行分析，每步都有状态和详情：

| # | 步骤 | 数据源 | 输出 |
|---|------|--------|------|
| 1 | 获取 JMA 预报 | Open-Meteo JMA API | 最高温 + 逐时曲线 |
| 2 | 获取 ECMWF 预报 | Open-Meteo ECMWF API | 最高温 + 逐时曲线 |
| 3 | 获取 GFS 集合预报 | Open-Meteo Ensemble API | 均值 ± 标准差 + 30 成员 |
| 4 | 获取 HKO 实测 | HKO 实时 API | 当前温度 + 湿度 |
| 5 | 计算档位概率 | 加权集合 + 卡尔曼 | 各温度档位概率分布 |
| 6 | 卡尔曼实时修正 | HKO 实测 + 外推参数 | 修正后概率 + K 值 |
| 7 | 置信度评估 | 预报距离 + 模型一致性 | 置信等级 + 模型偏差 |
| 8 | Polymarket 对比 | Gamma API | Edge + 交易信号 + 流动性 |
| 9 | 生成交易策略 | 策略 B 引擎 | 推荐 + 止盈止损 + 仓位 |

---

## 10. 已知限制与改进空间

### 10.1 当前限制

| 限制 | 说明 |
|------|------|
| **模型预报滞后** | JMA/GFS 预报偏保守，经常低估最高温 1-2°C |
| **HKO API 限制** | 实时 API 不提供当日最高温记录，只有当前温度 |
| **流动性不足** | 单档位流动性 $200-800，大额交易会造成滑点 |
| **无自动交易** | 需要人工在 Polymarket 下单 |
| **ECMWF 不稳定** | 主端点偶尔不可用，需 fallback |

### 10.2 改进方向

| 方向 | 难度 | 效果 |
|------|------|------|
| 接入更多实测站点（取最高值）| 低 | 结算温度可能高于 HKO 总部 |
| 引入太阳辐射/云量数据 | 中 | 晴天升温更大 |
| 历史准确率回测 | 中 | 量化系统的 Brier Score |
| 自动交易（CLOB API） | 高 | 需要 API Key + 钱包私钥 |
| 多日联合策略 | 中 | 分散风险到 3-4 天 |
| 机器学习误差修正 | 中 | 用 NN 学习模型偏差模式 |

---

## 11. 启动与使用

### 11.1 环境要求

```
Python 3.9+
依赖: requests, numpy, scipy
可选: scikit-learn (训练模型)
```

### 11.2 安装

```bash
cd hk-weather-predictor
pip install requests numpy scipy
```

### 11.3 启动

```bash
python3 src/server.py
# → 打开 http://localhost:8080
```

### 11.4 使用流程

1. 打开浏览器访问 `http://localhost:8080`
2. 选择要预测的日期（今天/明天/后天等）
3. 点击「开始分析」
4. 等待 9 步分析完成（约 15-30 秒，取决于 API 响应）
5. 查看预测结果、概率分布、交易信号
6. 查看策略 B 面板中的推荐操作
7. 可在"交易本金"输入框调整金额查看对应仓位

---

## 12. 关键参数速查

### 模型权重

| 模型 | 权重 | σ 基准 |
|------|------|--------|
| JMA | 35% | 1.5 |
| ECMWF | 30% | 1.3 |
| GFS | 20% | 直接计数 |
| Historical | 15% | 均匀分布 |

### 卡尔曼增益 K（今天）

| 时间 | K 值 | 含义 |
|------|------|------|
| 6am | 0.20 | 80% 模型, 20% 实测 |
| 8am | 0.40 | 60% 模型, 40% 实测 |
| 10am | 0.65 | 35% 模型, 65% 实测 |
| 11am | 0.75 | 25% 模型, 75% 实测 |
| 12pm | 0.85 | 15% 模型, 85% 实测 |
| 14pm | 0.95 | 5% 模型, 95% 实测 |

### 3 月份升温参数

```
平均日升温: 4.68°C
标准差: 2.05°C
中位数: 4.6°C
P25: 2.9°C
P75: 6.2°C
样本量: 7,728 天
数据范围: 2005-2026
```

---

*文档生成时间：2026-03-21 13:07 HKT*
*系统版本：v2.0（含策略 B + UI Pro Max + 精度修复）*
