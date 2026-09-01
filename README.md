# Flood Risk Prediction - AI Model

## Overview

Machine learning component for the **Flood Alert & Rescue Support System**,
developed to predict short-term extreme-rainfall / flood-risk proxy events
for locations across Vietnam.

The project trains three separate **XGBoost classification models** to
predict the probability of an extreme-rainfall event at different lead
times:

- **1 day ahead**
- **2 days ahead**
- **3 days ahead**

The trained models are later served through a separate **FastAPI AI
service** and consumed by the Spring Boot backend.

---

## Research Foundation

This project **adapts the research ideas and data-processing approach**
from the open-source project:

**ECMWF Code for Earth - ml_flood**

https://github.com/ECMWFCode4Earth/ml_flood

The original `ml_flood` project investigates machine-learning techniques
for flood-event prediction using open meteorological datasets from
**ECMWF/Copernicus**, including ERA5 data. Its workflow explores
meteorological predictors, temporal dependencies, and hydrology-related
variables for flood prediction.

This project inherits the main research direction and processing concepts,
particularly:

- Using **ERA5 meteorological data** as predictors.
- Using precipitation and hydrological-related variables.
- Capturing temporal dependencies through **lag features**.
- Using **rolling-window features** to represent accumulated weather
  conditions.
- Incorporating hydrology-inspired information such as runoff and
  antecedent precipitation.

However, this project **does not directly reuse the original model**.
It implements a separate classification-based approach adapted to the
Vietnamese context.

### Main differences from the original project

| Original `ml_flood` | This project |
|---|---|
| ECMWF/Copernicus flood prediction research | Vietnam-focused flood-risk prediction |
| ERA5 + GloFAS / discharge-oriented targets | ERA5-based extreme-rainfall proxy target |
| Multiple ML techniques | XGBoost classification |
| Flood/discharge prediction | Extreme-rainfall / flood-risk proxy classification |
| Original research datasets and structure | Reprocessed dataset for 34 Vietnamese provinces/cities |
| Research comparison of ML methods | Three lead-time prediction models |

---

## Project Objective

The objective is to develop a machine-learning pipeline that can:

1. Process historical ERA5 meteorological data.
2. Construct temporal and meteorological features.
3. Define an extreme-rainfall proxy label for each province.
4. Generate prediction targets for different lead times.
5. Train separate XGBoost models for 1-, 2-, and 3-day prediction.
6. Evaluate model performance using classification metrics.
7. Select an operational probability threshold for early warning.
8. Export trained models for use by the FastAPI prediction service.

---

## Data

### Historical Data

The training dataset is built from **ERA5 single-level meteorological
data** covering Vietnam.

- Period: **2000-2025**
- Temporal resolution: **6-hourly**
- Spatial scope: Vietnam
- Administrative coverage: **34 provinces/cities**

The processed dataset contains:

- **307,836 rows**
- **103 columns**
- **97 training features**
- Approximately **10.01% positive samples**

### Meteorological Variables

The ERA5 data includes:

- Total precipitation
- 2m temperature
- 2m dewpoint temperature
- Surface pressure
- 10m U-component of wind
- 10m V-component of wind
- Evaporation
- Runoff

---

## Data Processing & Feature Engineering

The preprocessing pipeline converts the original ERA5 data into
province-level daily features.

### Time Conversion

ERA5 timestamps are converted from UTC to Vietnam local time using
**UTC+7** before daily aggregation.

### Daily Features

Daily meteorological statistics include:

- `tp_mean`
- `tp_max`
- `tp_p90`
- `tp_p99`
- `t2m_mean`
- `t2m_max`
- `d2m_mean`
- `rh_mean`
- `sp_mean`
- `u10_mean`
- `v10_mean`
- `ws_mean`
- `ws_max`
- `evap_mean`
- `ro_mean`
- `ro_max`

### Lag Features

Temporal lag features are created for up to **7 previous days** for
selected weather and hydrological variables.

These features allow the model to capture recent weather conditions and
their temporal persistence.

### Rolling Features

Rolling-window features are generated using:

- 3-day windows
- 5-day windows
- 7-day windows

These features represent accumulated or persistent precipitation and
runoff conditions.

### Antecedent Precipitation Index

An Antecedent Precipitation Index (API) is used to represent the influence
of previous rainfall:

```text
API_t = k × API_(t-1) + P_t
```

The training pipeline uses:

```text
k = 0.85
```

---

## Target Definition

The project does **not** use directly observed flood water levels as the
training label.

Instead, `flood_label` is constructed as a **proxy for extreme-rainfall
conditions**.

For each province, the 90th percentile of daily maximum precipitation
(`tp_max`) is calculated:

```python
province_thresholds = (
    df.groupby('province')['tp_max']
    .quantile(0.90)
)
```

A day is labeled as an extreme event when:

```python
flood_label = tp_max >= province_specific_p90
```

This creates a province-specific extreme-rainfall threshold rather than
using one fixed rainfall threshold for the entire country.

### Why province-specific thresholds?

Different regions of Vietnam have different rainfall regimes. A
province-specific percentile threshold allows the model to adapt to
local climatological differences.

### Important Limitation

The resulting label is an **extreme-rainfall proxy for flood risk**, not
direct ground-truth flood observations.

Therefore, the model should not be interpreted as directly predicting
observed river flooding without additional validation against independent
flood observations such as:

- River water levels
- River discharge
- Flood extent
- Inundation maps
- Damage reports
- GloFAS discharge thresholds

---

## Prediction Targets

Separate targets are generated for each lead time.

For a province:

```python
target = flood_label.shift(-lead)
```

Three prediction targets are used:

```text
Lead 1 → Extreme event 1 day ahead
Lead 2 → Extreme event 2 days ahead
Lead 3 → Extreme event 3 days ahead
```

Three independent XGBoost models are trained:

```text
xgboost_lead1d.pkl
xgboost_lead2d.pkl
xgboost_lead3d.pkl
```

---

## Model Training

### Algorithm

The project uses:

**XGBClassifier**

The models are trained independently for each prediction lead time.

### Train / Validation / Test Split

A temporal split is used:

```text
70% → Training
15% → Validation
15% → Test
```

A temporal split is used instead of a random split to reduce information
leakage from future observations into the training data.

### Class Imbalance

Extreme events represent only around 10% of the processed dataset.

To address this imbalance, the training process calculates:

```text
scale_pos_weight = negative_samples / positive_samples
```

and passes the resulting value to XGBoost.

### Model Configuration

```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    eval_metric="auc",
    early_stopping_rounds=50
)
```

---

## Model Evaluation

The trained models are evaluated using:

- ROC-AUC
- Precision
- Recall
- F1-score

### Results

| Lead Time | ROC-AUC | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 1 day | 0.8645 | 0.8121 | 0.2659 | 0.4006 |
| 2 days | 0.8327 | 0.8267 | 0.2260 | 0.3550 |
| 3 days | 0.8235 | 0.8311 | 0.2179 | 0.3453 |

The models achieve ROC-AUC values above 0.82 across all three lead times.

Recall is intentionally important for the early-warning use case because
missing a potentially dangerous event can be more costly than generating
additional false alarms.

The relatively low precision indicates a significant false-alarm trade-off
and is an important limitation of the current approach.

---

## Warning Threshold

A separate threshold-tuning process is used to convert model probability
into an operational warning decision.

The threshold is tuned using a Precision-Recall curve.

The selected operational threshold is:

```text
0.35
```

At this threshold:

```text
Precision ≈ 0.217
Recall    ≈ 0.921
F1        ≈ 0.352
```

The threshold prioritizes **high recall** for an early-warning scenario.

This means the system is designed to detect a large proportion of
extreme-rainfall events while accepting a higher number of false alarms.

---

## Historical Verification

A separate historical verification process evaluates the 1-day model on
the final 15% of the dataset.

Reported results include:

```text
Test samples: 46,135
ROC-AUC:      0.8940
```

For the extreme class:

```text
Precision: 0.24
Recall:    0.97
F1:        0.38
```

The verification also checks selected historical event windows.

These results provide additional evidence that the model can identify
days matching the extreme-rainfall proxy label.

However, this is **not independent ground-truth flood validation**.

---

## Realtime Prediction

The trained models are used by the separate FastAPI prediction service.

The realtime pipeline retrieves weather data from **Open-Meteo** and
transforms the data into features compatible with the trained models.

```text
Open-Meteo
    |
    v
Hourly Weather Data
    |
    v
Daily Aggregation
    |
    v
Feature Engineering
    |
    v
XGBoost Model
    |
    v
Prediction Probability
    |
    v
Flood Alert Backend
```

### Realtime Input Variables

The realtime weather service retrieves variables including:

- Precipitation
- 2m temperature
- 2m dewpoint
- Surface pressure
- Wind speed
- Wind direction
- Relative humidity
- Reference evapotranspiration

The data is aggregated into daily features before inference.

---

## Model Artifacts

The trained model artifacts include:

```text
models/
├── xgboost_lead1d.pkl
├── xgboost_lead2d.pkl
├── xgboost_lead3d.pkl
├── feature_cols.pkl
└── optimal_threshold.pkl
```

These artifacts are consumed by the FastAPI AI service.

---

## Project Structure

```text
flood-alert-ai/
├── .github/
│   └── workflows/
├── docs/
│   └── model_documentation.md
├── models/
├── reports/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

---

## Related Repositories

### Main Backend

Spring Boot backend responsible for weather integration, IoT monitoring,
flood-risk assessment, citizen alerts, authentication, and rescue
coordination.

https://github.com/khanhan0603/flood-alert

### FastAPI Prediction Service

FastAPI service that loads the trained models and exposes prediction
endpoints to the main backend.

https://github.com/khanhan0603/ai-server-flood-alert

### Research Reference

This project adapts research ideas and data-processing concepts from:

**ECMWF Code for Earth - ml_flood**

https://github.com/ECMWFCode4Earth/ml_flood

The original project investigates machine-learning approaches for flood
prediction using open ECMWF/Copernicus datasets, including ERA5. This
repository is an independent adaptation for a Vietnam-focused
classification and early-warning use case.

---

## Limitations & Future Improvements

The current model has several limitations:

- The target label is an extreme-rainfall proxy rather than observed flood
  events.
- Precision is relatively low because the current approach prioritizes
  recall.
- Realtime runoff is approximated because Open-Meteo does not provide the
  same ERA5 runoff variable used during training.
- Independent validation using river discharge, water-level observations,
  flood extent, or other ground-truth flood datasets is still required.
- Training and realtime inference should use consistent feature-processing
  parameters.

Future improvements can include validating the model against independent
flood observations and improving the consistency between the training and
realtime feature pipelines.

---

## Acknowledgement

This project acknowledges the research direction and open-source work of
the **ECMWF Code for Earth `ml_flood` project**.

The original project can be found at:

https://github.com/ECMWFCode4Earth/ml_flood
