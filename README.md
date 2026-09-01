# Flood Risk Prediction - AI Model

## Overview

Machine learning component for the **Flood Alert & Rescue Support System**,
developed to predict short-term extreme-rainfall events as a proxy for
flood-risk assessment across Vietnam.

The project trains three independent **XGBoost classification models** to
predict the probability of an extreme-rainfall event at different lead times:

- **1 day ahead**
- **2 days ahead**
- **3 days ahead**

The trained models are exported and served through a separate
**FastAPI AI service**, which is consumed by the main Spring Boot backend.

---

## My Role

**AI / Machine Learning Developer**

Responsible for the machine-learning implementation, including:

- Meteorological data preprocessing
- Feature engineering
- Extreme-rainfall target generation
- Lead-time target construction
- XGBoost model training
- Class-imbalance handling
- Model evaluation
- Prediction-threshold tuning
- Model artifact generation for deployment

The machine-learning pipeline and models were independently implemented for
this project with the support of AI-assisted development tools.

---

## Research Foundation

This project **adapts research ideas and data-processing concepts** from the
open-source project:

**ECMWF Code for Earth - `ml_flood`**

https://github.com/ECMWFCode4Earth/ml_flood

The original `ml_flood` project investigates machine-learning approaches for
flood prediction using open meteorological datasets from
**ECMWF/Copernicus**, including ERA5 data.

This project inherits the main research direction and selected processing
concepts, including:

- Using **ERA5 meteorological data** as predictors
- Using precipitation and hydrological-related variables
- Capturing temporal dependencies through **lag features**
- Using **rolling-window features** to represent accumulated weather
  conditions
- Incorporating hydrology-inspired information such as **runoff** and
  **antecedent precipitation**

However, this project **does not directly reuse the original model**.
The data-processing pipeline, target definition, feature engineering,
XGBoost training, evaluation, and lead-time prediction setup were adapted
and implemented for a Vietnam-focused use case.

### Adaptation from the Original Research

| `ml_flood` Research | This Project |
|---|---|
| ECMWF/Copernicus flood prediction research | Vietnam-focused flood-risk prediction |
| ERA5 and hydrological data | ERA5 meteorological and hydrological variables |
| Flood/discharge-oriented prediction | Extreme-rainfall proxy for flood risk |
| Multiple machine-learning approaches | XGBoost classification |
| Research-oriented prediction | 1-, 2-, and 3-day lead-time prediction |
| Original research workflow | Adapted preprocessing and feature-engineering pipeline |
| Original datasets and target definitions | Reprocessed Vietnam-focused dataset |

---

## Project Objective

The objective is to build a machine-learning pipeline that can:

1. Process historical ERA5 meteorological data.
2. Convert meteorological observations into province-level daily data.
3. Construct temporal and meteorological features.
4. Define a province-specific extreme-rainfall target.
5. Generate prediction targets for different lead times.
6. Train separate XGBoost models for 1-, 2-, and 3-day prediction.
7. Evaluate model performance using classification metrics.
8. Tune a probability threshold for early-warning purposes.
9. Export trained models for use by the FastAPI prediction service.

---

## Data

### Historical Data

The training dataset is built from **ERA5 single-level meteorological data**
covering Vietnam.

- Period: **2000-2025**
- Temporal resolution: **6-hourly**
- Spatial scope: Vietnam
- Administrative coverage: **34 provinces/cities**
- Dataset size: **333,506 records**

The processed dataset contains:

- **333,506 rows**
- **103 columns**
- **97 training features**
- Approximately **10.01% positive samples**

### Meteorological Variables

The dataset includes meteorological and hydrological-related variables such
as:

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

The preprocessing pipeline transforms the original ERA5 observations into
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

Temporal lag features are created for up to **7 previous days** for selected
weather and hydrological variables.

These features allow the models to capture recent weather conditions and
their temporal persistence.

### Rolling Features

Rolling-window features are generated using:

- 3-day windows
- 5-day windows
- 7-day windows

These features represent accumulated or persistent precipitation and runoff
conditions.

### Antecedent Precipitation Index

An **Antecedent Precipitation Index (API)** is used to represent the influence
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

The model does **not** use directly observed flood water levels as the
training label.

Instead, `flood_label` is constructed as a **proxy for extreme-rainfall
conditions**.

### Province-Specific P90 Threshold

For each province, the **90th percentile (P90) of daily maximum precipitation
(`tp_max`)** is calculated separately.

```python
province_thresholds = (
    df.groupby('province')['tp_max']
    .quantile(0.90)
)
```

A day is labeled as an extreme-rainfall event when its daily maximum
precipitation reaches or exceeds the P90 threshold of that province:

```python
flood_label = tp_max >= province_specific_p90
```

This creates a **province-specific extreme-rainfall threshold** instead of
using a single fixed rainfall threshold for the entire country.

### Why Province-Specific P90?

Rainfall characteristics differ between regions of Vietnam.

Using the P90 of `tp_max` separately for each province allows the target
definition to account for local rainfall characteristics and avoids applying
the same absolute threshold to regions with substantially different
precipitation regimes.

### Important Limitation

The resulting `flood_label` is an **extreme-rainfall proxy for flood risk**,
not direct ground-truth flood observations.

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

Separate prediction targets are generated for each lead time.

For each province:

```python
target = flood_label.shift(-lead)
```

The project uses three prediction targets:

```text
Lead 1 → Extreme-rainfall event 1 day ahead
Lead 2 → Extreme-rainfall event 2 days ahead
Lead 3 → Extreme-rainfall event 3 days ahead
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

Three models are trained independently for the three prediction lead times.

### Train / Validation / Test Split

A chronological split is used:

```text
70% → Training
15% → Validation
15% → Test
```

A temporal split is used instead of a random split to reduce the risk of
information leakage from future observations into the training process.

### Class Imbalance

Extreme-rainfall events represent approximately **10%** of the processed
dataset.

To address this class imbalance, the training process calculates:

```text
scale_pos_weight = negative_samples / positive_samples
```

The resulting value is passed to XGBoost during training.

---

## Model Evaluation

The trained models are evaluated using:

- **ROC-AUC**
- **Recall**
- **Precision**
- **F1-score**

### Results by Lead Time

| Lead Time | AUC | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 1 day | **0.8621** | 82.01% | 26.33% | 0.3986 |
| 2 days | **0.8292** | 82.91% | 22.30% | 0.3514 |
| 3 days | **0.8222** | **84.63%** | 21.52% | 0.3432 |

The models maintain ROC-AUC values above **0.82** across all three lead
times.

The 3-day model achieves the highest recall at **84.63%**, while the
1-day model achieves the highest AUC at **0.8621**.

The relatively low precision indicates a significant false-alarm trade-off,
which is an important limitation of the current approach.

---

## Prediction Threshold Tuning

The XGBoost models produce prediction probabilities. A probability threshold
is then applied to convert the predicted probability into an operational
warning decision.

The threshold was evaluated using a **Precision-Recall analysis**.

### Threshold Evaluation

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 18.2% | 97.3% | 0.306 |
| 0.25 | 19.2% | 96.0% | 0.320 |
| 0.30 | 20.3% | 94.4% | 0.334 |
| 0.35 | 21.6% | 92.5% | 0.350 |
| **0.396** | **22.9%** | **90.0%** | **0.365** |
| 0.50 | 26.3% | 89.8% | 0.366 |
| 0.655 | 34.4% | 59.3% | **0.436** |

The selected operational threshold is:

```text
0.396
```

At this threshold:

```text
Precision = 22.9%
Recall    = 90.0%
F1        = 0.365
```

The selected threshold provides a **90.0% recall** while maintaining a higher
precision than the lower thresholds evaluated.

For an early-warning application, this represents a deliberate trade-off
between detecting extreme-rainfall events and limiting false alarms.

---

## Prediction Workflow

The trained models are deployed through a separate **FastAPI AI service**.

The system follows a scheduled prediction workflow:

```text
00:30 / 12:30
     |
     v
Retrieve weather data from Open-Meteo
     |
     v
Process and prepare weather features
     |
     v
06:30 / 18:30
     |
     v
Call FastAPI AI service
     |
     v
XGBoost prediction
     |
     v
Return prediction results
     |
     v
Flood-risk assessment & alerting
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

The data is aggregated into daily features before model inference.

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

These artifacts are loaded by the FastAPI AI service for prediction.

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

Spring Boot backend responsible for:

- Authentication and authorization
- Weather data integration
- IoT monitoring
- Flood-risk assessment
- Citizen alerting
- SOS rescue requests
- Rescue coordination and dispatch
- Notifications

https://github.com/khanhan0603/flood-alert

### FastAPI Prediction Service

FastAPI service responsible for loading the trained XGBoost models and
providing prediction APIs consumed by the Spring Boot backend.

https://github.com/khanhan0603/ai-server-flood-alert

### Research Reference

**ECMWF Code for Earth - `ml_flood`**

https://github.com/ECMWFCode4Earth/ml_flood

This project adapts research ideas and data-processing concepts from the
original `ml_flood` project while implementing an independent
Vietnam-focused machine-learning pipeline.

---

## Limitations

The current approach has several limitations:

- The target label is an extreme-rainfall proxy rather than observed flood
  events.
- Precision is relatively low because the current approach prioritizes
  recall for early warning.
- Independent validation against observed flood events is still required.
- Realtime feature processing needs to remain consistent with the training
  feature pipeline.
- Additional hydrological and ground-truth flood observations could improve
  the reliability of flood-risk prediction.

---

## Future Improvements

Potential improvements include:

- Validating predictions against independent flood observations.
- Incorporating river water-level and discharge data.
- Improving realtime and training feature consistency.
- Evaluating additional machine-learning algorithms.
- Optimizing warning thresholds for different operational requirements.
- Expanding the training dataset and incorporating additional environmental
  predictors.

---

## Acknowledgement

This project acknowledges the research direction and open-source work of the
**ECMWF Code for Earth `ml_flood` project**.

The original research project can be found at:

https://github.com/ECMWFCode4Earth/ml_flood
