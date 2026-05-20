# Tai lieu mo hinh du bao nguy co lu Viet Nam

## 1. Nguon goc tu model goc ECMWFCode4Earth/ml_flood

Repo goc `ECMWFCode4Earth/ml_flood` la mot nghien cuu so sanh cac ky thuat machine learning de du bao flood events bang du lieu mo cua ECMWF/Copernicus. README cua repo goc neu ro cach tiep can:

- Dung ERA5 lam predictor.
- Dung GloFAS reanalysis/forecast rerun hoac severe event catalogue lam predictand/doi tuong danh gia.
- Thu cac model ML nhu Linear Regression, Support Vector Regression, Gradient Boosting va Neural Net.
- Co tu duy hydrology/physics-informed: mua o thuong nguon, dong chay, do tre thoi gian va tac dong cuc bo.

He thong `vietnam_flood_system` ke thua y tuong khoa hoc chinh nay: dung du lieu khi tuong ERA5, bien thuy van lien quan den mua/runoff, lag theo thoi gian va rolling window de du bao nguy co flood/extreme-rainfall trong tuong lai ngan han.

Tuy nhien, he thong hien tai khong copy truc tiep model goc. No da duoc chuyen thanh bai toan classification cho Viet Nam:

- Model: `XGBClassifier`.
- Output: xac suat canh bao cho lead 1, 2, 3 ngay.
- Don vi khong gian: 34 tinh/thanh.
- Dataset train: `data/processed/flood_dataset_v3.parquet`.

Chung cu trong code:

- `ml_flood/README.md`: mo ta ERA5, GloFAS, ML techniques.
- `vietnam_flood_system/src/train_model.py`: train XGBoost classifier cho 3 lead days.

## 2. Du lieu va bien dau vao

Du lieu lich su duoc tai bang ERA5 single-levels tu nam 2000 den 2023, voi tan suat 6 gio/luc. Bounding box dung cho Viet Nam: `[23.5, 102.0, 8.5, 110.0]`.

Bien ERA5 duoc tai:

- `total_precipitation`
- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `evaporation`
- `runoff`

Chung cu: `vietnam_flood_system/src/download_era5.py`.

Pipeline preprocess trong `preprocess_era5.py`:

- Chuyen ERA5 UTC sang ngay Viet Nam bang offset UTC+7.
- Tong hop theo 34 tinh/thanh.
- Tao bien daily: `tp_mean`, `tp_max`, `tp_p90`, `tp_p99`, `t2m_mean`, `t2m_max`, `d2m_mean`, `rh_mean`, `sp_mean`, `u10_mean`, `v10_mean`, `ws_mean`, `ws_max`, `evap_mean`, `ro_mean`, `ro_max`.
- Tao lag 1-7 ngay cho cac bien mua, runoff, gio, do am, ap suat.
- Tao rolling windows 3/5/7 ngay cho mua va runoff.
- Tao Antecedent Precipitation Index:

```text
API_t = k * API_{t-1} + P_t
```

Trong training, `k = 0.85`.

Dataset da doc bang moi truong `C:\conda_envs\flood_vn\python.exe`:

- `flood_dataset_v3.parquet`: 307,836 dong, 103 cot.
- Date range: 1999-12-31 07:00:00 den 2023-12-31 07:00:00.
- So tinh: 34.
- So feature dung train: 97.
- Flood ratio: 0.10008, tuc khoang 10.01%.

## 3. Phuong phap xac dinh nguong lu

Can noi chinh xac: trong version `v3`, nhan `flood_label` hien tai khong phai la muc nuoc lu quan trac truc tiep. Nhan nay duoc tao nhu mot proxy cho ngay mua cuc doan theo nguong percentile 90 cua `tp_max` rieng theo tung tinh.

Cong thuc trong `build_labels_v3.py`:

```python
province_thresholds = (
    df.groupby('province')['tp_max']
    .quantile(0.90)
    .rename('tp_p90_province')
    .reset_index()
)

df['flood_label'] = (df['tp_max'] >= df['tp_p90_province']).astype(int)
```

Y nghia khoa hoc:

- Dung percentile threshold la cach pho bien trong phan tich extreme precipitation.
- Dung nguong theo tung tinh giup thich nghi voi khac biet khi hau dia phuong. Mien Trung, mien nui phia Bac va Dong bang song Cuu Long co che do mua khac nhau, nen mot nguong mm/toan quoc se kem hop ly.
- Nguong p90 tao ra nhan "extreme rainfall day" khoang 10% moi tinh.

Gioi han khoa hoc:

- Day la nhan proxy cho nguy co lu, khong phai xac nhan lu that.
- De ket luan model predict lu that chinh xac, can validate them bang du lieu doc lap: muc nuoc song, discharge, ban do ngap, bao cao thiet hai, hoac GloFAS discharge threshold.

## 4. Train model

File train: `vietnam_flood_system/src/train_model.py`.

Cach train:

- Load `data/processed/flood_dataset_v3.parquet`.
- Loai bo cot khong phai feature: `date`, `province`, `region`, `flood_label`, `risk_level`, `tp_p90_province`, target columns.
- Tao target theo lead day:

```python
df[f'target'] = df.groupby('province')['flood_label'].shift(-lead)
```

- Train rieng 3 model cho `lead = 1, 2, 3`.
- Split theo thoi gian 70/15/15.
- Xu ly imbalance bang `scale_pos_weight = neg / pos`.
- Model:

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
    eval_metric='auc',
    early_stopping_rounds=50,
)
```

Artefacts duoc luu:

- `models/xgboost_lead1d.pkl`
- `models/xgboost_lead2d.pkl`
- `models/xgboost_lead3d.pkl`
- `models/feature_cols.pkl`
- `models/optimal_threshold.pkl`

## 5. Ket qua test model

Ket qua tu `reports/model_results.csv`:

| Lead | AUC | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 1 ngay | 0.8645 | 0.8121 | 0.2659 | 0.4006 |
| 2 ngay | 0.8327 | 0.8267 | 0.2260 | 0.3550 |
| 3 ngay | 0.8235 | 0.8311 | 0.2179 | 0.3453 |

Y nghia:

- AUC > 0.82 cho ca 3 lead days cho thay model co kha nang phan biet ngay extreme va normal tot hon random ro rang.
- Recall cao hon precision. Dieu nay phu hop voi early warning, vi muc tieu la bat duoc nhieu ngay nguy co cao, chap nhan false alarm.
- Precision thap nghia la nhieu canh bao co the khong trung nhan extreme. Can trinh bay ro neu dua vao ung dung thuc te.

## 6. Chon threshold canh bao

File: `vietnam_flood_system/src/tune_threshold.py`.

Threshold duoc tune bang Precision-Recall curve tren test set lead 1d.

Ket qua chay lai:

- Best F1 threshold: 0.668.
- Tai best F1: Precision 0.362, Recall 0.568, F1 0.442.
- Threshold cho Recall >= 0.85: 0.460.
- Threshold duoc chon trong he thong: 0.35.

Bang so sanh:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.182 | 0.973 | 0.307 |
| 0.25 | 0.193 | 0.961 | 0.322 |
| 0.30 | 0.204 | 0.946 | 0.336 |
| 0.35 | 0.217 | 0.921 | 0.352 |
| 0.40 | 0.232 | 0.896 | 0.369 |
| 0.50 | 0.266 | 0.812 | 0.401 |

Ly do chon 0.35:

- Uu tien recall cao trong bai toan canh bao som.
- Bat duoc khoang 92% ngay extreme-rainfall trong test threshold tuning.
- Trade-off la precision chi khoang 21.7%, tuc co nhieu false alarm.

## 7. Verify lich su

File: `vietnam_flood_system/src/verify_history.py`.

Muc tieu:

- Lay 15% cuoi dataset lam test set.
- Predict bang model lead 1d.
- Dung threshold `0.35`.
- Kiem tra overall metrics va mot so event lich su hard-code.

Ket qua chay lai:

- Test set: 46,135 dong.
- Date range: 2020-05-26 den 2023-12-30.
- AUC: 0.8940.

Classification report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.99 | 0.59 | 0.74 | 40,795 |
| EXTREME | 0.24 | 0.97 | 0.38 | 5,340 |

Accuracy tong: 0.63.

Kiem tra event:

- Lu Mien Trung 2020 tai `ha_tinh`: 14 ngay trong window, 12 ngay label=1, model alert 14 ngay, max probability 0.980, recall event 1.00.
- Lu DBSCL 2021 tai `an_giang`: 61 ngay trong window, 10 ngay label=1, model alert 59 ngay, max probability 0.929, recall event 1.00.
- `quang_binh` va `quang_nam` khong co trong danh sach 34 tinh hien tai nen script khong verify duoc hai event nay.

Ket luan tu verify:

- Model bat duoc cac ngay duoc gan nhan extreme trong cac window da test.
- So ngay alert nhieu hon so ngay label=1, phu hop voi precision thap/recall cao.
- Day van la verify theo nhan proxy, chua phai verify bang quan trac lu doc lap.

## 8. Bien du lieu thoi tiet lay tu API free khi predict realtime

File: `vietnam_flood_system/src/weather_fetcher.py`.

API free dang dung: Open-Meteo, endpoint:

```text
https://api.open-meteo.com/v1/forecast
```

Bien hourly lay tu Open-Meteo:

- `precipitation`
- `temperature_2m`
- `dewpoint_2m`
- `surface_pressure`
- `windspeed_10m`
- `winddirection_10m`
- `relativehumidity_2m`
- `et0_fao_evapotranspiration`

Cach aggregate sang daily:

- Mua: `tp_mean`, `tp_max`, `tp_p90`, `tp_p99`.
- Nhiet do: `t2m_mean`, `t2m_max`.
- Dewpoint: `d2m_mean`.
- Do am: `rh_mean`.
- Ap suat: `sp_mean`.
- Gio: `ws_mean`, `ws_max`, `u10_mean`, `v10_mean`.
- Evapotranspiration: `evap_mean`.

Open-Meteo khong co runoff, nen code uoc luong runoff tu precipitation:

```python
daily["ro_mean"] = daily["tp_mean"] * 0.15
daily["ro_max"]  = daily["tp_max"]  * 0.20
```

Gioi han:

- Runoff realtime la approximation, khong tuong duong ERA5 runoff that.
- Trong training, API dung `k=0.85`, nhung realtime trong `weather_fetcher.py` dang dung `k=0.9`. Day la diem nen dong bo neu muon consistency khoa hoc giua train va inference.

## 9. Cach biet model predict chinh xac

Hien tai co the noi model chinh xac theo 3 muc chung cu:

1. Cross/test theo temporal split:
   - AUC, precision, recall, F1 trong `reports/model_results.csv`.

2. Threshold validation:
   - Precision-Recall curve trong `tune_threshold.py`.
   - Threshold `0.35` duoc chon vi recall cao.

3. Historical window verification:
   - `verify_history.py` kiem tra mot so event lich su trong test period.

Nhung khong duoc noi qua muc:

- Chua co bang chung trong code rang model da verify bang mực nước song, discharge observation, satellite flood extent, ban do ngap, hay thiệt hai thuc te theo ngay/tinh.
- Do do, ket luan dung nhat la: model predict kha tot nguy co `EXTREME rainfall/flood proxy`, chua du bang chung de khang dinh predict chinh xac flood event thuc dia.

## 10. Ket luan ngan gon

He thong `vietnam_flood_system` la ban thich nghi tu y tuong cua `ml_flood`: dung ERA5 va machine learning de du bao nguy co lũ. Khac voi repo goc thien ve discharge/GloFAS, he thong nay train XGBoost classifier tren 34 tinh Viet Nam, voi nhan duoc tao tu nguong p90 cua `tp_max` theo tinh.

Ve mat khoa hoc, cach dung percentile threshold, lag rainfall, rolling rainfall/runoff va API la hop ly cho bai toan early warning. Ket qua test cho thay recall cao, dac biet threshold 0.35 bat duoc nhieu ngay extreme. Tuy nhien precision thap va nhan la proxy, nen can bo sung du lieu quan trac lu doc lap neu muon ket luan ve do chinh xac cua du bao lu that.
