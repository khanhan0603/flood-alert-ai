# Tài liệu mô hình dự báo nguy cơ lũ tại Việt Nam

## 1. Nguồn gốc và cơ sở nghiên cứu

Mô hình được phát triển dựa trên việc **kế thừa ý tưởng nghiên cứu và quy trình xử lý dữ liệu** từ dự án mã nguồn mở **ECMWF Code for Earth – `ml_flood`**:

https://github.com/ECMWFCode4Earth/ml_flood

Dự án `ml_flood` nghiên cứu việc sử dụng các kỹ thuật Machine Learning để dự báo các sự kiện lũ dựa trên dữ liệu mở từ ECMWF/Copernicus. Một số định hướng được kế thừa gồm:

- Sử dụng **ERA5** làm nguồn dữ liệu khí tượng.
- Sử dụng các biến liên quan đến lượng mưa và thủy văn.
- Khai thác **độ trễ theo thời gian (lag features)**.
- Sử dụng **rolling window features** để biểu diễn tác động tích lũy của điều kiện thời tiết.
- Kết hợp các biến mang tính thủy văn như **runoff** và **Antecedent Precipitation Index (API)**.
- Xem xét mối quan hệ giữa điều kiện khí tượng hiện tại, điều kiện trước đó và nguy cơ xảy ra sự kiện cực đoan trong tương lai.

Tuy nhiên, mô hình trong dự án này **không sử dụng trực tiếp mô hình được huấn luyện sẵn từ `ml_flood`**. Đây là một pipeline Machine Learning được xây dựng và huấn luyện riêng cho bài toán tại Việt Nam, với cách xác định nhãn, feature engineering và thời gian dự báo được điều chỉnh cho hệ thống cảnh báo sớm.

### Khác biệt chính

| `ml_flood` | Mô hình trong dự án |
|---|---|
| Nghiên cứu dự báo flood events | Dự báo extreme-rainfall events làm proxy cho nguy cơ lũ |
| ERA5 kết hợp dữ liệu thủy văn | ERA5 và các biến khí tượng/thủy văn liên quan |
| Nhiều kỹ thuật Machine Learning | XGBoost classification |
| Hướng nghiên cứu flood/discharge | Early-warning theo extreme rainfall |
| Quy trình nghiên cứu gốc | Quy trình được điều chỉnh cho Việt Nam |
| — | 3 mô hình cho lead time 1, 2 và 3 ngày |
| — | 34 tỉnh/thành |

---

## 2. Dữ liệu và biến đầu vào

### 2.1. Dữ liệu lịch sử

Dữ liệu huấn luyện được xây dựng từ **ERA5 single-level data** cho khu vực Việt Nam.

- Giai đoạn dữ liệu: **2000–2025**
- Tần suất dữ liệu gốc: **6 giờ/lần**
- Phạm vi không gian: Việt Nam
- Đơn vị không gian: **34 tỉnh/thành**
- Quy mô dataset: **330,000+ records**

Các biến ERA5 được sử dụng gồm:

- `total_precipitation`
- `2m_temperature`
- `2m_dewpoint_temperature`
- `surface_pressure`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `evaporation`
- `runoff`

### 2.2. Tiền xử lý

Pipeline tiền xử lý thực hiện:

1. Chuyển timestamp từ UTC sang thời gian Việt Nam bằng **UTC+7**.
2. Ánh xạ dữ liệu vào **34 tỉnh/thành**.
3. Tổng hợp dữ liệu 6 giờ thành các đặc trưng theo ngày.
4. Tạo các đặc trưng thống kê về lượng mưa, nhiệt độ, độ ẩm, áp suất, gió, bốc hơi và runoff.
5. Tạo lag features từ 1 đến 7 ngày.
6. Tạo rolling features với cửa sổ 3, 5 và 7 ngày.
7. Tính Antecedent Precipitation Index (API).

### 2.3. Daily features

Các đặc trưng theo ngày gồm:

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

### 2.4. Lag features

Các biến thời tiết và thủy văn được tạo lag tối đa **7 ngày** nhằm giúp mô hình khai thác điều kiện thời tiết trong những ngày trước đó.

### 2.5. Rolling features

Các rolling windows **3 ngày, 5 ngày và 7 ngày** được sử dụng để biểu diễn tác động tích lũy hoặc kéo dài của lượng mưa và runoff.

### 2.6. Antecedent Precipitation Index

API được sử dụng để biểu diễn ảnh hưởng của lượng mưa trong quá khứ:

```text
API_t = k × API_(t-1) + P_t
```

Trong quá trình huấn luyện:

```text
k = 0.85
```

---

## 3. Xác định nhãn extreme-rainfall

### 3.1. Bản chất của nhãn

Trong phiên bản hiện tại, `flood_label` **không phải là dữ liệu quan trắc mực nước lũ trực tiếp**.

Nhãn được xây dựng như một **proxy cho extreme-rainfall event**, sau đó được sử dụng làm proxy cho nguy cơ lũ.

### 3.2. P90 theo từng tỉnh

Đối với mỗi tỉnh/thành, tính **P90 (90th percentile) của lượng mưa cực đại theo ngày `tp_max` riêng cho tỉnh đó**.

```python
province_thresholds = (
    df.groupby('province')['tp_max']
      .quantile(0.90)
      .rename('tp_p90_province')
      .reset_index()
)
```

Sau đó xác định nhãn:

```python
df['flood_label'] = (
    df['tp_max'] >= df['tp_p90_province']
).astype(int)
```

Như vậy:

```text
flood_label = 1
```

khi lượng mưa cực đại trong ngày đạt hoặc vượt **P90 của chính tỉnh đó**.

### 3.3. Ý nghĩa của P90 theo từng tỉnh

Việc sử dụng ngưỡng P90 riêng cho từng tỉnh giúp phản ánh sự khác biệt về chế độ mưa giữa các khu vực.

Thay vì áp dụng một ngưỡng mm cố định cho toàn Việt Nam, mỗi tỉnh có một ngưỡng extreme-rainfall riêng dựa trên phân bố lượng mưa lịch sử của tỉnh đó.

---

## 4. Xây dựng target cho các lead time

Target được tạo theo từng tỉnh:

```python
df['target'] = (
    df.groupby('province')['flood_label']
      .shift(-lead)
)
```

Ba mô hình được huấn luyện độc lập:

```text
Lead 1 → dự báo extreme-rainfall event trước 1 ngày
Lead 2 → dự báo extreme-rainfall event trước 2 ngày
Lead 3 → dự báo extreme-rainfall event trước 3 ngày
```

---

## 5. Huấn luyện mô hình

### 5.1. Thuật toán

Sử dụng **XGBoost Classifier (`XGBClassifier`)**.

Ba mô hình được huấn luyện riêng cho ba lead time:

```text
XGBoost Lead 1 day
XGBoost Lead 2 days
XGBoost Lead 3 days
```

### 5.2. Temporal split

Dataset được chia theo thứ tự thời gian:

```text
70% → Training
15% → Validation
15% → Test
```

Việc chia theo thời gian được sử dụng thay vì random split nhằm hạn chế việc thông tin từ tương lai xuất hiện trong tập huấn luyện.

### 5.3. Class imbalance

Extreme-rainfall events chiếm khoảng **10%** dataset.

Để xử lý mất cân bằng lớp, sử dụng:

```text
scale_pos_weight = negative_samples / positive_samples
```

---

## 6. Kết quả đánh giá mô hình

Các model được đánh giá bằng:

- **ROC-AUC**
- **Recall**
- **Precision**
- **F1-score**

| Lead Time | ROC-AUC | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| 1 day | **0.8621** | 82.01% | 26.33% | 0.3986 |
| 2 days | **0.8292** | 82.91% | 22.30% | 0.3514 |
| 3 days | **0.8222** | **84.63%** | 21.52% | 0.3432 |

- Model 1 ngày có **ROC-AUC cao nhất: 0.8621**.
- Model 3 ngày có **Recall cao nhất: 84.63%**.
- Precision thấp hơn Recall, phản ánh trade-off giữa khả năng phát hiện event và false alarm.

---

## 7. Ngưỡng cảnh báo

Model trả về xác suất dự báo. Xác suất được chuyển thành quyết định cảnh báo bằng threshold.

Threshold được đánh giá bằng **Precision-Recall analysis** trên test set của model lead 1 day.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 18.2% | 97.3% | 0.307 |
| 0.25 | 19.3% | 96.1% | 0.322 |
| 0.30 | 20.4% | 94.6% | 0.336 |
| **0.35** | **21.7%** | **92.1%** | **0.352** |
| 0.40 | 23.2% | 89.6% | 0.369 |
| 0.50 | 26.6% | 81.2% | 0.401 |

Threshold được chọn trong hệ thống:

```text
0.35
```

Lý do lựa chọn là ưu tiên **Recall cao** cho bài toán early warning. Tại threshold 0.35, Recall đạt **92.1%**, đổi lại Precision ở mức **21.7%**.

**0.35 không phải threshold có F1 cao nhất**; đây là threshold được lựa chọn theo mục tiêu vận hành của hệ thống.

---

## 8. Scheduled Prediction Workflow

Mô hình **không thực hiện realtime prediction liên tục**.

Hệ thống sử dụng quy trình dự báo theo lịch.

### 8.1. Lấy dữ liệu thời tiết

Dữ liệu thời tiết được lấy từ **Open-Meteo** hai lần mỗi ngày:

```text
00:30
12:30
```

Các biến được lấy gồm:

- `precipitation`
- `temperature_2m`
- `dewpoint_2m`
- `surface_pressure`
- `windspeed_10m`
- `winddirection_10m`
- `relativehumidity_2m`
- `et0_fao_evapotranspiration`

### 8.2. Xử lý dữ liệu

```text
Open-Meteo hourly data
        ↓
Daily aggregation
        ↓
Feature processing
        ↓
Model-compatible features
```

### 8.3. Gọi FastAPI AI service

Spring Boot backend gọi **FastAPI AI service** hai lần mỗi ngày:

```text
06:30
18:30
```

Quy trình tổng thể:

```text
00:30 / 12:30
      ↓
Retrieve weather data from Open-Meteo
      ↓
Process weather data
      ↓
Prepare model features
      ↓
06:30 / 18:30
      ↓
Spring Boot calls FastAPI AI service
      ↓
XGBoost prediction
      ↓
Prediction probability
      ↓
Spring Boot flood-risk assessment
      ↓
Alert / notification workflow
```

FastAPI AI service chịu trách nhiệm tải model và thực hiện inference.

---

## 9. Runoff trong prediction

Training sử dụng biến `runoff` từ ERA5.

Open-Meteo không cung cấp cùng biến runoff như ERA5. Trong pipeline prediction, runoff được ước lượng từ precipitation:

```python
daily["ro_mean"] = daily["tp_mean"] * 0.15
daily["ro_max"]  = daily["tp_max"] * 0.20
```

Đây là approximation và **không tương đương với ERA5 runoff thực tế**.

---

## 10. Model Artifacts

Các artifact được sử dụng bởi AI prediction service:

```text
models/
├── xgboost_lead1d.pkl
├── xgboost_lead2d.pkl
├── xgboost_lead3d.pkl
├── feature_cols.pkl
└── optimal_threshold.pkl
```

---

## 11. Historical Verification

Model lead 1 day được kiểm tra trên 15% dữ liệu cuối.

```text
Test samples: 46,135
Date range: 2020-05-26 → 2023-12-30
ROC-AUC: 0.8940
```

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.99 | 0.59 | 0.74 | 40,795 |
| EXTREME | 0.24 | 0.97 | 0.38 | 5,340 |

Accuracy tổng:

```text
0.63
```

Historical windows được kiểm tra gồm:

- **Hà Tĩnh 2020:** 12/14 ngày được gán label = 1; model cảnh báo 14 ngày; maximum probability 0.980; event recall 1.00.
- **An Giang 2021:** 10/61 ngày được gán label = 1; model cảnh báo 59 ngày; maximum probability 0.929; event recall 1.00.

Kết quả cho thấy model phát hiện được các ngày extreme-rainfall theo proxy label trong các historical windows được kiểm tra. Tuy nhiên, số ngày cảnh báo cao hơn số ngày label = 1, phù hợp với precision thấp và recall cao.

---

## 12. Giới hạn của mô hình

Không nên kết luận rằng mô hình đã chứng minh khả năng dự báo chính xác **flood events thực địa**.

Các giới hạn hiện tại:

- Target là **extreme-rainfall proxy**, không phải observed flood label.
- Precision tương đối thấp.
- Runoff trong prediction là approximation.
- Pipeline training và inference cần duy trì tính nhất quán về feature engineering.
- Chưa có validation độc lập với mực nước sông, discharge, flood extent hoặc dữ liệu thiệt hại thực tế.

Vì vậy, kết luận phù hợp nhất là:

> Mô hình có khả năng phân biệt và phát hiện các extreme-rainfall events theo proxy label với ROC-AUC trên 0.82 cho cả ba lead times, đồng thời ưu tiên Recall cao cho mục tiêu cảnh báo sớm. Cần thêm dữ liệu flood observations độc lập để đánh giá khả năng dự báo lũ thực địa.

---

## 13. Kết luận

Đây là một pipeline Machine Learning được xây dựng riêng cho bài toán cảnh báo nguy cơ lũ tại Việt Nam, **kế thừa ý tưởng nghiên cứu và quy trình xử lý dữ liệu từ ECMWF Code for Earth `ml_flood`**, thay vì sử dụng trực tiếp model có sẵn.

Pipeline:

```text
ERA5 historical data
        ↓
UTC → Vietnam time
        ↓
34 provinces/cities
        ↓
Daily aggregation
        ↓
Lag / Rolling / API features
        ↓
Province-specific P90
        ↓
Extreme-rainfall proxy label
        ↓
Temporal 70/15/15 split
        ↓
XGBoost training
        ↓
1 / 2 / 3-day models
        ↓
Model evaluation
        ↓
FastAPI AI service
        ↓
Scheduled prediction
        ↓
Spring Boot flood-risk assessment
```

### Kết quả chính

- **3 XGBoost models** cho lead time 1, 2 và 3 ngày.
- ROC-AUC: **0.8621 / 0.8292 / 0.8222**.
- Recall: **82.01% / 82.91% / 84.63%**.
- Threshold cảnh báo: **0.35**, với Recall **92.1%** trên threshold evaluation.
- Open-Meteo được gọi **2 lần/ngày** lúc **00:30 và 12:30**.
- Spring Boot gọi FastAPI AI service **2 lần/ngày** lúc **06:30 và 18:30**.
- Model được sử dụng như một thành phần trong hệ thống đánh giá nguy cơ và cảnh báo lũ, không phải một hệ thống dự báo flood ground-truth độc lập.
