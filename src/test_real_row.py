# src/test_real_row.py
import joblib
import pandas as pd
import numpy as np
import requests

FEATURES  = joblib.load("models/feature_cols.pkl")
THRESHOLD = joblib.load("models/optimal_threshold.pkl")

df = pd.read_parquet("data/processed/flood_dataset_v3.parquet")
df['date'] = pd.to_datetime(df['date'])

# Lấy row flood thật — tháng 10, tỉnh Miền Trung, label=1
flood_rows = df[
    (df['flood_label'] == 1) &
    (df['month'] == 10) &
    (df['province'].isin(['Ha Tinh', 'Quang Binh', 'Quang Nam']))
].dropna(subset=FEATURES)

if len(flood_rows) == 0:
    print("Không tìm thấy flood row — thử điều kiện rộng hơn")
    flood_rows = df[df['flood_label'] == 1].dropna(subset=FEATURES)

row = flood_rows.iloc[0]
print(f"Province: {row['province']}")
print(f"Date:     {row['date'].date()}")
print(f"tp_max:   {row['tp_max']:.2f} mm")
print(f"api:      {row.get('api', 'N/A')}")
print(f"Label:    {row['flood_label']}")

# Predict trực tiếp — không qua API
model1 = joblib.load("models/xgboost_lead1d.pkl")
X = row[FEATURES].values.reshape(1, -1)
prob = model1.predict_proba(X)[0, 1]
print(f"\nDirect predict probability: {prob:.4f}")
print(f"Alert: {prob >= THRESHOLD} (threshold={THRESHOLD})")

# Build JSON để test API
payload = {"province": str(row['province']), "lat": 18.34, "lon": 105.9}
for feat in FEATURES:
    val = row.get(feat)
    if pd.notna(val):
        payload[feat] = float(val)

print(f"\nJSON payload ({len(payload)} fields):")
import json
print(json.dumps(payload, indent=2))

# Gọi API
try:
    resp = requests.post("http://127.0.0.1:8000/predict",
                         json=payload, timeout=5)
    result = resp.json()
    print(f"\nAPI response:")
    print(f"  overall_risk:    {result['overall_risk']}")
    print(f"  max_probability: {result['max_probability']}")
    print(f"\n{'✅ API khớp với direct predict' if abs(result['max_probability'] - prob) < 0.05 else '❌ API KHÁC direct predict — schema bị sai'}")
except Exception as e:
    print(f"API error: {e}")