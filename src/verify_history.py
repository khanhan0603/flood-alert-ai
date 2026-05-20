# src/verify_history.py
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# ── Load ─────────────────────────────────────────────────────
df = pd.read_parquet("data/processed/flood_dataset_v3.parquet")
df['date'] = pd.to_datetime(df['date'])

FEATURES  = joblib.load("models/feature_cols.pkl")
model     = joblib.load("models/xgboost_lead1d.pkl")
THRESHOLD = joblib.load("models/optimal_threshold.pkl")

# ── Sort theo DATE only — không sort theo province ────────────
df = df.sort_values('date').reset_index(drop=True)

# ── Tạo target ────────────────────────────────────────────────
df['target'] = df.groupby('province')['flood_label'].shift(-1)

# ── Keep cols ─────────────────────────────────────────────────
extra_cols = ['target', 'date', 'province', 'flood_label']
extra_cols = [c for c in extra_cols if c not in FEATURES and c in df.columns]
keep_cols  = FEATURES + extra_cols
df_lead    = df[keep_cols].dropna()

# ── Split theo thời gian ──────────────────────────────────────
n       = len(df_lead)
df_test = df_lead.iloc[int(n * 0.85):].copy()

print(f"Test set: {len(df_test)} rows")
print(f"Date range: {df_test['date'].min().date()} → {df_test['date'].max().date()}")
print(f"Provinces: {sorted(df_test['province'].unique())}")
print(f"X shape: {df_test[FEATURES].shape}")

# ── Predict ───────────────────────────────────────────────────
X      = df_test[FEATURES].values
probas = model.predict_proba(X)[:, 1]
preds  = (probas >= THRESHOLD).astype(int)

df_test['probability'] = probas
df_test['predicted']   = preds

# ── Verify đợt lũ lịch sử ────────────────────────────────────
flood_events = [
    ("Lũ Miền Trung 2020", "2020-10-06", "2020-10-20", "ha_tinh"),
    ("Lũ Miền Trung 2020", "2020-10-06", "2020-10-20", "quang_binh"),
    ("Lũ ĐBSCL 2021",      "2021-09-01", "2021-10-31", "an_giang"),
    ("Lũ Miền Trung 2022", "2022-10-01", "2022-11-30", "quang_nam"),
]

print("\n" + "="*65)
print("VERIFY TRÊN CÁC ĐỢT LŨ LỊCH SỬ")
print("="*65)

for event_name, start, end, province in flood_events:
    mask = (
        (df_test['date'] >= start) &
        (df_test['date'] <= end) &
        (df_test['province'] == province)
    )
    sub = df_test[mask]

    if len(sub) == 0:
        # Thử tìm tên tỉnh gần đúng
        similar = [p for p in df_test['province'].unique()
                   if province[:4] in p]
        print(f"\n{event_name} ({province}): Không có data")
        print(f"  Tỉnh gần giống: {similar}")
        continue

    detected     = int(sub['predicted'].sum())
    max_prob     = sub['probability'].max()
    actual_flood = int(sub['flood_label'].sum())

    print(f"\n{event_name} — {province}")
    print(f"  Ngày trong window: {len(sub)}")
    print(f"  Ngày label=1:      {actual_flood}")
    print(f"  Model alert:       {detected} ngày")
    print(f"  Max probability:   {max_prob:.3f}")

    if actual_flood > 0:
        flood_days    = sub[sub['flood_label'] == 1]
        flood_alerted = int(flood_days['predicted'].sum())
        recall_event  = flood_alerted / len(flood_days)
        print(f"  Recall event:      {recall_event:.2f} "
              f"({flood_alerted}/{len(flood_days)})")

        if recall_event >= 0.7:   print(f"  ✅ Bắt được đợt lũ này")
        elif recall_event >= 0.4: print(f"  ⚠️  Bắt được một phần")
        else:                     print(f"  ❌ Bỏ sót đợt lũ này")
    else:
        print(f"  ⚠️  Không có ngày label=1 trong window này")

# ── Overall metrics ───────────────────────────────────────────
print("\n" + "="*65)
print("OVERALL TEST SET METRICS")
print("="*65)

y_true = df_test['target'].astype(int)
y_prob = df_test['probability']
y_pred = df_test['predicted']

print(f"Date range: {df_test['date'].min().date()} → {df_test['date'].max().date()}")
print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
print(classification_report(
    y_true, y_pred,
    target_names=['NORMAL', 'EXTREME']
))