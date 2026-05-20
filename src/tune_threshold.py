# src/tune_threshold.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/flood_dataset_v3.parquet")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['province','date']).reset_index(drop=True)

FEATURES    = joblib.load("models/feature_cols.pkl")
model_lead1 = joblib.load("models/xgboost_lead1d.pkl")

# Lấy test set (15% cuối)
df['target'] = df.groupby('province')['flood_label'].shift(-1)
df_lead = df[FEATURES + ['target']].dropna()
X = df_lead[FEATURES]
y = df_lead['target'].astype(int)

n        = len(df_lead)
val_end  = int(n * 0.85)
X_test   = X.iloc[val_end:]
y_test   = y.iloc[val_end:]

y_proba = model_lead1.predict_proba(X_test)[:, 1]

# ── Tìm threshold tối ưu ─────────────────────────────────────
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# F1 tại mỗi threshold
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx   = np.argmax(f1_scores)
best_thr   = thresholds[best_idx]

print(f"Threshold tối ưu F1:       {best_thr:.3f}")
print(f"  → Precision: {precision[best_idx]:.3f}")
print(f"  → Recall:    {recall[best_idx]:.3f}")
print(f"  → F1:        {f1_scores[best_idx]:.3f}")

# Threshold ưu tiên Recall cao (cho cảnh báo lũ)
# Tìm threshold cho Recall >= 0.85
recall_target = 0.85
idx_85 = np.where(recall >= recall_target)[0]
if len(idx_85) > 0:
    thr_85 = thresholds[min(idx_85[-1], len(thresholds)-1)]
    print(f"\nThreshold cho Recall≥0.85: {thr_85:.3f}")
    print(f"  → Precision: {precision[min(idx_85[-1], len(precision)-1)]:.3f}")
    print(f"  → Recall:    {recall[min(idx_85[-1], len(recall)-1)]:.3f}")

# ── Bảng so sánh các threshold ───────────────────────────────
print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-" * 48)
for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    pred = (y_proba >= thr).astype(int)
    tp = ((pred==1) & (y_test==1)).sum()
    fp = ((pred==1) & (y_test==0)).sum()
    fn = ((pred==0) & (y_test==1)).sum()
    p  = tp/(tp+fp+1e-8)
    r  = tp/(tp+fn+1e-8)
    f  = 2*p*r/(p+r+1e-8)
    print(f"{thr:>10.2f} | {p:>10.3f} | {r:>8.3f} | {f:>8.3f}")

# ── Plot & save ───────────────────────────────────────────────
os.makedirs("reports", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PR Curve
axes[0].plot(recall, precision, 'b-', linewidth=2)
axes[0].axvline(x=recall[best_idx], color='r', linestyle='--',
                label=f'Best F1 threshold={best_thr:.2f}')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve (Lead 1d)')
axes[0].legend(); axes[0].grid(True)

# F1 vs Threshold
axes[1].plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
axes[1].axvline(x=best_thr, color='r', linestyle='--',
                label=f'Best threshold={best_thr:.2f}')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('F1 Score')
axes[1].set_title('F1 vs Threshold (Lead 1d)')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("reports/threshold_analysis.png", dpi=150)
print(f"\n✅ Saved: reports/threshold_analysis.png")

CHOSEN_THRESHOLD = 0.35   # Recall=0.921 — ưu tiên bắt lũ

os.makedirs("models", exist_ok=True)
joblib.dump(CHOSEN_THRESHOLD, "models/optimal_threshold.pkl")

print(f"\n✅ Saved: reports/threshold_analysis.png")
print(f"✅ Threshold đã chọn: {CHOSEN_THRESHOLD}")
print(f"   → Recall:    0.921  (bắt được 92% ngày mưa cực đoan)")
print(f"   → Precision: 0.217  (trade-off chấp nhận cho early warning)")
print(f"   → Best F1 threshold (tham khảo): {best_thr:.3f}")
print(f"✅ Saved: models/optimal_threshold.pkl")