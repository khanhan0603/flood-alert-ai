# src/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)

# ── Load ─────────────────────────────────────────────────────
df = pd.read_parquet("data/processed/flood_dataset_v3.parquet")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['province', 'date']).reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Flood ratio: {df['flood_label'].mean():.4f}")

# ── Features ─────────────────────────────────────────────────
# Loại bỏ non-feature columns
EXCLUDE = [
    'date', 'province', 'region',
    'flood_label', 'risk_level',
    'tp_p90_province',
    # Loại target columns khác nếu có
    'target_lead1', 'target_lead2', 'target_lead3',
]

FEATURES = [c for c in df.columns
            if c not in EXCLUDE
            and df[c].dtype in ['float32', 'float64', 'int32', 'int64']
            and not c.startswith('target_')]

print(f"Features: {len(FEATURES)}")

# ── Config ───────────────────────────────────────────────────
LEAD_DAYS = [1, 2, 3]
results   = {}
os.makedirs("models",   exist_ok=True)
os.makedirs("reports",  exist_ok=True)

# ── Train mỗi lead time ───────────────────────────────────────
for lead in LEAD_DAYS:
    print(f"\n{'='*60}")
    print(f"LEAD TIME = {lead} DAY(S)")
    print(f"{'='*60}")

    # Tạo target — shift theo province để không bị leak giữa tỉnh
    df[f'target'] = (
        df.groupby('province')['flood_label']
        .shift(-lead)
    )

    df_lead = df[FEATURES + ['target']].dropna()
    X = df_lead[FEATURES]
    y = df_lead['target'].astype(int)

    # Temporal split — 70/15/15
    n         = len(df_lead)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end],         y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end],   y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],            y.iloc[val_end:]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Train flood ratio: {y_train.mean():.4f}")

    # Class weight
    pos          = (y_train == 1).sum()
    neg          = (y_train == 0).sum()
    scale_weight = round(neg / pos, 2)
    print(f"scale_pos_weight: {scale_weight}")

    # Model
    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        scale_pos_weight  = scale_weight,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        random_state      = 42,
        eval_metric       = 'auc',
        early_stopping_rounds = 50,
    )

    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        verbose   = 100,
    )

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auc    = roc_auc_score(y_test, y_proba)
    report = classification_report(
                 y_test, y_pred,
                 target_names=['NORMAL', 'EXTREME'],
                 output_dict=True
             )

    recall    = report['EXTREME']['recall']
    precision = report['EXTREME']['precision']
    f1        = report['EXTREME']['f1-score']

    print(f"\nROC-AUC:   {auc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    importance = pd.DataFrame({
        'feature':    FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))

    results[lead] = {
        'auc': auc, 'recall': recall,
        'precision': precision, 'f1': f1
    }

    joblib.dump(model, f"models/xgboost_lead{lead}d.pkl")
    print(f"\n✅ Saved: models/xgboost_lead{lead}d.pkl")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(FEATURES, "models/feature_cols.pkl")

results_df = pd.DataFrame(results).T
results_df.index.name = 'lead_days'
results_df.to_csv("reports/model_results.csv")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Lead':>6} | {'AUC':>7} | {'Recall':>7} | {'F1':>7}")
print("-" * 40)
for lead, r in results.items():
    print(f"{lead:>4}d  | {r['auc']:>7.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f}")

print("\n✅ DONE!")