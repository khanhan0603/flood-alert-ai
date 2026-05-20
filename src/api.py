# src/api.py  ← đặt ở đây cho nhất quán với uvicorn src.api:app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import joblib
import numpy as np
from datetime import datetime
import sys, os

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.weather_fetcher import get_prediction_payload  # ← import ở ĐẦU FILE

app = FastAPI(title="Vietnam Flood Early Warning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Load models ───────────────────────────────────────────────
models = {
    lead: joblib.load(f"models/xgboost_lead{lead}d.pkl")
    for lead in [1, 2, 3]
}
FEATURES  = joblib.load("models/feature_cols.pkl")
THRESHOLD = float(joblib.load("models/optimal_threshold.pkl"))
print(f"✅ Features: {len(FEATURES)} | Threshold: {THRESHOLD}")

RISK_META = {
    "HIGH":   {"color": "red",    "emoji": "🔴"},
    "MEDIUM": {"color": "orange", "emoji": "🟡"},
    "LOW":    {"color": "green",  "emoji": "🟢"},
}

def prob_to_risk(prob: float) -> str:
    if prob >= THRESHOLD * 1.8: return "HIGH"
    if prob >= THRESHOLD:       return "MEDIUM"
    return "LOW"

def _run_predict(payload: dict) -> dict:
    province = payload.get("province", "Unknown")
    lat      = payload.get("lat", 0.0)
    lon      = payload.get("lon", 0.0)

    vec, missing = [], []
    for feat in FEATURES:
        val = payload.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
            missing.append(feat)
        vec.append(float(val))

    if missing:
        print(f"⚠️  {len(missing)} features fallback 0")

    X        = np.array(vec).reshape(1, -1)
    forecast = {}

    for lead, model in models.items():
        prob = float(model.predict_proba(X)[0, 1])
        risk = prob_to_risk(prob)
        forecast[f"day_{lead}"] = {
            "lead_days":   lead,
            "probability": round(prob, 4),
            "risk_level":  risk,
            "alert":       prob >= THRESHOLD,
            "color":       RISK_META[risk]["color"],
        }

    max_prob     = max(v["probability"] for v in forecast.values())
    overall_risk = prob_to_risk(max_prob)
    meta         = RISK_META[overall_risk]

    return {
        "timestamp":        datetime.now().isoformat(),
        "province":         province,
        "lat":              lat,
        "lon":              lon,
        "overall_risk":     overall_risk,
        "color":            meta["color"],
        "emoji":            meta["emoji"],
        "max_probability":  round(max_prob, 4),
        "threshold":        THRESHOLD,
        "features_used":    len(FEATURES) - len(missing),
        "features_missing": len(missing),
        "forecast":         forecast,
    }

# ── Endpoints ─────────────────────────────────────────────────
@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """Predict với đầy đủ 97 features."""
    return _run_predict(payload)

@app.get("/predict/realtime")
def predict_realtime(lat: float, lon: float, province: str = "unknown"):
    """
    Tự động fetch Open-Meteo → predict.
    Swagger: điền lat, lon, province vào query params.
    """
    payload = get_prediction_payload(province, lat, lon)
    return _run_predict(payload)

@app.get("/features")
def get_features():
    return {"total": len(FEATURES), "features": FEATURES}

@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES),
            "threshold": THRESHOLD, "models": ["lead1d","lead2d","lead3d"]}
