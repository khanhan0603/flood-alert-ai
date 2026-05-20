# src/weather_fetcher.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Map Open-Meteo variable → tên trong model
OPENMETEO_VARS = {
    # Hourly variables — lấy hourly rồi aggregate thành daily
    "hourly": [
        "precipitation",           # → tp_mean, tp_max
        "temperature_2m",          # → t2m_mean, t2m_max
        "dewpoint_2m",             # → d2m_mean
        "surface_pressure",        # → sp_mean
        "windspeed_10m",           # → ws_mean, ws_max
        "winddirection_10m",       # → u10, v10
        "relativehumidity_2m",     # → rh_mean
        "et0_fao_evapotranspiration", # → evap_mean
    ]
}

# Tọa độ 34 tỉnh
PROVINCES = {
    "an_giang":        (10.52,  105.12),
    "ha_tinh":         (18.34,  105.90),
    "quang_binh":      (17.47,  106.62),
    "quang_nam":       (15.57,  108.05),
    "quang_ngai":      (15.12,  108.80),
    "thua_thien_hue":  (16.46,  107.60),
    "da_nang":         (16.07,  108.22),
    "binh_dinh":       (13.78,  109.22),
    "ha_noi":          (21.03,  105.85),
    "hoa_binh":        (20.68,  105.34),
    "can_tho":         (10.03,  105.78),
    "dong_thap":       (10.49,  105.63),
    "long_an":         (10.53,  106.41),
    "khanh_hoa":       (12.25,  109.18),
    "nghe_an":         (18.67,  105.68),
    # thêm các tỉnh còn lại...
}


def fetch_weather(lat: float, lon: float, days_back: int = 8) -> pd.DataFrame:
    """
    Lấy data thời tiết từ Open-Meteo.
    days_back=8 → lấy 8 ngày để có đủ 7 ngày history + hôm nay.
    KHÔNG cần API key.
    """
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(OPENMETEO_VARS["hourly"]),
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "timezone":   "Asia/Bangkok",   # GMT+7 — Việt Nam
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Parse hourly → DataFrame
    hourly = data["hourly"]
    df = pd.DataFrame({
        "time":        pd.to_datetime(hourly["time"]),
        "precip":      hourly["precipitation"],
        "temp":        hourly["temperature_2m"],
        "dewpoint":    hourly["dewpoint_2m"],
        "pressure":    hourly["surface_pressure"],
        "windspeed":   hourly["windspeed_10m"],
        "winddir":     hourly["winddirection_10m"],
        "humidity":    hourly["relativehumidity_2m"],
        "evap":        hourly["et0_fao_evapotranspiration"],
    })

    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly → daily, tính đúng các biến model cần."""
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg(
        tp_mean   = ("precip",   "mean"),
        tp_max    = ("precip",   "max"),
        tp_p90    = ("precip",   lambda x: np.percentile(x, 90)),
        tp_p99    = ("precip",   lambda x: np.percentile(x, 99)),
        t2m_mean  = ("temp",     "mean"),
        t2m_max   = ("temp",     "max"),
        d2m_mean  = ("dewpoint", "mean"),
        rh_mean   = ("humidity", "mean"),
        sp_mean   = ("pressure", "mean"),
        ws_mean   = ("windspeed","mean"),
        ws_max    = ("windspeed","max"),
        evap_mean = ("evap",     "sum"),
    ).reset_index()

    # Tính u10, v10 từ windspeed + winddirection
    wind_daily = df.groupby("date").agg(
        ws_mean_raw = ("windspeed", "mean"),
        wd_mean     = ("winddir",   "mean"),
    ).reset_index()

    wd_rad = np.deg2rad(wind_daily["wd_mean"])
    wind_daily["u10_mean"] = -wind_daily["ws_mean_raw"] * np.sin(wd_rad)
    wind_daily["v10_mean"] = -wind_daily["ws_mean_raw"] * np.cos(wd_rad)

    daily = daily.merge(
        wind_daily[["date","u10_mean","v10_mean"]],
        on="date", how="left"
    )

    # ro_mean, ro_max — Open-Meteo không có runoff
    # Estimate từ precip (rough approximation)
    daily["ro_mean"] = daily["tp_mean"] * 0.15
    daily["ro_max"]  = daily["tp_max"]  * 0.20

    daily["date"] = daily["date"].astype(str)
    return daily


def build_features_from_history(daily_df: pd.DataFrame) -> dict:
    """
    Từ 8 ngày daily data → tính đủ features cho ngày cuối cùng.
    """
    df = daily_df.sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Cần ít nhất 2 ngày data")

    # Ngày predict = ngày cuối
    today = df.iloc[-1]
    hist  = df.iloc[:-1]   # 7 ngày trước

    features = {}

    # Current day features
    for col in ["tp_mean","tp_max","tp_p90","tp_p99",
                "t2m_mean","t2m_max","d2m_mean","rh_mean",
                "sp_mean","ws_mean","ws_max","evap_mean",
                "ro_mean","ro_max","u10_mean","v10_mean"]:
        features[col] = float(today.get(col, 0.0))

    # Date features
    dt = pd.to_datetime(today["date"])
    features["month"] = float(dt.month)
    features["doy"]   = float(dt.dayofyear)

    # Lag features (1-7 ngày)
    lag_vars = ["tp_mean","tp_max","tp_p90","ro_max","ro_mean",
                "ws_max","rh_mean","sp_mean","u10_mean","v10_mean"]

    for lag in range(1, 8):
        idx = len(hist) - lag
        for var in lag_vars:
            key = f"{var}_lag{lag}"
            features[key] = float(hist.iloc[idx][var]) if idx >= 0 else 0.0

    # Rolling features
    for w, suffix in [(3,"3d"),(5,"5d"),(7,"7d")]:
        window = df.tail(w + 1).head(w)   # w ngày trước today
        features[f"tp_max_{suffix}"]  = float(window["tp_max"].max())
        features[f"tp_mean_{suffix}"] = float(window["tp_mean"].mean())
        if suffix in ("3d","7d"):
            features[f"ro_max_{suffix}"] = float(window["ro_max"].max())

    # API (Antecedent Precipitation Index, k=0.9)
    api = 0.0
    for _, row in df.iterrows():
        api = 0.9 * api + row["tp_mean"]
    features["api"] = float(api)

    return features


def get_prediction_payload(province: str, lat: float, lon: float) -> dict:
    """
    Pipeline đầy đủ: lat/lon → payload sẵn sàng gửi /predict
    """
    print(f"Fetching weather for {province} ({lat}, {lon})...")

    hourly_df = fetch_weather(lat, lon, days_back=8)
    daily_df  = aggregate_to_daily(hourly_df)
    features  = build_features_from_history(daily_df)

    payload = {
        "province": province,
        "lat":      lat,
        "lon":      lon,
    }
    payload.update(features)

    print(f"  Date:     {daily_df.iloc[-1]['date']}")
    print(f"  tp_max:   {features['tp_max']:.2f} mm")
    print(f"  tp_max_7d:{features['tp_max_7d']:.2f} mm")
    print(f"  api:      {features['api']:.2f}")
    print(f"  Features: {len(features)}")

    return payload


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import requests as req

    # Test 3 tỉnh
    test_provinces = [
        ("ha_tinh",  18.34, 105.90),
        ("an_giang", 10.52, 105.12),
        ("da_nang",  16.07, 108.22),
    ]

    for province, lat, lon in test_provinces:
        print(f"\n{'='*50}")
        payload = get_prediction_payload(province, lat, lon)

        resp   = req.post("http://127.0.0.1:8000/predict",
                          json=payload, timeout=10)
        result = resp.json()

        print(f"  Risk:     {result['overall_risk']} {result['emoji']}")
        print(f"  Prob:     {result['max_probability']}")
        print(f"  Day 1:    {result['forecast']['day_1']['probability']:.4f}")
        print(f"  Day 2:    {result['forecast']['day_2']['probability']:.4f}")
        print(f"  Day 3:    {result['forecast']['day_3']['probability']:.4f}")
# src/weather_fetcher.py — thêm hàm này

def predict_any_location(
    lat: float,
    lon: float,
    name: str = None,
    api_url: str = "http://127.0.0.1:8000/predict"
) -> dict:
    """
    Predict flood risk tại BẤT KỲ tọa độ nào.

    lat, lon: tọa độ bất kỳ trong/gần Việt Nam
    name:     tên hiển thị (tuỳ chọn)
    """
    import requests

    # Tên mặc định nếu không truyền vào
    if name is None:
        name = f"point_{lat:.3f}_{lon:.3f}"

    # Validate tọa độ trong phạm vi VN (rough check)
    if not (8.0 <= lat <= 24.0 and 102.0 <= lon <= 110.0):
        print(f"⚠️  Tọa độ ({lat}, {lon}) ngoài phạm vi Việt Nam")
        print(f"   VN: lat 8-24°N, lon 102-110°E")
        print(f"   Vẫn tiếp tục predict nhưng kết quả kém tin cậy hơn")

    payload = get_prediction_payload(name, lat, lon)

    resp   = requests.post(api_url, json=payload, timeout=10)
    result = resp.json()

    return {
        "name":            name,
        "lat":             lat,
        "lon":             lon,
        "date":            datetime.now().strftime("%Y-%m-%d"),
        "overall_risk":    result["overall_risk"],
        "emoji":           result["emoji"],
        "max_probability": result["max_probability"],
        "forecast": {
            f"day_{i}": result["forecast"][f"day_{i}"]["probability"]
            for i in [1, 2, 3]
        }
    }


# ── Test tọa độ tự do ─────────────────────────────────────────
if __name__ == "__main__":

    points = [
        (18.34,  105.90, "Ha Tinh (có trong training)"),
        (17.50,  106.20, "Giữa Quảng Bình - Hà Tĩnh"),
        (16.123, 107.456,"Tọa độ lẻ Miền Trung"),
        (10.52,  105.12, "An Giang (có trong training)"),
        (10.00,  106.00, "Giữa ĐBSCL"),
    ]

    print(f"{'Tên':35} | {'Risk':8} | {'Prob':8} | D1→D3")
    print("-" * 75)

    for lat, lon, name in points:
        try:
            r = predict_any_location(lat, lon, name)
            d = r["forecast"]
            print(
                f"{name:35} | "
                f"{r['overall_risk']:8} | "
                f"{r['max_probability']:.4f}   | "
                f"{d['day_1']:.3f} → {d['day_2']:.3f} → {d['day_3']:.3f}"
            )
        except Exception as e:
            print(f"{name:35} | ERROR: {e}")