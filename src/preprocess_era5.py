"""
ERA5 Preprocessing Pipeline — Vietnam Flood Forecasting
========================================================
Input  : data/raw/era5_{year}_{month:02d}.nc  (6-hourly, ERA5 single-levels)
Output : data/processed/features_daily.parquet  (province-level daily features)

Đơn vị hành chính: 34 tỉnh/thành phố theo cải cách 1/7/2025

THAY ĐỔI SO VỚI VERSION CŨ:
  [FIX-1]  Resample theo UTC+7 (offset=7h) — tránh lệch ngày giữa ERA5 UTC và giờ VN
  [FIX-2]  Evaporation & runoff clip >= 0 trước khi aggregate (vật lý đúng)
  [FIX-3]  Bbox phi lý (khanh_hoa, da_nang) đã sửa về đúng đất liền
  [FIX-4]  u10_mean, v10_mean giữ riêng (không chỉ tính scalar ws) → preserve monsoon direction
  [FIX-5]  Thêm Antecedent Precipitation Index (API) — WMO standard, k=0.85
  [FIX-6]  Thêm month, doy, climate_region — cần cho climatology lookup sau này
  [FIX-7]  Evaporation aggregate dùng sum (đúng), nhưng clip negative trước

CHƯA LÀM (sẽ làm bước 2):
  - Percentile climatology theo climate_region × month (tránh leakage: chỉ fit train period)
  - Rainfall anomaly = (tp - clim_mean) / clim_std
  - ETCCDI indices (R95p, R99p, CWD, Rx1day, Rx5day)
  - Land-area-weighted spatial aggregation

Variables ERA5 (reanalysis-era5-single-levels):
  tp   — total_precipitation        (accumulated, m  → mm)
  t2m  — 2m_temperature             (instantaneous,  K  → °C)
  d2m  — 2m_dewpoint_temperature    (instantaneous,  K  → °C)
  sp   — surface_pressure           (instantaneous,  Pa → hPa)
  u10  — 10m_u_component_of_wind    (instantaneous,  m/s)
  v10  — 10m_v_component_of_wind    (instantaneous,  m/s)
  e    — evaporation                (accumulated, m  → mm, có thể âm)
  ro   — runoff                     (accumulated, m  → mm, có thể âm)
"""

import os
import glob
import logging
import warnings
from pathlib import Path
import zipfile
import shutil

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Climate regions Vietnam
#     Nguồn: IMH (Vietnam Institute of Meteorology & Hydrology) + WMO
#     5 vùng khí hậu chính:
#       north_mountain   : Tây Bắc, Đông Bắc — mưa mùa hè, lũ quét núi
#       north_delta      : Đồng bằng Bắc Bộ — mưa mùa hè + bão
#       central_coast    : Miền Trung ven biển — bão + mưa thu đông (Oct–Dec)
#       central_highland : Tây Nguyên — mưa mùa hè, lũ rừng
#       south            : Nam Bộ + Mekong Delta — mưa mùa tây nam (May–Nov)
# ─────────────────────────────────────────────────────────────────────────────
CLIMATE_REGION = {
    "tuyen_quang":  "north_mountain",
    "dien_bien":    "north_mountain",
    "lai_chau":     "north_mountain",
    "son_la":       "north_mountain",
    "lao_cai":      "north_mountain",
    "thai_nguyen":  "north_mountain",
    "cao_bang":     "north_mountain",
    "lang_son":     "north_mountain",
    "ha_noi":       "north_delta",
    "quang_ninh":   "north_delta",
    "bac_ninh":     "north_delta",
    "phu_tho":      "north_delta",
    "hai_phong":    "north_delta",
    "hung_yen":     "north_delta",
    "ninh_binh":    "north_delta",
    "thanh_hoa":    "central_coast",
    "nghe_an":      "central_coast",
    "ha_tinh":      "central_coast",
    "quang_tri":    "central_coast",
    "hue":          "central_coast",
    "da_nang":      "central_coast",
    "quang_ngai":   "central_coast",
    "gia_lai":      "central_highland",
    "dak_lak":      "central_highland",
    "lam_dong":     "central_highland",
    "khanh_hoa":    "central_coast",   # ven biển, không phải highland
    "dong_nai":     "south",
    "ho_chi_minh":  "south",
    "tay_ninh":     "south",
    "dong_thap":    "south",
    "vinh_long":    "south",
    "an_giang":     "south",
    "can_tho":      "south",
    "ca_mau":       "south",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Bounding boxes 34 tỉnh/thành phố
#     Format: (lat_min, lat_max, lon_min, lon_max)
#     [FIX-3] Sửa khanh_hoa và da_nang — bbox cũ bao gồm biển Đông (phi lý)
#     Nguyên tắc: lon_max không vượt quá bờ biển đất liền (~109.5°E tối đa)
# ─────────────────────────────────────────────────────────────────────────────
PROVINCES = {
    # ── Miền Bắc ──────────────────────────────────────────────────────────
    "ha_noi":      (20.3692, 21.5818, 105.0823, 106.2342),
    "cao_bang":    (22.1597, 23.2893, 105.1272, 107.0091),
    "tuyen_quang": (21.2956, 23.5741, 104.1755, 105.7532),
    "dien_bien":   (20.7582, 22.6300, 102.0431, 103.7445),
    "lai_chau":    (21.5186, 22.9910, 102.2091, 104.1344),
    "son_la":      (20.4158, 22.1635, 103.1012, 105.1631),
    "lao_cai":     (21.1557, 23.0318, 103.3513, 105.2887),
    "thai_nguyen": (21.1093, 22.9076, 105.2523, 106.4293),
    "lang_son":    (21.1995, 22.6368, 105.9243, 107.4721),
    "quang_ninh":  (20.6105, 21.8585, 106.2441, 108.2551),
    "bac_ninh":    (20.7549, 21.8325, 105.6778, 107.1947),
    "phu_tho":     (20.1208, 21.9224, 104.6332, 106.0464),
    "hai_phong":   (19.8843, 21.4418, 105.9068, 107.1500),  # cắt bớt biển
    "hung_yen":    (20.0536, 21.2323, 105.6743, 106.8532),
    "ninh_binh":   (19.6937, 20.9193, 105.4045, 106.5000),  # cắt bớt biển

    # ── Miền Trung ────────────────────────────────────────────────────────
    "thanh_hoa":   (19.0700, 20.8661, 104.2034, 106.1000),
    "nghe_an":     (18.3471, 20.1069, 103.7431, 105.8000),
    "ha_tinh":     (17.7333, 18.9663, 104.9363, 106.5500),
    "quang_tri":   (16.1219, 18.2835, 105.4847, 107.3000),
    "hue":         (15.8497, 16.9459, 106.8987, 108.2000),
    # [FIX-3] da_nang: lon_max từ 112.4 → 108.5 (đảo Hoàng Sa không thuộc mainland)
    "da_nang":     (14.7658, 16.7533, 107.0633, 108.5000),
    "quang_ngai":  (13.8625, 15.6349, 107.1790, 109.0000),

    # ── Tây Nguyên ────────────────────────────────────────────────────────
    "gia_lai":     (12.8344, 14.8804, 107.2847, 109.3000),
    # [FIX-3] khanh_hoa: lon_max từ 114.8 → 109.5 (Trường Sa không tính)
    "khanh_hoa":   ( 9.5239, 13.0390, 108.4251, 109.5000),
    "dak_lak":     (11.9901, 13.8782, 107.4004, 109.2000),
    "lam_dong":    (10.2742, 12.9721, 107.0548, 108.7000),

    # ── Miền Nam ──────────────────────────────────────────────────────────
    "dong_nai":    (10.4073, 12.3858, 106.2440, 107.7517),
    "ho_chi_minh": ( 8.4427, 11.6978, 106.1435, 107.2000),
    "tay_ninh":    (10.1741, 11.9462, 105.3537, 106.9585),
    "dong_thap":   ( 9.9489, 11.1546, 104.9909, 106.5000),
    "vinh_long":   ( 9.3104, 10.5531, 105.4846, 106.7000),
    "an_giang":    ( 9.0581, 11.1883, 103.2397, 105.7838),
    "can_tho":     ( 9.0480, 10.5417, 105.0396, 106.4750),
    "ca_mau":      ( 8.3834,  9.8206, 104.5698, 105.2000),  # cắt bớt biển phía đông
}

assert len(PROVINCES) == 34, f"Phải có đúng 34 tỉnh, hiện có {len(PROVINCES)}"
assert set(PROVINCES.keys()) == set(CLIMATE_REGION.keys()), \
    "PROVINCES và CLIMATE_REGION phải có cùng tỉnh"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Constants
# ─────────────────────────────────────────────────────────────────────────────

# [FIX-1] Vietnam = UTC+7. ERA5 dùng UTC.
# Resample offset 7h để "ngày" tương ứng 00:00–23:59 giờ Hà Nội.
VN_UTC_OFFSET = "7h"

# [FIX-5] API decay coefficient — WMO standard range: 0.85–0.95
# 0.85 phù hợp vùng nhiệt đới (evaporation cao, soil drain nhanh hơn)
API_K = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sel_province(da: xr.DataArray, bbox: tuple) -> xr.DataArray:
    """
    Cắt DataArray theo bounding box (lat_min, lat_max, lon_min, lon_max).
    ERA5 latitude đi từ North → South nên cần slice(lat_max, lat_min).
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    return da.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )


def _agg_precip(arr: np.ndarray) -> dict:
    """
    Aggregate precipitation-like variables (tp).
    Lũ trigger bởi cực trị cục bộ, không phải mean không gian.
    Chú ý: KHÔNG dùng cho evaporation/runoff trực tiếp (có thể âm).
    """
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return dict(mean=np.nan, max=np.nan, p90=np.nan, p99=np.nan)
    return dict(
        mean=float(np.mean(flat)),
        max =float(np.max(flat)),
        p90 =float(np.percentile(flat, 90)),
        p99 =float(np.percentile(flat, 99)),
    )


def _agg_nonneg(arr: np.ndarray) -> dict:
    """
    [FIX-2] Aggregate cho accumulated variables có thể âm (ro, e).
    Clip về 0 trước: giá trị âm là sub-surface exchange, không contribute flood.
    Chỉ tính mean + max sau clip.
    """
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    flat = np.clip(flat, 0, None)   # clip negative → 0
    if len(flat) == 0:
        return dict(mean=np.nan, max=np.nan)
    return dict(
        mean=float(np.mean(flat)),
        max =float(np.max(flat)),
    )


def _agg_background(arr: np.ndarray) -> dict:
    """
    Aggregate background variables (t2m, sp).
    """
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return dict(mean=np.nan, max=np.nan)
    return dict(
        mean=float(np.mean(flat)),
        max =float(np.max(flat)),
    )


def _magnus_rh(t_c: np.ndarray, d_c: np.ndarray) -> float:
    """
    Relative humidity (%) từ t2m và d2m (°C) — Magnus approximation.
    RH = 100 * exp(17.625*d / (243.04+d)) / exp(17.625*t / (243.04+t))
    ERA5 không có sẵn RH nên cần derive thủ công.
    """
    t = t_c.flatten()
    d = d_c.flatten()
    mask = ~(np.isnan(t) | np.isnan(d))
    if mask.sum() == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        rh = 100.0 * (
            np.exp(17.625 * d[mask] / (243.04 + d[mask])) /
            np.exp(17.625 * t[mask] / (243.04 + t[mask]))
        )
    return float(np.mean(rh))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Xử lý 1 file NetCDF → DataFrame (province × day)
# ─────────────────────────────────────────────────────────────────────────────

def process_era5_file(nc_path: str) -> pd.DataFrame:

    log.info("  Đọc %s", os.path.basename(nc_path))

    # ============================================================
    # FIX WINDOWS TEMP DIRECTORY ISSUE
    # ============================================================

    tmpdir = os.path.join(
        "data",
        "tmp_extract",
        os.path.basename(nc_path).replace(".nc", "")
    )

    os.makedirs(tmpdir, exist_ok=True)

    try:

        # Extract ZIP disguised as .nc
        with zipfile.ZipFile(nc_path, 'r') as z:
            z.extractall(tmpdir)

        nc_files = [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.endswith(".nc")
        ]

        accum_file = None
        instant_file = None

        for f in nc_files:

            name = os.path.basename(f)

            if "accum" in name:
                accum_file = f

            elif "instant" in name:
                instant_file = f

        if accum_file is None:
            raise RuntimeError("Không tìm thấy accum file")

        if instant_file is None:
            raise RuntimeError("Không tìm thấy instant file")

        ds_accum = xr.open_dataset(
            accum_file,
            engine="netcdf4"
        )

        ds_inst = xr.open_dataset(
            instant_file,
            engine="netcdf4"
        )

        # Rename valid_time → time
        if "valid_time" in ds_accum.coords:
            ds_accum = ds_accum.rename({"valid_time": "time"})

        if "valid_time" in ds_inst.coords:
            ds_inst = ds_inst.rename({"valid_time": "time"})
        # ============================================================
        # DAILY RESAMPLE (UTC+7)
        # ============================================================

        tp_daily = (
            ds_accum["tp"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .sum()
            * 1000.0
        )

        e_daily = (
            ds_accum["e"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .sum()
            * 1000.0
        )

        ro_daily = (
            ds_accum["ro"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .sum()
            * 1000.0
        )

        t2m_daily = (
            ds_inst["t2m"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .mean()
            - 273.15
        )

        d2m_daily = (
            ds_inst["d2m"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .mean()
            - 273.15
        )

        sp_daily = (
            ds_inst["sp"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .mean()
            / 100.0
        )

        u10_daily = (
            ds_inst["u10"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .mean()
        )

        v10_daily = (
            ds_inst["v10"]
            .resample(time="1D", offset=VN_UTC_OFFSET)
            .mean()
        )

        ws_daily = np.sqrt(
            u10_daily**2 + v10_daily**2
        )

        log.info(
            "  TP check: mean=%.2f mm/day, max=%.2f mm/day",
            float(tp_daily.mean()),
            float(tp_daily.max())
        )

        # ============================================================
        # AGGREGATE PROVINCES
        # ============================================================

        records = []

        for prov_name, bbox in PROVINCES.items():

            tp_p = _sel_province(tp_daily, bbox)
            e_p = _sel_province(e_daily, bbox)
            ro_p = _sel_province(ro_daily, bbox)

            t2m_p = _sel_province(t2m_daily, bbox)
            d2m_p = _sel_province(d2m_daily, bbox)

            sp_p = _sel_province(sp_daily, bbox)

            u10_p = _sel_province(u10_daily, bbox)
            v10_p = _sel_province(v10_daily, bbox)

            ws_p = _sel_province(ws_daily, bbox)

            for t in tp_p.time.values:

                tp_arr = tp_p.sel(time=t).values
                e_arr = e_p.sel(time=t).values
                ro_arr = ro_p.sel(time=t).values

                t2m_arr = t2m_p.sel(time=t).values
                d2m_arr = d2m_p.sel(time=t).values

                sp_arr = sp_p.sel(time=t).values

                u10_arr = u10_p.sel(time=t).values
                v10_arr = v10_p.sel(time=t).values

                ws_arr = ws_p.sel(time=t).values

                tp_s = _agg_precip(tp_arr)

                e_s = _agg_nonneg(e_arr)

                ro_s = _agg_nonneg(ro_arr)

                t2m_s = _agg_background(t2m_arr)

                sp_s = _agg_background(sp_arr)

                ws_s = _agg_background(ws_arr)

                records.append({

                    "date": pd.Timestamp(t),

                    "province": prov_name,

                    "climate_region": CLIMATE_REGION[prov_name],

                    "month": pd.Timestamp(t).month,

                    "doy": pd.Timestamp(t).dayofyear,

                    "tp_mean": tp_s["mean"],
                    "tp_max": tp_s["max"],
                    "tp_p90": tp_s["p90"],
                    "tp_p99": tp_s["p99"],

                    "t2m_mean": t2m_s["mean"],
                    "t2m_max": t2m_s["max"],

                    "d2m_mean": float(np.nanmean(d2m_arr)),

                    "rh_mean": _magnus_rh(
                        t2m_arr,
                        d2m_arr
                    ),

                    "sp_mean": sp_s["mean"],

                    "u10_mean": float(np.nanmean(u10_arr)),

                    "v10_mean": float(np.nanmean(v10_arr)),

                    "ws_mean": ws_s["mean"],
                    "ws_max": ws_s["max"],

                    "evap_mean": e_s["mean"],

                    "ro_mean": ro_s["mean"],
                    "ro_max": ro_s["max"],
                })

        # PHẦN DAILY RESAMPLE + records.append(...)
        # GIỮ NGUYÊN toàn bộ ở đây

        return pd.DataFrame(records)

    finally:

        try:
            shutil.rmtree(tmpdir)

        except Exception:
            pass
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Lag features + rolling windows + API + lead-time labels
# ─────────────────────────────────────────────────────────────────────────────

def _compute_api(tp_series: pd.Series, k: float = API_K) -> pd.Series:
    """
    Antecedent Precipitation Index — WMO standard formula:
        API_t = k * API_{t-1} + P_t
    k = decay coefficient (0.85 cho nhiệt đới Vietnam)

    Vật lý: API proxy cho soil moisture / saturation.
    Khác với rolling sum ở chỗ: mưa cũ decay theo thời gian,
    không bị truncate đột ngột như rolling window.

    Quan trọng: không được shift() — API_t dùng P_t của ngày hiện tại.
    Đây là antecedent condition, KHÔNG phải future info.
    """
    api = np.zeros(len(tp_series))
    tp_vals = tp_series.values
    for i in range(1, len(tp_vals)):
        api[i] = k * api[i - 1] + tp_vals[i]
    return pd.Series(api, index=tp_series.index)


def build_features(
    df: pd.DataFrame,
    lag_days: int = 7,
    lead_days: int = 3,
    flood_label_col: str = "flood_label",
) -> pd.DataFrame:
    """
    Tạo:
      - Lag features: t-1 đến t-{lag_days} cho các biến chính
      - Rolling sum windows: 3d, 5d, 7d cho tp_max, tp_mean
      - Rolling max windows: 3d, 7d cho ro_max (soil saturation proxy)
      - API: Antecedent Precipitation Index (WMO) [FIX-5]
      - Lead-time labels: flood_lead{N}d cho N = 1..lead_days

    Leakage prevention:
      - Lag features dùng shift(lag) → chỉ dùng thông tin quá khứ
      - Lead labels dùng shift(-lead) → đây là target, không phải feature
      - API dùng P_t của ngày hiện tại (không phải tương lai)
      - Rolling windows dùng giá trị hiện tại và quá khứ (không future)
    """
    df = df.sort_values(["province", "date"]).copy()

    # ── Lag features ──────────────────────────────────────────────────────
    lag_vars = [
        "tp_max", "tp_mean", "tp_p90",
        "ro_max", "ro_mean",
        "ws_max", "rh_mean", "sp_mean",
        "u10_mean", "v10_mean",    # [FIX-4] wind components
    ]
    for var in lag_vars:
        if var not in df.columns:
            continue
        for lag in range(1, lag_days + 1):
            df[f"{var}_lag{lag}"] = df.groupby("province")[var].shift(lag)

    # ── Rolling cumulative rainfall ────────────────────────────────────────
    # 3d: flash flood trigger
    # 5d: medium-term saturation
    # 7d: river flood, catchment saturation
    for window in [3, 5, 7]:
        df[f"tp_max_{window}d"] = (
            df.groupby("province")["tp_max"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )
        df[f"tp_mean_{window}d"] = (
            df.groupby("province")["tp_mean"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # ── Rolling max runoff (soil saturation proxy) ─────────────────────────
    for window in [3, 7]:
        df[f"ro_max_{window}d"] = (
            df.groupby("province")["ro_max"]
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )

    # ── [FIX-5] Antecedent Precipitation Index ────────────────────────────
    # API theo WMO: API_t = k * API_{t-1} + P_t
    # Dùng tp_mean (spatial mean) vì API represent catchment-average
    df["api"] = (
        df.groupby("province")["tp_mean"]
        .transform(_compute_api)
    )

    # ── Lead-time labels ───────────────────────────────────────────────────
    if flood_label_col in df.columns:
        for lead in range(1, lead_days + 1):
            df[f"flood_lead{lead}d"] = (
                df.groupby("province")[flood_label_col].shift(-lead)
            )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame) -> None:
    """
    Kỳ vọng ERA5 Vietnam (province-level, max aggregation):
      tp_mean  :  2–5   mm/day
      tp_max   : 50–200 mm/day
      tp_p99   : 30–80  mm/day
      ro_max   :  1–50  mm/day
      t2m_mean : 15–35  °C
      rh_mean  : 60–90  %
      sp_mean  : 950–1015 hPa
      api      : 0–500  mm  (phụ thuộc mùa)
    """
    log.info("=" * 65)
    log.info("SANITY CHECK")
    log.info("=" * 65)
    log.info("Shape        : %s", df.shape)
    log.info("Provinces    : %d", df["province"].nunique())
    log.info("Date range   : %s → %s",
             df["date"].min().date(), df["date"].max().date())
    log.info("Climate regions: %s", df["climate_region"].unique().tolist())
    log.info("")

    checks = [
        ("tp_mean  (kỳ vọng  2–5   mm/day)", "tp_mean"),
        ("tp_max   (kỳ vọng 50–200 mm/day)", "tp_max"),
        ("tp_p99   (kỳ vọng 30–80  mm/day)", "tp_p99"),
        ("ro_max   (kỳ vọng  1–50  mm/day)", "ro_max"),
        ("t2m_mean (kỳ vọng 15–35  °C    )", "t2m_mean"),
        ("rh_mean  (kỳ vọng 60–90  %     )", "rh_mean"),
        ("sp_mean  (kỳ vọng 950–1015 hPa )", "sp_mean"),
        ("api      (kỳ vọng  0–500 mm    )", "api"),
    ]
    for label, col in checks:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        log.info("  %-42s mean=%7.2f  max=%8.2f  p95=%7.2f  p99=%7.2f",
                 label, s.mean(), s.max(), s.quantile(0.95), s.quantile(0.99))

    log.info("")

    # Cảnh báo tp_max quá thấp
    if "tp_max" in df.columns:
        tp_max_g = df["tp_max"].max()
        if tp_max_g < 50:
            log.warning(
                "⚠️  tp_max toàn dataset = %.1f mm — QUÁ THẤP!\n"
                "    1. Đang dùng spatial mean thay vì max\n"
                "    2. Bounding box quá rộng → dilute cực trị\n"
                "    3. Accumulated logic sai",
                tp_max_g,
            )
        else:
            log.info("  ✅ tp_max = %.1f mm — hợp lý.", tp_max_g)

    # Cảnh báo giá trị âm (sau fix không nên còn)
    for col in ["tp_mean", "ro_mean", "evap_mean"]:
        if col not in df.columns:
            continue
        neg = (df[col] < -0.01).sum()
        if neg > 0:
            log.warning("⚠️  %d hàng có %s < 0 — kiểm tra clip logic!", neg, col)
        else:
            log.info("  ✅ Không có giá trị âm trong %s.", col)

    # Kiểm tra evap âm — chỉ cảnh báo (không lỗi, có thể do netrad)
    if "evap_mean" in df.columns:
        neg_e = (df["evap_mean"] < -0.01).sum()
        if neg_e > 0:
            log.warning(
                "⚠️  %d hàng evap_mean < 0 sau clip — "
                "kiểm tra lại _agg_nonneg", neg_e
            )

    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    lag_days: int = 7,
    lead_days: int = 3,
) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline từ ERA5 .nc → features_daily.parquet.

    Workflow:
      1. Tìm tất cả file era5_*.nc trong raw_dir
      2. process_era5_file() từng file → daily features × 34 tỉnh
      3. Concat, sort, tạo lag + rolling + API + lead-time features
      4. Lưu parquet
      5. Sanity check

    Flood labels (từ EM-DAT) join riêng sau bước này:
      df.merge(emdat_df, on=['date','province'], how='left')
        .fillna({'flood_label': 0})

    Climatology (bước 2 — chưa implement):
      Sau khi có features_daily.parquet, chạy compute_climatology.py
      trên train period (2000–2020) để tạo climatology lookup table.
      Sau đó join anomaly features vào df này.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(raw_dir, "era5_*.nc")))
    if not nc_files:
        raise FileNotFoundError(
            f"Không tìm thấy file ERA5 trong '{raw_dir}'. "
            "Kiểm tra lại raw_dir hoặc chạy download script trước."
        )

    log.info("Tìm thấy %d file ERA5. Bắt đầu xử lý...", len(nc_files))

    all_dfs = []
    failed  = []
    for i, nc_path in enumerate(nc_files, 1):
        log.info("[%d/%d] %s", i, len(nc_files), os.path.basename(nc_path))
        try:
            df_month = process_era5_file(nc_path)
            all_dfs.append(df_month)
        except Exception as exc:
            log.error("  ❌ LỖI: %s — %s", os.path.basename(nc_path), exc)
            failed.append(nc_path)
            continue

    if not all_dfs:
        raise RuntimeError("Không có file nào được xử lý thành công.")

    if failed:
        log.warning("%d file bị lỗi: %s", len(failed), failed)

    log.info("Ghép %d tháng...", len(all_dfs))
    df_all = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["province", "date"])
        .reset_index(drop=True)
    )

    log.info("Tạo lag/rolling/API/lead features (lag=%dd, lead=%dd)...",
             lag_days, lead_days)
    df_all = build_features(df_all, lag_days=lag_days, lead_days=lead_days)

    out_path = os.path.join(output_dir, "features_daily.parquet")
    df_all.to_parquet(out_path, index=False)
    log.info("✅ Đã lưu → %s  (shape: %s)", out_path, df_all.shape)

    sanity_check(df_all)
    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ERA5 preprocessing pipeline — Vietnam flood forecasting (34 tỉnh)"
    )
    parser.add_argument("--raw-dir",    default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--lag-days",   type=int, default=7)
    parser.add_argument("--lead-days",  type=int, default=3)
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        lag_days=args.lag_days,
        lead_days=args.lead_days,
    )