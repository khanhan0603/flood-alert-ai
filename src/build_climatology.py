# src/build_climatology.py

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/processed/features_daily.parquet"

OUTPUT_PATH = "data/processed/climatology.parquet"

TRAIN_END_DATE = "2020-01-01"

# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("BUILD VIETNAM FLOOD CLIMATOLOGY")
print("=" * 60)

df = pd.read_parquet(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

print(f"\nInput shape: {df.shape}")

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

required_cols = [
    "province",
    "climate_region",
    "date",
    "tp_mean",
    "tp_max",
    "ro_mean",
    "ro_max",
]

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Thiếu columns: {missing}"
    )

# ============================================================
# TRAIN YEARS ONLY
# ============================================================

train_df = df[
    df["date"] < TRAIN_END_DATE
].copy()

print(
    f"\nTrain climatology shape: "
    f"{train_df.shape}"
)

print(
    f"Train range: "
    f"{train_df['date'].min().date()} → "
    f"{train_df['date'].max().date()}"
)

# ============================================================
# MONTH
# ============================================================

train_df["month"] = (
    train_df["date"]
    .dt.month
)

# ============================================================
# VARIABLES
# ============================================================

CLIMATE_VARS = [
    "tp_mean",
    "tp_max",
    "ro_mean",
    "ro_max",
]

# ============================================================
# BUILD CLIMATOLOGY
# ============================================================

agg_dict = {}

for var in CLIMATE_VARS:

    agg_dict[var] = [
        "mean",
        "std",
        lambda x: np.percentile(x, 90),
        lambda x: np.percentile(x, 95),
        lambda x: np.percentile(x, 99),
    ]

print("\nBuilding regional climatology...")

climatology = (
    train_df
    .groupby([
        "climate_region",
        "month"
    ])
    .agg(agg_dict)
)

# ============================================================
# RENAME COLUMNS
# ============================================================

new_cols = []

for var in CLIMATE_VARS:

    new_cols.extend([
        f"{var}_clim_mean",
        f"{var}_clim_std",
        f"{var}_clim_p90",
        f"{var}_clim_p95",
        f"{var}_clim_p99",
    ])

climatology.columns = new_cols

climatology = climatology.reset_index()

# ============================================================
# CHECK
# ============================================================

print("\nClimatology shape:")
print(climatology.shape)

print("\nClimate regions:")
print(
    climatology["climate_region"]
    .unique()
)

print("\nSample:")
print(
    climatology.head()
)

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

climatology.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CLIMATOLOGY BUILD COMPLETE")
print("=" * 60)

print(f"\nSaved:")
print(OUTPUT_PATH)

print("\nFeatures generated:")

for var in CLIMATE_VARS:

    print(f"\n{var}")

    print(f"  - {var}_clim_mean")
    print(f"  - {var}_clim_std")
    print(f"  - {var}_clim_p90")
    print(f"  - {var}_clim_p95")
    print(f"  - {var}_clim_p99")

print("\nScientific basis:")

print("- ERA5 climatology")
print("- ETCCDI precipitation extremes")
print("- WMO climate normals")
print("- Regional adaptive thresholds")
print("- Hydrological anomaly detection")

print("\nDONE.")