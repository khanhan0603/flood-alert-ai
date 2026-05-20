# build_labels.py

import pandas as pd
import numpy as np

# ============================================================
# LOAD FEATURES
# ============================================================

print("\n" + "=" * 60)
print("LOAD FEATURES")
print("=" * 60)

df = pd.read_parquet(
    "data/processed/features_daily.parquet"
)

df["date"] = pd.to_datetime(
    df["date"]
)

print(f"Shape: {df.shape}")

print(
    f"Provinces: "
    f"{df['province'].nunique()}"
)

# ============================================================
# LOAD EM-DAT
# ============================================================

print("\nLoading EM-DAT...")

emdat = pd.read_excel(
    "data/raw/emdat_vietnam_flood.xlsx"
)

print(f"EM-DAT events: {len(emdat)}")

# ============================================================
# LOAD CLIMATOLOGY
# ============================================================

print("\nLoading climatology...")

clim = pd.read_parquet(
    "data/processed/climatology.parquet"
)

print(
    f"Climatology shape: "
    f"{clim.shape}"
)

# ============================================================
# ADD MONTH
# ============================================================

df["month"] = df["date"].dt.month

# ============================================================
# MERGE CLIMATOLOGY
# ============================================================

df = df.merge(
    clim,
    on=[
        "climate_region",
        "month",
    ],
    how="left",
)

print(
    f"After climatology merge: "
    f"{df.shape}"
)

# ============================================================
# DATE PARSE
# ============================================================

emdat["start_date"] = pd.to_datetime(
    dict(
        year=emdat["Start Year"],
        month=emdat["Start Month"]
        .fillna(1),
        day=emdat["Start Day"]
        .fillna(1),
    ),
    errors="coerce",
)

emdat["end_date"] = pd.to_datetime(
    dict(
        year=emdat["End Year"]
        .fillna(emdat["Start Year"]),

        month=emdat["End Month"]
        .fillna(emdat["Start Month"])
        .fillna(1),

        day=emdat["End Day"]
        .fillna(emdat["Start Day"])
        .fillna(1),
    ),
    errors="coerce",
)

emdat = emdat.dropna(
    subset=[
        "start_date",
        "end_date",
    ]
)

print(
    f"Date range: "
    f"{emdat['start_date'].min().date()} → "
    f"{emdat['end_date'].max().date()}"
)

# ============================================================
# PROVINCE NORMALIZATION
# ============================================================

PROVINCE_MAP = {

    "Ha Noi": "ha_noi",
    "Cao Bang": "cao_bang",
    "Tuyen Quang": "tuyen_quang",
    "Dien Bien": "dien_bien",
    "Lai Chau": "lai_chau",
    "Son La": "son_la",
    "Lao Cai": "lao_cai",
    "Thai Nguyen": "thai_nguyen",
    "Lang Son": "lang_son",
    "Quang Ninh": "quang_ninh",
    "Bac Ninh": "bac_ninh",
    "Phu Tho": "phu_tho",
    "Hai Phong": "hai_phong",
    "Hung Yen": "hung_yen",
    "Ninh Binh": "ninh_binh",
    "Thanh Hoa": "thanh_hoa",
    "Nghe An": "nghe_an",
    "Ha Tinh": "ha_tinh",
    "Quang Tri": "quang_tri",
    "Hue": "hue",
    "Da Nang": "da_nang",
    "Quang Ngai": "quang_ngai",
    "Gia Lai": "gia_lai",
    "Khanh Hoa": "khanh_hoa",
    "Dak Lak": "dak_lak",
    "Lam Dong": "lam_dong",
    "Dong Nai": "dong_nai",
    "Ho Chi Minh": "ho_chi_minh",
    "Tay Ninh": "tay_ninh",
    "Dong Thap": "dong_thap",
    "Vinh Long": "vinh_long",
    "An Giang": "an_giang",
    "Can Tho": "can_tho",
    "Ca Mau": "ca_mau",
}

# ============================================================
# BINARY FLOOD LABEL
# 0 = NORMAL
# 1 = FLOOD
# ============================================================

df["flood_label"] = 0

province_match = 0
fallback_match = 0

# ============================================================
# SCIENTIFIC FLOOD LABELING
# ============================================================

print(
    "\nBuilding climatology-aware flood labels..."
)

for _, event in emdat.iterrows():

    start = event["start_date"]
    end = event["end_date"]

    # --------------------------------------------------------
    # HYDROLOGICAL BUFFER
    # antecedent rainfall + basin memory
    # --------------------------------------------------------

    start_buf = (
        start - pd.Timedelta(days=5)
    )

    end_buf = (
        end + pd.Timedelta(days=3)
    )

    # --------------------------------------------------------
    # EVENT WINDOW
    # --------------------------------------------------------

    event_df = df[
        (
            df["date"] >= start_buf
        )
        &
        (
            df["date"] <= end_buf
        )
    ].copy()

    # --------------------------------------------------------
    # PROVINCE MATCH
    # --------------------------------------------------------

    province_col = None

    for c in [
        "Admin Units",
        "Location",
        "Province",
    ]:

        if c in event.index:
            province_col = c
            break

    matched = False

    if province_col is not None:

        val = event[province_col]

        if pd.notna(val):

            val = str(val)

            for em_name, p_name in PROVINCE_MAP.items():

                if em_name.lower() in val.lower():

                    event_df = event_df[
                        event_df["province"] == p_name
                    ]

                    matched = True

                    province_match += 1

    if not matched:
        fallback_match += 1

    # --------------------------------------------------------
    # HYDROLOGICAL ANOMALIES
    # --------------------------------------------------------

    tp_z = (

        (
            event_df["tp_max"]
            -
            event_df["tp_max_clim_mean"]
        )

        /

        (
            event_df["tp_max_clim_std"]
            + 1e-6
        )
    )

    ro_z = (

        (
            event_df["ro_max"]
            -
            event_df["ro_max_clim_mean"]
        )

        /

        (
            event_df["ro_max_clim_std"]
            + 1e-6
        )
    )

    api_p75 = (
        event_df["api"]
        .quantile(0.75)
    )

    # ========================================================
    # PHYSICS-INFORMED FLOOD LABEL
    #
    # Scientific basis:
    # - ETCCDI rainfall extremes
    # - climatological anomaly
    # - runoff anomaly
    # - antecedent precipitation
    # - tropical monsoon hydrology
    # ========================================================

    flood_mask = (

        (

            tp_z >= 1.0

        )

        |

        (

            event_df["tp_max"]
            >=
            event_df["tp_max_clim_p90"]

        )

        |

        (

            (

                tp_z >= 2.0

            )

            &

            (

                ro_z >= 1.0

            )

        )

        |

        (

            (

                tp_z >= 1.5

            )

            &

            (

                event_df["api"]
                >=
                api_p75

            )

        )
    )

    flood_idx = event_df[
        flood_mask
    ].index

    df.loc[
        flood_idx,
        "flood_label"
    ] = 1

# ============================================================
# CHECK
# ============================================================

print("\nLabeling summary:")

print(
    f"  Province-matched events: "
    f"{province_match}"
)

print(
    f"  Nationwide fallback: "
    f"{fallback_match}"
)

# ============================================================
# ANOMALY FEATURES
# ============================================================

df["tp_max_zscore"] = (

    (
        df["tp_max"]
        -
        df["tp_max_clim_mean"]
    )

    /

    (
        df["tp_max_clim_std"]
        + 1e-6
    )
)

df["ro_max_zscore"] = (

    (
        df["ro_max"]
        -
        df["ro_max_clim_mean"]
    )

    /

    (
        df["ro_max_clim_std"]
        + 1e-6
    )
)

# ============================================================
# RISK LEVELS
# ============================================================

RISK_MAP = {

    0: "NORMAL",
    1: "FLOOD",
}

df["risk_level"] = (
    df["flood_label"]
    .map(RISK_MAP)
)

# ============================================================
# LABEL QUALITY CHECK
# ============================================================

print("\n=== LABEL QUALITY CHECK ===")

flood_df = df[
    df["flood_label"] == 1
]

normal_df = df[
    df["flood_label"] == 0
]

print(
    f"Flood tp_max: "
    f"{flood_df['tp_max'].mean():.2f} mm"
)

print(
    f"Normal tp_max: "
    f"{normal_df['tp_max'].mean():.2f} mm"
)

print(
    f"Flood ro_max: "
    f"{flood_df['ro_max'].mean():.2f} mm"
)

print(
    f"Normal ro_max: "
    f"{normal_df['ro_max'].mean():.2f} mm"
)

# ============================================================
# DISTRIBUTION
# ============================================================

print("\n=== LABEL DISTRIBUTION ===")

dist = (
    df["risk_level"]
    .value_counts()
)

total = len(df)

for level, count in dist.items():

    print(
        f"  {level:10s}: "
        f"{count:8d} "
        f"({count/total*100:.2f}%)"
    )

# ============================================================
# SAVE
# ============================================================

out_path = (
    "data/processed/flood_dataset.parquet"
)

df.to_parquet(
    out_path,
    index=False
)

print("\nSaved:")
print(out_path)

print(
    f"\nFinal shape: "
    f"{df.shape}"
)

print("\nDONE.")