# src/build_labels_v3.py
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/features_daily.parquet")
df['date'] = pd.to_datetime(df['date'])

# ── Threshold theo region — không phải toàn quốc ─────────────
# WHY: p90 của Miền Trung khác p90 của Đồng bằng SCL
REGION_MAP = {
    'north_delta':     ['Ha Noi', 'Hai Phong', 'Hung Yen', 'Nam Dinh',
                        'Ninh Binh', 'Ha Nam', 'Thai Binh', 'Hai Duong',
                        'Bac Ninh', 'Vinh Phuc', 'Bac Giang'],
    'north_mountain':  ['Hoa Binh', 'Son La', 'Lai Chau', 'Lao Cai',
                        'Yen Bai', 'Ha Giang', 'Cao Bang', 'Lang Son',
                        'Bac Kan', 'Thai Nguyen', 'Tuyen Quang', 'Phu Tho'],
    'central_coast':   ['Thanh Hoa', 'Nghe An', 'Ha Tinh', 'Quang Binh',
                        'Quang Tri', 'Thua Thien Hue', 'Da Nang',
                        'Quang Nam', 'Quang Ngai', 'Binh Dinh',
                        'Phu Yen', 'Khanh Hoa'],
    'central_highland':['Kon Tum', 'Gia Lai', 'Dak Lak',
                        'Dak Nong', 'Lam Dong'],
    'south':           ['Ninh Thuan', 'Binh Thuan', 'Binh Phuoc',
                        'Tay Ninh', 'Binh Duong', 'Dong Nai',
                        'Ba Ria Vung Tau', 'Ho Chi Minh',
                        'Long An', 'Tien Giang', 'Ben Tre',
                        'Dong Thap', 'An Giang', 'Vinh Long',
                        'Tra Vinh', 'Can Tho', 'Hau Giang',
                        'Soc Trang', 'Bac Lieu', 'Ca Mau', 'Kien Giang'],
}

province_to_region = {}
for region, provinces in REGION_MAP.items():
    for p in provinces:
        province_to_region[p] = region

df['region'] = df['province'].map(province_to_region).fillna('unknown')

# ── Tính threshold theo từng tỉnh ────────────────────────────
# WHY per-province: tỉnh miền Trung mưa nhiều hơn → threshold khác
province_thresholds = (
    df.groupby('province')['tp_max']
    .quantile(0.90)
    .rename('tp_p90_province')
    .reset_index()
)

df = df.merge(province_thresholds, on='province', how='left')

# ── Label: Extreme Rainfall Day ───────────────────────────────
df['flood_label'] = (df['tp_max'] >= df['tp_p90_province']).astype(int)

# ── Verify: phải đúng ~10% mỗi tỉnh ─────────────────────────
ratio_by_province = df.groupby('province')['flood_label'].mean()
print("Label ratio by province (kỳ vọng ~0.10):")
print(ratio_by_province.describe().round(3))

# ── Sanity check rainfall signal ─────────────────────────────
g = df.groupby('flood_label')['tp_max']
ratio = g.mean()[1] / g.mean()[0]
print(f"\ntp_max — extreme: {g.mean()[1]:.2f} | normal: {g.mean()[0]:.2f} | ratio: {ratio:.1f}x")

# ── Distribution ─────────────────────────────────────────────
print(f"\nLabel distribution:")
print(df['flood_label'].value_counts())
print(f"Flood ratio: {df['flood_label'].mean():.4f}")

df.to_parquet("data/processed/flood_dataset_v3.parquet", index=False)
print(f"\nSaved! Shape: {df.shape}")