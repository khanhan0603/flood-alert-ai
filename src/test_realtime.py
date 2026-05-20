# src/test_realtime.py
import requests

points = [
    ("ha_tinh",          18.34,  105.90),
    ("giua_qbinh_htinh", 17.50,  106.20),
    ("toa_do_le",        16.123, 107.456),
    ("an_giang",         10.52,  105.12),
    ("giua_dbscl",       10.00,  106.00),
]

print(f"{'Tên':30} | {'Risk':6} | {'Prob':6} | D1 → D2 → D3")
print("-" * 70)

for province, lat, lon in points:
    resp = requests.get(
        "http://127.0.0.1:8000/predict/realtime",
        params={"lat": lat, "lon": lon, "province": province},
        timeout=30
    )
    r = resp.json()
    f = r["forecast"]
    print(
        f"{province:30} | "
        f"{r['overall_risk']:6} {r['emoji']} | "
        f"{r['max_probability']:.4f} | "
        f"{f['day_1']['probability']:.3f} → "
        f"{f['day_2']['probability']:.3f} → "
        f"{f['day_3']['probability']:.3f}"
    )