import cdsapi
import os

c = cdsapi.Client(
    timeout=600,
    retry_max=10,
    sleep_max=120
)

os.makedirs("data/raw", exist_ok=True)

for year in range(2000, 2024):

    for month in range(1, 13):

        output_file = f"data/raw/era5_{year}_{month:02d}.nc"

        if os.path.exists(output_file):
            print(f"{output_file} already exists, skipping...")
            continue

        print(f"Downloading {year}-{month:02d}...")

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'total_precipitation',
                    '2m_temperature',
                    '2m_dewpoint_temperature',
                    'surface_pressure',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'evaporation',
                    'runoff',
                ],
                'year': str(year),
                'month': [f'{month:02d}'],
                'day': [f'{d:02d}' for d in range(1, 32)],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': [23.5, 102.0, 8.5, 110.0],
            },
            output_file
        )

        print(f"{year}-{month:02d} complete!")