# ============================= #
# extract_daily_means_from_mswx.py
# ============================= #
import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
from datetime import date, timedelta

# ============================= #
# USER CONFIGURATION
# ============================= #

shapefile = "/content/watershed.shp"
base_path = "/content/drive/MyDrive/MSWX_V100/"
variables = ["Temp", "Wind", "Pres", "SpecHum", "P"]
start_date = date(2017, 1, 1)
end_date = date(2025, 7, 15)
chunk_size = 10
output_dir = "/content/"
dam_name = "Hirakud"

# Read AOI
aoi = gpd.read_file(shapefile).to_crs("EPSG:4326")

records = []
current = start_date

while current <= end_date:
    chunk_dates = [current + timedelta(days=i) for i in range(chunk_size) if current + timedelta(days=i) <= end_date]
    for day in chunk_dates:
        date_str = f"{day.year}{day.timetuple().tm_yday:03d}"
        for var in variables:
            path = os.path.join(base_path, "Past", var, "Daily", f"{date_str}.nc")
            if not os.path.exists(path):
                continue

            try:
                ds = rioxarray.open_rasterio(path)
                ds = ds.rio.write_crs("EPSG:4326", inplace=True)
                clipped = ds.rio.clip(aoi.geometry.apply(mapping), aoi.crs, drop=True)
                data = list(clipped.data_vars.values())[0] if isinstance(clipped, xr.Dataset) else clipped
                arr = data.where(data != -9999, np.nan).values.flatten()
                arr = arr[~np.isnan(arr)]

                if arr.size > 0:
                    mean = np.mean(arr)
                    std = np.std(arr)
                    if std != 0:
                        z_scores = (arr - mean) / std
                        arr = arr[np.abs(z_scores) <= 3]  # Remove outliers
                    if arr.size > 0:
                        records.append({
                            "Date": day.strftime("%Y-%m-%d"),
                            "Variable": var,
                            "Mean_Value": np.mean(arr)
                        })
            except Exception as e:
                print(f"Error processing {path}: {e}")

    current = chunk_dates[-1] + timedelta(days=1)

df_raw = pd.DataFrame(records)
df_pivot = df_raw.pivot_table(index="Date", columns="Variable", values="Mean_Value").reset_index()
desired_cols = ["Date"] + [v for v in variables if v in df_pivot.columns]
df_pivot = df_pivot[desired_cols]

means_csv_path = os.path.join(output_dir, f"{dam_name}_means.csv")
df_pivot.to_csv(means_csv_path, index=False)
print(f" Climate extraction complete: {means_csv_path}")
