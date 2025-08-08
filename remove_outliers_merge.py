import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import os

class WeatherWSEMerger:
    def __init__(self):
        pass

    @staticmethod
    def remove_outliers(df, columns):
        for col in columns:
            if col in df.columns:
                z = np.abs(stats.zscore(df[col], nan_policy='omit'))
                df.loc[z > 3, col] = np.nan
        return df

    def process(self, wse_hourly_csv_path, weather_csv_path, output_dir, dam_name="Dam"):
        # -------------------------
        # Step 1: Load and aggregate hourly WSE to daily mean
        # -------------------------
        df_wse_hourly = pd.read_csv(wse_hourly_csv_path)
        df_wse_hourly.columns = df_wse_hourly.columns.str.strip()

        datetime_col = 'DateTime' if 'DateTime' in df_wse_hourly.columns else df_wse_hourly.columns[0]
        df_wse_hourly[datetime_col] = pd.to_datetime(df_wse_hourly[datetime_col])
        df_wse_hourly['Date'] = df_wse_hourly[datetime_col].dt.date

        if 'WSE' not in df_wse_hourly.columns:
            raise ValueError("Missing 'WSE' column in WSE CSV.")

        df_wse = df_wse_hourly.groupby('Date')['WSE'].mean().reset_index()

        # Optional: Save the daily WSE file
        daily_wse_path = os.path.join(output_dir, f"{dam_name}_daily_wse.csv")
        df_wse.to_csv(daily_wse_path, index=False)

        # -------------------------
        # Step 2: Load and clean weather file
        # -------------------------
        df_weather = pd.read_csv(weather_csv_path)
        df_weather.columns = df_weather.columns.str.strip()

        weather_datetime_col = 'DateTime' if 'DateTime' in df_weather.columns else 'Date'
        df_weather['Date'] = pd.to_datetime(df_weather[weather_datetime_col]).dt.date

        # -------------------------
        # Step 3: Remove outliers
        # -------------------------
        df_wse = self.remove_outliers(df_wse, ['WSE'])

        weather_cols = df_weather.columns.drop('Date')
        df_weather = self.remove_outliers(df_weather, weather_cols)

        # -------------------------
        # Step 4: Interpolation
        # -------------------------
        df_wse = df_wse.sort_values('Date').reset_index(drop=True)
        df_weather = df_weather.sort_values('Date').reset_index(drop=True)

        df_wse['WSE'] = df_wse['WSE'].interpolate(method='nearest', limit_direction='both')

        for col in weather_cols:
            df_weather[col] = df_weather[col].interpolate(method='nearest', limit_direction='both')

        # -------------------------
        # Step 5: Merge and save
        # -------------------------
        merged_df = pd.merge(df_wse, df_weather, on='Date', how='inner')

        merged_csv_path = os.path.join(output_dir, f"{dam_name}_weather_cleaned.csv")
        merged_df.to_csv(merged_csv_path, index=False)

        print(f" WSE aggregated, outliers removed, interpolated, and merged.")
        print(f" Final CSV saved at: {merged_csv_path}")
        print(merged_df.head())

        return merged_df
