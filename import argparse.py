import argparse
from WeatherWSEMerger import remove_outliers, process

def main():
    parser = argparse.ArgumentParser(
        description="Merge precomputed daily mean WSE and weather CSVs, remove outliers, interpolate missing data, and merge into a cleaned dataset."
    )
    parser.add_argument("--wse_csv", required=True, help="Path to precomputed daily mean WSE CSV.")
    parser.add_argument("--weather_csv", required=True, help="Path to precomputed daily mean weather CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory where the merged CSV will be saved.")
    parser.add_argument("--dam_name", default="Dam", help="Name of the dam/body for output naming.")

    args = parser.parse_args()

    merger = WeatherWSEMerger()
    merger.process(
        wse_csv_path=args.wse_csv,
        weather_csv_path=args.weather_csv,
        output_dir=args.output_dir,
        dam_name=args.dam_name
    )

if __name__ == "__main__":
    main()


def main():
    import pandas as pd
    csv_path = r"D:\new_project\rihand_dam\Rihand_weather.csv"
    output_dir = r"D:\new_project\rihand_dam\EDA_plots"
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    target_col = 'WSE'
    exogenous_cols = ['P', 'Pres', 'SpecHum', 'Temp', 'Wind']
    eda = EDA(df, target_col, exogenous_cols, output_dir, window=30, lags=100)
    eda.full_eda_pipeline()

if __name__ == '__main__':
    main()


from fourier_deterministic_generator import FourierDeterministicGenerator
from sarimax_fourier_forecaster import SarimaxFourierForecaster

# 1️⃣ Generate Fourier + trend
fourier_gen = FourierDeterministicGenerator(freq='A', order=3, add_constant=True, add_trend=True)
X_train_fourier, X_test_fourier = fourier_gen.generate(train.index, len(test))

# 2️⃣ Add exogenous variables
X_train_all = X_train_fourier.copy()
X_train_all['mean_temp'] = train['mean_temperature']
X_train_all['mean_precip'] = train['mean_precip']

X_test_all = X_test_fourier.copy()
X_test_all['mean_temp'] = test['mean_temperature'].values
X_test_all['mean_precip'] = test['mean_precip'].values

# 3️⃣ Fit SARIMAX
forecaster = SarimaxFourierForecaster(order=(2,0,1))
forecaster.fit(y_train=train['WSE'], X_train_all=X_train_all)

# 4️⃣ Predict
forecast_df = forecaster.predict(steps=len(test), X_future=X_test_all)


from fourier_deterministic_generator import FourierDeterministicGenerator
from unobserved_components_fourier_forecaster import UnobservedComponentsFourierForecaster

# 1️⃣ Generate Fourier + trend features
fourier_gen = FourierDeterministicGenerator(freq='A', order=3, add_constant=True, add_trend=True)
X_train_fourier, X_test_fourier = fourier_gen.generate(train.index, len(test))

# 2️⃣ Add exogenous variables
X_train_all = X_train_fourier.copy()
X_train_all['mean_temp'] = train['mean_temperature']
X_train_all['mean_precip'] = train['mean_precip']

X_test_all = X_test_fourier.copy()
X_test_all['mean_temp'] = test['mean_temperature'].values
X_test_all['mean_precip'] = test['mean_precip'].values

# 3️⃣ Initialize and fit forecaster
forecaster_uc = UnobservedComponentsFourierForecaster(level='local linear trend')
forecaster_uc.fit(y_train=train['WSE'], X_train_all=X_train_all)

# 4️⃣ Forecast
forecast_df_uc = forecaster_uc.predict(steps=len(test), X_future=X_test_all)


from your_fourier_module import FourierDeterministicGenerator

# Setup Fourier Generator
fourier_gen = FourierDeterministicGenerator(index=train.index, order=3)

# Initialize & preprocess
lstm_model = LSTMForecaster(seq_len=30)
lstm_model.preprocess(train, test, fourier_gen=fourier_gen)

# Hyperparameter tuning
param_grid = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.2],
    "lr": [0.001],
    "batch_size": [16]
}
best_params = lstm_model.hyperparameter_tuning(param_grid)

# Train final model
lstm_model.train_final_model()

# Forecast
preds, metrics, best_config = lstm_model.forecast()


from models.lstm_forecaster import LSTMForecaster
from utils.fourier_generator import FourierDeterministicGenerator

# Example usage
forecaster = LSTMForecaster(order=3, seq_len=30)

# Step 1: Preprocess
forecaster.preprocess(train_df, test_df)

# Step 2: Tune hyperparameters
param_grid = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'dropout': [0.1, 0.3],
    'lr': [0.001],
    'batch_size': [32]
}
best_params = forecaster.hyperparameter_tuning(param_grid)

# Step 3: Train with best config
forecaster.train_final_model()

# Step 4: Forecast
forecast, metrics, _ = forecaster.forecast()



    if __name__ == "__main__":
    # Example usage:
    csv_path = "/content/data_RIHAND_combined.csv"
    output_dir = "/content/EDA_plots"

    df = pd.read_csv(csv_path)

    eda = TimeSeriesEDA(df, output_dir, window=30, lags=100)
    eda.run()

    dam_name = "Kalagarh_Dam_"

    merger = WeatherWSEMerger()
    merged_df = merger.process(wse_csv_path, weather_csv_path, output_dir, dam_name)
