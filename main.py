import pandas as pd
from FourierDeterministicGenerator import FourierDeterministicGenerator  # your class as given
from LSTM_hyperparameter import LSTMForecaster  # your LSTMForecaster class as discussed

def main():
    # === Load Data ===
    csv_path = r"D:\new_project\rihand_dam\Rihand_weather.csv"
    df = pd.read_csv(csv_path)

    # === Convert Date column to datetime and set as index ===
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # === Split Train/Test ===
    split_ratio = 0.7
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # === Create Fourier Features Using Your Provided Generator ===
    fourier_gen = FourierDeterministicGenerator(freq='A', order=3, add_constant=True, add_trend=True)

    # Generate Fourier features for train and test
    X_train_fourier, X_test_fourier = fourier_gen.generate(train.index, test_steps=len(test))

    # The returned X_train_fourier and X_test_fourier are DataFrames indexed by dates

    # === Initialize LSTM Forecaster ===
    lstm_model = LSTMForecaster(seq_len=30)

    # === Preprocess with your Fourier features ===
    # Pass train, test, train_fourier, test_fourier, and Fourier column names
    lstm_model.preprocess(
        train=train,
        test=test,
        train_fourier=X_train_fourier,
        test_fourier=X_test_fourier,
        fourier_cols=X_train_fourier.columns
    )

    # === Hyperparameter tuning ===
    param_grid = {
        "hidden_size": [32],
        "num_layers": [1],
        "dropout": [0.2],
        "lr": [0.001],
        "batch_size": [16],
    }
    model, best_params, results = lstm_model.hyperparameter_tuning_and_train(param_grid)

    # === Forecast ===
    preds, best_params = lstm_model.forecast()

    # === Output forecast ===
    forecast_index = test.index[lstm_model.seq_len:]
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Forecasted_WSE": preds
    }).set_index("Date")

    print("\nSample Forecast:")
    print(forecast_df.head())

    # === Optionally Save ===
    forecast_df.to_csv("forecast_output.csv")


if __name__ == "__main__":
    main()
