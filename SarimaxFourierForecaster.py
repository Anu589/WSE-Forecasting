import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaxFourierForecaster:
    """
    SARIMAX forecaster that accepts pre-generated Fourier + trend features
    for consistent seasonal handling across the pipeline.

    Usage:
        # Initialize
        forecaster = SarimaxFourierForecaster(order=(2,0,1))

        # Fit
        forecaster.fit(y_train, X_train_all)

        # Predict
        forecast_df = forecaster.predict(steps=len(y_test), X_future=X_test_all)
    """
    def __init__(self,
                 order=(2, 0, 1),
                 seasonal_order=(0, 0, 0, 0),
                 enforce_stationarity=False,
                 enforce_invertibility=False):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.result = None

    def fit(self, y_train, X_train_all=None):
        """
        Fit the SARIMAX model.

        Parameters:
            y_train: pd.Series with DateTimeIndex
            X_train_all: pd.DataFrame with pre-generated Fourier + exogenous features
        """
        self.model = SARIMAX(
            endog=y_train,
            exog=X_train_all,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )

        self.result = self.model.fit(disp=False)
        return self.result

    def predict(self, steps, X_future=None):
        """
        Predict future values.

        Parameters:
            steps: int, forecast horizon
            X_future: pd.DataFrame with pre-generated Fourier + exogenous features

        Returns:
            forecast_df: pd.DataFrame with 'mean', 'mean_ci_lower', 'mean_ci_upper'
        """
        forecast_obj = self.result.get_forecast(steps=steps, exog=X_future)
        forecast_df = forecast_obj.summary_frame()
        return forecast_df

    def get_fitted_values(self):
        """Return in-sample fitted values."""
        if self.result is not None:
            return self.result.fittedvalues
        else:
            raise ValueError("Model has not been fitted yet.")
