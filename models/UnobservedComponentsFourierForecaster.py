import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

class UnobservedComponentsFourierForecaster:
    """
    Unobserved Components forecaster with Fourier + trend features.

    Example:
        forecaster = UnobservedComponentsFourierForecaster(level='local linear trend')
        forecaster.fit(y_train, X_train_all)
        forecast_df = forecaster.predict(steps=len(y_test), X_future=X_test_all)
    """

    def __init__(self, level='local linear trend', seasonal=None, freq_seasonal=None):
        """
        Parameters:
            level: str, e.g., 'local linear trend'
            seasonal: int or None, optional
            freq_seasonal: list of dicts, optional, for periodic seasonalities
        """
        self.level = level
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal

        self.model = None
        self.result = None

    def fit(self, y_train, X_train_all=None):
        """
        Fit the UnobservedComponents model.

        Parameters:
            y_train: pd.Series with DateTimeIndex
            X_train_all: pd.DataFrame with pre-generated Fourier + exogenous features
        """
        self.model = UnobservedComponents(
            endog=y_train,
            level=self.level,
            seasonal=self.seasonal,
            freq_seasonal=self.freq_seasonal,
            exog=X_train_all
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
            forecast_df: pd.DataFrame with forecast mean and confidence intervals
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
