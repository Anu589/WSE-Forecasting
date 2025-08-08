import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

class FourierDeterministicGenerator:
    def __init__(self, freq='A', order=3, add_constant=True, add_trend=True):
        """
        Initialize the Fourier deterministic generator.

        Args:
            freq: Frequency for CalendarFourier ('A' for annual, 'M' for monthly).
            order: Number of Fourier harmonics.




            add_constant: Whether to add intercept.
            add_trend: Whether to add linear trend.
        """
        self.freq = freq
        self.order = order
        self.add_constant = add_constant
        self.add_trend = add_trend
        self.dp = None  # Will hold the fitted DeterministicProcess

    def generate(self, train_index, test_steps): ## This is the generate funtion
        fourier = CalendarFourier(freq=self.freq, order=self.order)
        self.dp = DeterministicProcess(
            index=train_index,
            constant=self.add_constant,
            order=1 if self.add_trend else 0,
            seasonal=False,
            additional_terms=[fourier]
        )

        X_train = self.dp.in_sample()

        # 🔧 Fix: Generate proper future index
        freq = pd.infer_freq(train_index)
        if freq is None:
            freq = 'D'  # default to daily if not inferable

        future_index = pd.date_range(start=train_index[-1] + pd.Timedelta(1, unit=freq[0]), periods=test_steps, freq=freq)
        X_test = self.dp.out_of_sample(steps=test_steps, forecast_index=future_index)

        return X_train, X_test

