import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from IPython.display import Image, display
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
# from graphviz import Digraph

warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, df, target_col, exogenous_cols, output_dir,
                 window=30, lags=100, maxlag_granger=30):
        self.df = df
        self.target_col = target_col
        self.exogenous_cols = exogenous_cols
        self.output_dir = output_dir
        self.window = window
        self.lags = lags
        self.maxlag_granger = maxlag_granger
        os.makedirs(self.output_dir, exist_ok=True)

    def adf_test(self, series):
        result = adfuller(series.dropna())
        return result[0], result[1]

    def plot_series(self, series, col_name, suffix):
        plt.figure(figsize=(8, 4))
        plt.plot(series, label=col_name, color='blue')
        if suffix == 'timeseries':
            plt.plot(series.rolling(self.window).mean(), label=f'{self.window}-day Rolling Mean', color='orange')
        plt.grid(True)
        plt.title(f'{col_name} {"with Rolling Mean" if suffix=="timeseries" else "Differenced"}')
        plt.xlabel('Date')
        plt.ylabel(col_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{col_name}_{suffix}.png")
        plt.show()

    def plot_acf_pacf(self, series, col_name, suffix):
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(series.dropna(), lags=self.lags, ax=ax[0])
        ax[0].set_title(f'ACF of {col_name} {"(Differenced)" if "diff" in suffix else ""}')
        plot_pacf(series.dropna(), lags=self.lags, ax=ax[1])
        ax[1].set_title(f'PACF of {col_name} {"(Differenced)" if "diff" in suffix else ""}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{col_name}_{suffix}.png")
        plt.show()

    def analyze_column(self, col):
        print(f"\n{'='*60}\nProcessing: {col}\n{'='*60}")
        series = self.df[col].dropna()

        self.plot_series(series, col, 'timeseries')
        self.plot_acf_pacf(series, col, 'ACF_PACF')

        adf_stat, p_value = self.adf_test(series)
        print(f"ADF Statistic for {col}: {adf_stat:.4f}, p-value: {p_value:.4f}")

        if p_value > 0.05:
            print("--> Non-stationary, differencing applied.\n")
            diff_series = series.diff().dropna()
            self.plot_series(diff_series, col, 'first_difference')
            self.plot_acf_pacf(diff_series, col, 'ACF_PACF_diff')
            adf_stat_diff, p_value_diff = self.adf_test(diff_series)
            print(f"ADF Statistic after differencing: {adf_stat_diff:.4f}, p-value: {p_value_diff:.4f}")
        else:
            print("--> Stationary, no differencing needed.\n")

    def plot_correlation_heatmap(self):
        corr = self.df[[self.target_col] + self.exogenous_cols].corr()
        plt.figure(figsize=(5, 3))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png")
        plt.show()

    def plot_cross_correlation(self, max_lag=365):
        lags = range(max_lag)
        plt.figure(figsize=(8, 5))
        for exog in self.exogenous_cols:
            cross_corr = [self.df[exog].shift(lag).corr(self.df[self.target_col]) for lag in lags]
            plt.plot(lags, cross_corr, label=f'{exog} vs {self.target_col}')
        plt.xlabel('Lag (days)')
        plt.ylabel('Correlation')
        plt.title('Cross-Correlation with Lags')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cross_correlation_lags.png")
        plt.show()

    '''def check_granger_causality(self, alpha=0.05):
        dot = Digraph(comment='Granger Causality DAG')
        dot.node(self.target_col[:3].upper(), self.target_col)
        results = {}

        for exog in self.exogenous_cols:
            dot.node(exog[0].upper(), exog)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_result = grangercausalitytests(
                    self.df[[self.target_col, exog]].dropna(),
                    maxlag=self.maxlag_granger,
                    verbose=False
                )

            causality_found = False
            for lag in range(1, self.maxlag_granger + 1):
                p_val = test_result[lag][0]['ssr_ftest'][1]
                if p_val < alpha:
                    print(f"{exog} Granger-causes {self.target_col} at lag {lag} (p={p_val:.4f})")
                    dot.edge(exog[0].upper(), self.target_col[:3].upper(),
                             label=f'p={p_val:.4f}, lag={lag}')
                    results[exog] = (lag, p_val)
                    causality_found = True
                    break
            if not causality_found:
                print(f"{exog} does NOT Granger-cause {self.target_col} (all p >= {alpha})")

        dot.render(f"{self.output_dir}/granger_causality_dag", format='png', cleanup=True)
        display(Image(filename=f"{self.output_dir}/granger_causality_dag.png"))
        return results'''

    def feature_importance_rf(self):
        df_ml = self.df[[self.target_col] + self.exogenous_cols].dropna()
        X = df_ml[self.exogenous_cols]
        y = df_ml[self.target_col]
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        for feature, importance in zip(X.columns, importances):
            print(f"{feature} feature importance: {importance:.4f}")

    def full_eda_pipeline(self):
        self.plot_correlation_heatmap()
        self.plot_cross_correlation()
        for col in [self.target_col] + self.exogenous_cols:
            self.analyze_column(col)
        #self.check_granger_causality()
        self.feature_importance_rf()
