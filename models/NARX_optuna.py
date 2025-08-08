import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Dataset ----------
class NARXDataset(Dataset):
    def __init__(self, df, input_lags=5):
        self.X = []
        self.y = []
        for i in range(input_lags, len(df)):
            wse_lags = df['WSE'].values[i - input_lags:i]
            temp_lags = df['mean_temperature'].values[i - input_lags:i]
            precip_lags = df['mean_precip'].values[i - input_lags:i]
            features = np.concatenate([wse_lags, temp_lags, precip_lags])
            self.X.append(features)
            self.y.append(df['WSE'].values[i])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- Model ----------
class NARXModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):
        super(NARXModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class NARXPipeline:
    def __init__(self):
        self.model = None

    def _train_model(self, train_df, input_lags, hidden_dim, dropout_rate, learning_rate, batch_size, epochs):
        dataset = NARXDataset(train_df, input_lags)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = 3 * input_lags
        model = NARXModel(input_dim, hidden_dim, dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
        return model

    def _evaluate_model(self, model, train_df, val_df, input_lags):
        model.eval()
        true_vals = []
        pred_vals = []

        with torch.no_grad():
            combined = pd.concat([train_df, val_df]).reset_index(drop=True)
            for i in range(len(train_df), len(combined)):
                if i - input_lags < 0:
                    continue
                wse_lags = combined['WSE'].values[i - input_lags:i]
                temp_lags = combined['mean_temperature'].values[i - input_lags:i]
                precip_lags = combined['mean_precip'].values[i - input_lags:i]
                features = np.concatenate([wse_lags, temp_lags, precip_lags])
                X_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                pred = model(X_input)
                pred_vals.append(pred.item())
                true_vals.append(combined['WSE'].values[i])

        mse = np.mean((np.array(pred_vals) - np.array(true_vals)) ** 2)
        return mse

    def tune(self, train_df, val_df, n_trials=30, max_epochs=50):
        def objective(trial):
            input_lags = trial.suggest_int("input_lags", 5, 20)
            hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

            model = self._train_model(
                train_df, input_lags, hidden_dim, dropout_rate,
                learning_rate, batch_size, epochs=max_epochs
            )
            mse = self._evaluate_model(model, train_df, val_df, input_lags)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        for k, v in study.best_params.items():
            print(f"{k}: {v}")

        # Store best model for forecasting later
        best = study.best_params
        self.model = self._train_model(
            train_df, best['input_lags'], best['hidden_dim'], best['dropout_rate'],
            best['learning_rate'], best['batch_size'], epochs=max_epochs
        )
        self.best_params = best

    def forecast(self, train_df, test_df):
        if self.model is None:
            raise RuntimeError("Model is not trained. Run train or tune first.")

        input_lags = self.best_params['input_lags']
        return self._forecast_with_model(self.model, train_df, test_df, input_lags)

    def _forecast_with_model(self, model, train_df, test_df, input_lags):
        model.eval()
        y_pred = []
        with torch.no_grad():
            combined = pd.concat([train_df, test_df]).reset_index(drop=True)
            for i in range(len(train_df), len(combined)):
                if i - input_lags < 0:
                    continue
                wse_lags = combined['WSE'].values[i - input_lags:i]
                temp_lags = combined['mean_temperature'].values[i - input_lags:i]
                precip_lags = combined['mean_precip'].values[i - input_lags:i]
                features = np.concatenate([wse_lags, temp_lags, precip_lags])
                X_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                pred = model(X_input)
                y_pred.append(pred.item())
        return y_pred
