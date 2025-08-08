import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import itertools
import os

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))  # input gate
        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))  # forget gate
        self.W_ig = nn.Parameter(torch.randn(input_size, hidden_size))  # candidate
        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))  # output gate

        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))  # input gate
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))  # forget gate
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))  # candidate
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))  # output gate

        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.ones(hidden_size))  # forget bias init to 1
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.init_weights()

    def init_weights(self):
        for weight in [self.W_ii, self.W_if, self.W_ig, self.W_io,
                       self.W_hi, self.W_hf, self.W_hg, self.W_ho]:
            nn.init.xavier_uniform_(weight)

    def forward(self, x, state):
        h_prev, c_prev = state

        i = torch.sigmoid(torch.matmul(x, self.W_ii) + torch.matmul(h_prev, self.W_hi) + self.b_i)
        f = torch.sigmoid(torch.matmul(x, self.W_if) + torch.matmul(h_prev, self.W_hf) + self.b_f)
        g = torch.tanh(torch.matmul(x, self.W_ig) + torch.matmul(h_prev, self.W_hg) + self.b_g)
        o = torch.sigmoid(torch.matmul(x, self.W_io) + torch.matmul(h_prev, self.W_ho) + self.b_o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        h = self.layer_norm(h)

        return h, (h, c)

class ManualDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * mask / (1 - self.p)
        return x

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.attn_query_vector = nn.Parameter(torch.randn(hidden_size, 1))
        nn.init.xavier_uniform_(self.attn_query_vector)

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lstm_layers.append(LSTMCell(input_size, hidden_size))
            else:
                self.lstm_layers.append(LSTMCell(hidden_size, hidden_size))

        self.dropout_layers = nn.ModuleList([ManualDropout(dropout) for _ in range(num_layers - 1)])

        self.final_dropout = ManualDropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        device = x.device

        h_states = [torch.zeros(batch, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch, self.hidden_size, device=device) for _ in range(self.num_layers)]

        all_hidden_states = []

        for t in range(seq_len):
            layer_input = x[:, t, :]
            for idx in range(self.num_layers):
                h_prev = h_states[idx]
                c_prev = c_states[idx]

                h_new, (h_new, c_new) = self.lstm_layers[idx](layer_input, (h_prev, c_prev))

                # Residual connection if shapes match
                if layer_input.shape == h_new.shape:
                    h_new = h_new + layer_input

                h_states[idx] = h_new
                c_states[idx] = c_new

                if idx < self.num_layers - 1:
                    layer_input = self.dropout_layers[idx](h_new)
                else:
                    layer_input = h_new

            all_hidden_states.append(h_states[-1].unsqueeze(1)) # [batch, 1, hidden]

        all_hidden_states = torch.cat(all_hidden_states, dim=1) # [batch, seq_len, hidden]

        attn_scores = torch.softmax(all_hidden_states @ self.attn_query_vector, dim=1)
        context = torch.sum(attn_scores * all_hidden_states, dim=1)
        context = self.final_dropout(context)
        out = self.fc(context)
        return out.squeeze(-1)  # squeeze last dim for output shape `[batch]`

class LSTMForecaster:
    def __init__(self, seq_len=30):
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.fourier_model = None
        self.best_params = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optional for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def preprocess(self, train, test, train_fourier, test_fourier, fourier_cols):
        # Concatenate exogenous vars and Fourier terms
        train_exog = pd.concat([
            train[['P', 'Pres', 'SpecHum', 'Temp', 'Wind']].reset_index(drop=True),
            train_fourier.reset_index(drop=True)
        ], axis=1)
        test_exog = pd.concat([
            test[['P', 'Pres', 'SpecHum', 'Temp', 'Wind']].reset_index(drop=True),
            test_fourier.reset_index(drop=True)
        ], axis=1)

        self.fourier_cols = list(fourier_cols)

        # Fit linear regression to Fourier features for trend removal
        X_fourier_train = train_fourier[self.fourier_cols].values
        y_train = train['WSE'].values
        self.fourier_model = LinearRegression().fit(X_fourier_train, y_train)
        train_trend = self.fourier_model.predict(X_fourier_train)
        self.residual_train = y_train - train_trend

        # Check residual for NaNs or constant values
        if np.any(np.isnan(self.residual_train)):
            raise ValueError("Residual train target contains NaNs")
        if np.std(self.residual_train) < 1e-6:
            print("Warning: Residual train target is near constant, model may not learn well.")

        # Prepare combined features for scaling (target + exogenous + Fourier)
        full_train_features = pd.concat([train[['WSE']], train_exog], axis=1)
        full_test_features = pd.concat([test[['WSE']], test_exog], axis=1)

        # Fit scaler on train and transform both train and test
        self.scaled_train = self.scaler.fit_transform(full_train_features)
        self.scaled_test = self.scaler.transform(full_test_features)

        self.train = train
        self.test = test
        self.train_exog = train_exog
        self.test_exog = test_exog

    def _create_sequences(self, data, y_data):
        X_seq, y_seq = [], []
        max_idx = min(len(data), len(y_data)) - self.seq_len
        for i in range(max_idx):
            X_seq.append(data[i:i + self.seq_len])
            y_seq.append(y_data[i + self.seq_len])
        return np.array(X_seq), np.array(y_seq)

    def _train_single_model(self, config, X_train, y_train, X_val, y_val):
        model = MultiLayerLSTM(
            input_size=X_train.shape[2],
            hidden_size=config['hidden_size'],
            output_size=1,
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)
        loss_fn = nn.SmoothL1Loss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True)

        n_epochs = 50
        patience = 10
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            epoch_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)

                # Check inputs and targets for NaNs/Infs
                if torch.isnan(xb).any() or torch.isinf(xb).any():
                    raise RuntimeError("Found NaN or Inf in input batch.")
                if torch.isnan(yb).any() or torch.isinf(yb).any():
                    raise RuntimeError("Found NaN or Inf in target batch.")

                optimizer.zero_grad()
                out = model(xb)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError("Model output contains NaN or Inf.")

                loss = loss_fn(out, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("Loss is NaN or Inf during training.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_epoch_loss = np.mean(epoch_losses)

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val.to(self.device, dtype=torch.float32)).cpu().numpy()
                val_true = y_val.cpu().numpy()
                val_rmse = np.sqrt(np.mean((val_preds - val_true) ** 2))
                val_mae = np.mean(np.abs(val_preds - val_true))
                mase_denom = np.mean(np.abs(np.diff(val_true)))
                if mase_denom < 1e-6:  # Safety check to avoid division by zero
                    mase_denom = 1.0
                val_mase = val_mae / mase_denom

            scheduler.step(val_rmse)

            print(f"Epoch {epoch+1}: train loss={avg_epoch_loss:.4f}, val RMSE={val_rmse:.4f}, val MASE={val_mase:.4f}")

            if val_mase < best_val_loss:
                best_val_loss = val_mase
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_lstm_model_temp.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(torch.load("best_lstm_model_temp.pt"))
        if os.path.exists("best_lstm_model_temp.pt"):
            os.remove("best_lstm_model_temp.pt")

        # Recalculate validation metrics with best model
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(self.device, dtype=torch.float32)).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_rmse = np.sqrt(np.mean((val_preds - val_true) ** 2))
            val_mae = np.mean(np.abs(val_preds - val_true))
            mase_denom = np.mean(np.abs(np.diff(val_true)))
            if mase_denom < 1e-6:
                mase_denom = 1.0
            val_mase = val_mae / mase_denom

        return val_mase, val_rmse, val_mae, model

    def hyperparameter_tuning_and_train(self, param_grid):
        X_seq, y_seq = self._create_sequences(self.scaled_train, self.residual_train)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        split_idx = int(0.8 * len(X_tensor))
        X_train, y_train = X_tensor[:split_idx], y_tensor[:split_idx]
        X_val, y_val = X_tensor[split_idx:], y_tensor[split_idx:]

        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        results = []

        print(f"Starting hyperparameter tuning over {len(param_combinations)} configurations...")

        best_overall_loss = np.inf
        best_overall_model = None
        best_overall_params = None

        for idx, param_set in enumerate(param_combinations):
            config = dict(zip(param_names, param_set))
            print(f"\nTraining config {idx+1}/{len(param_combinations)}: {config}")

            val_mase, val_rmse, val_mae, model = self._train_single_model(config, X_train, y_train, X_val, y_val)

            results.append({
                "config": config,
                "val_mase": val_mase,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            })

            if val_mase < best_overall_loss:
                best_overall_loss = val_mase
                best_overall_model = model
                best_overall_params = config

            print(f"Validation MASE for this config: {val_mase:.4f}")

        if best_overall_model is None:
            raise RuntimeError("No valid model was found during hyperparameter tuning.")

        self.best_params = best_overall_params
        self.model = best_overall_model

        # Save best model overall
        torch.save(self.model.state_dict(), "best_lstm_model.pt")

        sorted_results = sorted(results, key=lambda x: x['val_mase'])
        print("\n=== Hyperparameter Tuning Complete ===")
        print(f"Best hyperparameters: {self.best_params}")
        print(f"Best validation MASE: {best_overall_loss:.4f}")
        print("\nTop 3 configs:")
        for i, res in enumerate(sorted_results[:3]):
            print(f"{i+1}. MASE: {res['val_mase']:.4f}, Config: {res['config']}")

        return self.model, self.best_params, sorted_results

    def forecast(self):
        if self.model is None:
            raise ValueError("Model has not been trained. Run hyperparameter_tuning_and_train first.")

        X_seq = []
        for i in range(len(self.scaled_test) - self.seq_len):
            X_seq.append(self.scaled_test[i:i + self.seq_len])

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds_residual = self.model(X_tensor).cpu().numpy()

        X_fourier_test = self.test_exog[self.fourier_cols].values
        fourier_trend = self.fourier_model.predict(X_fourier_test)
        final_preds = preds_residual + fourier_trend[-len(preds_residual):]

        return final_preds, self.best_params
