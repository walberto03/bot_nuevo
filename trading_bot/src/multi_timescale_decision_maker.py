import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from trading_bot.config import TradingConfig
from trading_bot.src.models.lstm_model import LSTMModel
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.utils.early_stopping import EarlyStopping
import numpy as np
import joblib

class MultiTimescaleTrainer:
    def __init__(self, learning_rate=None, batch_size=None, sequence_length=None,
                 hidden_size=None, num_layers=None, optuna_mode=False):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()

        # Hiperparámetros dinámicos
        self.learning_rate = learning_rate or self.cfg.learning_rate
        self.batch_size = batch_size or self.cfg.batch_size
        self.sequence_length = sequence_length or self.cfg.sequence_length
        self.hidden_size = hidden_size or self.cfg.hidden_size
        self.num_layers = num_layers or self.cfg.num_layers
        self.optuna_mode = optuna_mode

        self.input_size = len([
            "rsi", "macd", "signal", "bb_up", "bb_dn",
            f"sma_{self.cfg.sma_short}", f"sma_{self.cfg.sma_long}", "atr"
        ])
        self.model_path = self.cfg.models_dir / "multi_lstm_model.pt"
        self.scaler_path = self.cfg.models_dir / "multi_scaler.pkl"

    def prepare_data(self):
        df = load_all_data()
        df = df[df['symbol'].isin(self.cfg.symbols)].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date'])

        # Etiquetas diarias
        df_labels = df.copy()
        df_labels.columns = df_labels.columns.str.lower()
        daily_targets = self._build_daily_labels(df_labels)

        X_list, y_list = [], []
        for symbol in df['symbol'].unique():
            hourly = df[df['symbol'] == symbol].copy()
            hourly = self.pattern.calculate_technical_indicators(hourly)
            hourly.columns = hourly.columns.str.lower()
            hourly.dropna(inplace=True)

            for i in range(len(hourly) - self.sequence_length):
                window = hourly.iloc[i:i + self.sequence_length]
                date_key = window['date'].iloc[-1].date()
                if date_key in daily_targets.get(symbol, {}):
                    features = window[[
                        "rsi", "macd", "signal", "bb_up", "bb_dn",
                        f"sma_{self.cfg.sma_short}",
                        f"sma_{self.cfg.sma_long}", "atr"
                    ]].values
                    X_list.append(features)
                    y_list.append(daily_targets[symbol][date_key])

        X = np.array(X_list)
        y = np.array(y_list)
        if len(y) == 0:
            return None, None

        # Escalado
        scaler = MinMaxScaler()
        scaler.fit(X.reshape(-1, X.shape[-1]))
        joblib.dump(scaler, self.scaler_path)
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.cfg.validation_split,
            stratify=y, random_state=42
        )

        # Balanceo de clases en train
        if self.cfg.oversample or self.cfg.undersample:
            data_train = np.concatenate([
                X_train.reshape((X_train.shape[0], -1)),
                y_train.reshape(-1, 1)
            ], axis=1)
            df_train = pd.DataFrame(data_train)
            majority = df_train[df_train.iloc[:, -1] == 0]
            minority = df_train[df_train.iloc[:, -1] == 1]
            if self.cfg.oversample:
                minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
                df_train = pd.concat([majority, minority_up])
            if self.cfg.undersample:
                majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
                df_train = pd.concat([majority_down, minority])
            arr = df_train.values
            X_train = arr[:, :-1].reshape(-1, self.sequence_length, self.input_size)
            y_train = arr[:, -1]

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        return train_loader, val_loader

    def _build_daily_labels(self, df):
        targets = {}
        df_daily = df.groupby(['symbol', df['date'].dt.date])['close'].last()
        df_next = df_daily.groupby(level=0).shift(-1).rename('next_close')
        df_merged = df_daily.to_frame('close').join(df_next.to_frame())
        df_merged['target'] = (df_merged['next_close'] > df_merged['close']).astype(int)
        for (symbol, date), row in df_merged.iterrows():
            targets.setdefault(symbol, {})[date] = int(row['target'])
        return targets

    def train(self):
        print("[MultiTimescaleTrainer] Entrenando modelo multiescala...")
        train_loader, val_loader = self.prepare_data()
        if train_loader is None:
            print("[MultiTimescaleTrainer] No hay datos suficientes.")
            return

        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.cfg.bidirectional
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            verbose=True,
            path=str(self.model_path)
        )

        for epoch in range(self.cfg.num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{self.cfg.num_epochs} — Train loss: {train_loss:.4f} — Val loss: {val_loss:.4f}")
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Terminando entrenamiento temprano tras no mejorar.")
                break

        # Guardar y cargar modelo si no está en optuna
        if not self.optuna_mode:
            model.load_state_dict(torch.load(self.model_path))
            print(f"[MultiTimescaleTrainer] Mejor modelo cargado desde {self.model_path}")
            torch.save(model.state_dict(), self.model_path)
            print(f"[MultiTimescaleTrainer] Modelo guardado en {self.model_path}")

        # Evaluación final
        model.eval()
        X_eval, y_eval = next(iter(val_loader))
        with torch.no_grad():
            preds = torch.sigmoid(model(X_eval)).numpy()
            preds_bin = (preds > 0.5).astype(int)
            acc = accuracy_score(y_eval.numpy(), preds_bin)

        print("[MultiTimescaleTrainer] Evaluación final:")
        print(classification_report(y_eval.numpy(), preds_bin))
        
        return acc  # <-- Para Optuna también

    def train_and_evaluate(self):
        return self.train()
