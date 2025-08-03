# trading_bot/src/model_trainer_lstm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from trading_bot.src.data.db_manager import load_grouped_data
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.config import TradingConfig
from trading_bot.src.models.lstm_model import LSTMModel

class LSTMTrainer:
    def __init__(self):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()
        self.model_path = self.cfg.models_dir / "lstm_model.pt"
        self.input_size = 8  # número de features esperados por el modelo

    def prepare_data(self, sequence_length=10):
        grouped = load_grouped_data()
        X_seq, y_seq = [], []

        for df in grouped.values():
            df = df.copy()
            df = self.pattern.calculate_technical_indicators(df)
            df.dropna(inplace=True)

            if "Close" not in df.columns:
                continue

            df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

            features = [
                "RSI", "MACD", "Signal", "BB_up", "BB_dn", "ATR",
                f"SMA_{self.cfg.sma_short}", f"SMA_{self.cfg.sma_long}"
            ]
            if not all(col in df.columns for col in features):
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])

            for i in range(sequence_length, len(df) - 1):
                X_seq.append(scaled[i-sequence_length:i])
                y_seq.append(df["target"].iloc[i])

        if not X_seq:
            return None

        X = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    def train(self, epochs=15):
        print("[LSTMTrainer] Entrenando modelo LSTM...")
        dataloader = self.prepare_data()
        if dataloader is None or len(dataloader.dataset) == 0:
            print("[LSTMTrainer] No hay suficientes datos para entrenar.")
            return

        model = LSTMModel(input_size=self.input_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[LSTMTrainer] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Evaluación rápida
        model.eval()
        with torch.no_grad():
            X_eval, y_eval = next(iter(dataloader))
            preds = model(X_eval).numpy()
            preds_bin = (preds > 0.5).astype(int)
            print("[LSTMTrainer] Evaluación:")
            print(classification_report(y_eval.numpy(), preds_bin))

        torch.save(model.state_dict(), self.model_path)
        print(f"[LSTMTrainer] Modelo guardado en {self.model_path}")
