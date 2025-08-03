# Archivo: trading_bot/src/multi_timescale_supervisor.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from trading_bot.src.models.lstm_model import LSTMModel
from trading_bot.config import TradingConfig
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.pattern_analyzer import PatternAnalyzer

class MultiTimescaleSupervisor:
    def __init__(self):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()
        self.model_path = self.cfg.models_dir / "supervised_multi_lstm.pt"
        self.sequence_length = 24
        self.input_size = 8

    def prepare_combined_sequences(self):
        df_hourly = load_all_data(timeframe="1h")
        df_daily = load_all_data(timeframe="1d")
        df_hourly = df_hourly[df_hourly['Symbol'].isin(self.cfg.symbols)]
        df_daily = df_daily[df_daily['Symbol'].isin(self.cfg.symbols)]

        sequences, labels = [], []
        scaler = MinMaxScaler()

        for symbol in self.cfg.symbols:
            h_data = df_hourly[df_hourly['Symbol'] == symbol].copy()
            d_data = df_daily[df_daily['Symbol'] == symbol].copy()
            h_data = self.pattern.calculate_technical_indicators(h_data)
            h_data.dropna(inplace=True)

            daily_map = d_data.set_index(d_data['Date'].dt.date)['Close'].shift(-1) > d_data.set_index(d_data['Date'].dt.date)['Close']

            for i in range(0, len(h_data) - self.sequence_length):
                seq = h_data.iloc[i:i+self.sequence_length]
                ref_date = seq['Date'].iloc[-1].date()

                if ref_date in daily_map:
                    features = seq[[
                        "RSI", "MACD", "Signal", "BB_up", "BB_dn",
                        f"SMA_{self.cfg.sma_short}", f"SMA_{self.cfg.sma_long}", "ATR"
                    ]].values

                    features = scaler.fit_transform(features)
                    label = int(daily_map[ref_date])

                    sequences.append(features)
                    labels.append(label)

        X = torch.tensor(np.array(sequences), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    def train(self, epochs=20):
        dataloader = self.prepare_combined_sequences()
        if dataloader is None or len(dataloader.dataset) == 0:
            print("[Supervisor] No hay datos suficientes para entrenamiento.")
            return

        model = LSTMModel(input_size=self.input_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Supervisor] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            X_eval, y_eval = next(iter(dataloader))
            preds = model(X_eval).numpy()
            pred_bin = (preds > 0.5).astype(int)
            print("[Supervisor] Evaluaci\u00f3n:")
            print(classification_report(y_eval.numpy(), pred_bin))

        torch.save(model.state_dict(), self.model_path)
        print(f"[Supervisor] Modelo guardado en {self.model_path}")
