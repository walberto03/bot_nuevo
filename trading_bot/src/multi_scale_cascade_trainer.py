# Archivo: trading_bot/src/multi_scale_cascade_trainer.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from trading_bot.src.models.lstm_model import LSTMModel
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.config import TradingConfig
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.utils.early_stopping import EarlyStopping
import joblib

class MultiScaleCascadeTrainer:
    def __init__(self):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()

    def _prepare_sequences(self, df_input, df_target, input_seq_len, timeframe_name):
        df_input['date'] = pd.to_datetime(df_input['date'])
        df_target['date'] = pd.to_datetime(df_target['date'])
        df_input = df_input.sort_values(by=['symbol', 'date'])
        df_target = df_target.sort_values(by=['symbol', 'date'])

        X_list, y_list = [], []
        scaler = MinMaxScaler()

        for symbol in df_input['symbol'].unique():
            input_data = df_input[df_input['symbol'] == symbol].copy()
            target_data = df_target[df_target['symbol'] == symbol].copy()
            input_data = self.pattern.calculate_technical_indicators(input_data)
            input_data.dropna(inplace=True)

            target_map = target_data.set_index(target_data['date'].dt.floor(timeframe_name))['close'].shift(-1) > \
                         target_data.set_index(target_data['date'].dt.floor(timeframe_name))['close']

            for i in range(len(input_data) - input_seq_len):
                window = input_data.iloc[i:i+input_seq_len]
                ref_date = window['date'].iloc[-1].floor(timeframe_name)
                if ref_date in target_map:
                    features = window[[
                        "rsi", "macd", "signal", "bb_up", "bb_dn",
                        f"sma_{self.cfg.sma_short}", f"sma_{self.cfg.sma_long}", "atr"
                    ]].values
                    features = scaler.fit_transform(features)
                    label = int(target_map[ref_date])
                    X_list.append(features)
                    y_list.append(label)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(X, y), batch_size=self.cfg.batch_size, shuffle=True)

    def _train_model(self, dataloader, model_path, epochs):
        input_size = 8
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            bidirectional=self.cfg.bidirectional
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)
        early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            verbose=True,
            path=str(model_path)
        )

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
            print(f"[CascadeTrainer] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            early_stopping(total_loss, model)
            if early_stopping.early_stop:
                print("Entrenamiento detenido temprano.")
                break

        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Evaluar modelo final
        X_eval, y_eval = next(iter(dataloader))
        with torch.no_grad():
            preds = torch.sigmoid(model(X_eval)).numpy()
            pred_bin = (preds > 0.5).astype(int)
            print(classification_report(y_eval.numpy(), pred_bin))

        torch.save(model.state_dict(), model_path)
        print(f"[CascadeTrainer] Modelo guardado en {model_path}")

    def train_predict_1h_from_15m(self):
        print("[CascadeTrainer] Entrenando 1h desde 15m...")
        df_15m = load_all_data(timeframe="15min")
        df_1h = load_all_data(timeframe="1h")
        loader = self._prepare_sequences(df_15m, df_1h, input_seq_len=24, timeframe_name='H')
        model_path = self.cfg.models_dir / "lstm_15m_to_1h.pt"
        self._train_model(loader, model_path, epochs=self.cfg.num_epochs)

    def train_predict_4h_from_1h(self):
        print("[CascadeTrainer] Entrenando 4h desde 1h...")
        df_1h = load_all_data(timeframe="1h")
        df_4h = load_all_data(timeframe="4h")
        loader = self._prepare_sequences(df_1h, df_4h, input_seq_len=24, timeframe_name='4H')
        model_path = self.cfg.models_dir / "lstm_1h_to_4h.pt"
        self._train_model(loader, model_path, epochs=self.cfg.num_epochs)

    def train_predict_1d_from_1h_4h(self):
        print("[CascadeTrainer] Entrenando 1d desde 1h y 4h...")
        df_1h = load_all_data(timeframe="1h")
        df_4h = load_all_data(timeframe="4h")
        df_daily = load_all_data(timeframe="1d")
        # Aquí podrías hacer una fusión personalizada 1h+4h como features antes de entrenar
        df_combined = pd.concat([df_1h, df_4h]).sort_values(by=["symbol", "date"])
        loader = self._prepare_sequences(df_combined, df_daily, input_seq_len=24, timeframe_name='D')
        model_path = self.cfg.models_dir / "lstm_1h4h_to_1d.pt"
        self._train_model(loader, model_path, epochs=self.cfg.num_epochs)
