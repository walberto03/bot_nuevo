# Archivo: trading_bot/src/feature_builder.py

import pandas as pd
import numpy as np
from trading_bot.config import TradingConfig
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.data.db_manager import load_all_data
from sklearn.preprocessing import MinMaxScaler
import joblib

class FeatureBuilder:
    def __init__(self):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()
        self.sequence_length = 24
        self.input_size = 8
        self.scaler_path = self.cfg.models_dir / "multi_scaler.pkl"

    def build_sequences(self):
        df = load_all_data()
        df = df[df['Symbol'].isin(self.cfg.symbols)]
        df = df.sort_values(by=['Symbol', 'Date'])

        all_sequences, all_daily, labels = [], [], []
        daily_targets = self._build_daily_labels(df)

        for symbol in df['Symbol'].unique():
            df_sym = df[df['Symbol'] == symbol].copy()
            df_sym = self.pattern.calculate_technical_indicators(df_sym)
            df_sym.dropna(inplace=True)

            for i in range(len(df_sym) - self.sequence_length):
                window = df_sym.iloc[i:i+self.sequence_length]
                window_date = window['Date'].iloc[-1].date()

                if window_date not in daily_targets.get(symbol, {}):
                    continue

                hourly_feats = window[[
                    "RSI", "MACD", "Signal", "BB_up", "BB_dn",
                    f"SMA_{self.cfg.sma_short}", f"SMA_{self.cfg.sma_long}", "ATR"
                ]].values
                
                daily_feat = window[["Open", "High", "Low", "Close"]].agg(["first", "max", "min", "last"]).iloc[:, -1].values

                all_sequences.append(hourly_feats)
                all_daily.append(daily_feat)
                labels.append(daily_targets[symbol][window_date])

        scaler = MinMaxScaler()
        all_flattened = np.concatenate(all_sequences, axis=0)
        scaler.fit(all_flattened)
        joblib.dump(scaler, self.scaler_path)

        all_sequences = [scaler.transform(seq) for seq in all_sequences]
        X_hourly = np.array(all_sequences)
        X_daily = np.array(all_daily)
        y = np.array(labels)

        return X_hourly, X_daily, y

    def _build_daily_labels(self, df):
        targets = {}
        df_daily = df.groupby(['Symbol', df['Date'].dt.date]).agg({"Close": "last"})
        df_shift = df_daily.groupby(level=0).shift(-1).rename(columns={"Close": "Next_Close"})
        df_merged = df_daily.join(df_shift)
        df_merged['Target'] = (df_merged['Next_Close'] > df_merged['Close']).astype(int)

        for (symbol, date), row in df_merged.iterrows():
            if symbol not in targets:
                targets[symbol] = {}
            targets[symbol][date] = row['Target']

        return targets
