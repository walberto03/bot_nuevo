# Archivo: trading_bot/src/walkforward_tester.py

import pandas as pd
import numpy as np
from trading_bot.src.backtester import Backtester
from datetime import timedelta

class WalkForwardTester:
    def __init__(self, model, strategy_logic, data, window_size=60, step_size=20):
        self.model = model
        self.strategy_logic = strategy_logic  # Función que convierte predicciones en señales
        self.data = data.sort_values("Date")
        self.window_size = window_size
        self.step_size = step_size
        self.backtester = Backtester()

    def run(self):
        start_idx = 0
        results = []

        while start_idx + self.window_size <= len(self.data):
            train = self.data.iloc[start_idx:start_idx+self.window_size]
            test = self.data.iloc[start_idx+self.window_size:start_idx+self.window_size+self.step_size]

            if test.empty:
                break

            X_train = train.drop(columns=["Target", "Date"])
            y_train = train["Target"]

            self.model.fit(X_train, y_train)

            X_test = test.drop(columns=["Target", "Date"])
            test_preds = self.model.predict_proba(X_test)[:, 1]

            signals = self.strategy_logic(test["Date"].values, test["Symbol"].values, test_preds)
            stats = self.backtester.run_backtest(signals)
            results.append({
                "start": train["Date"].min(),
                "end": test["Date"].max(),
                "stats": stats
            })

            start_idx += self.step_size

        return results
