import numpy as np
import joblib
import torch
from trading_bot.config import TradingConfig
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.risk_manager import RiskManager
from trading_bot.src.models.lstm_model import LSTMModel

class DecisionMaker:
    def __init__(self):
        self.cfg              = TradingConfig()
        self.weights          = getattr(self.cfg,"decision_weights", { ... })
        self.buy_threshold    = getattr(self.cfg,"buy_threshold",0.6)
        self.sell_threshold   = getattr(self.cfg,"sell_threshold",0.4)
        self.rr               = self.cfg.risk_reward
        self.pattern_analyzer = PatternAnalyzer()
        self.risk_manager     = RiskManager(self.cfg)
        # carga MLP, RF, LSTM...

    def make_decision(self, patterns, sentiment, history, df, entry_price, account_balance):
        # calcula tech_score, news_score, hist_score, mlp_score, rf_score, lstm_score...
        total_score = ...
        return self._build_action(total_score, entry_price, account_balance)

    def _build_action(self, score, entry_price, account_balance):
        # normaliza score, decide BUY/SELL/HOLD...
        if action == "HOLD":
            return {"action":action,"score":round(score,3)}

        # cálculo de riesgos
        stop_pct    = ...
        position_size = self.risk_manager.adjust_size(account_balance,stop_pct)
        stop_price    = self.risk_manager.compute_stop(entry_price,stop_pct)
        take_profit   = entry_price + (entry_price-stop_price)*self.rr

        # **Bloque de impresión detallada**
        p_succ    = score
        p_fail    = 1 - p_succ
        riesgo_pct     = abs((stop_price-entry_price)/entry_price)*100
        recompensa_pct = abs((take_profit-entry_price)/entry_price)*100
        ev             = (p_succ*recompensa_pct) - (p_fail*riesgo_pct)
        ev_res         = "POSITIVO" if ev>0 else "NEGATIVO"

        print("\n=== DECISIÓN DE TRADE ===")
        print(f"Probabilidad de éxito: {p_succ*100:.2f}%")
        print(f"Precio entrada: {entry_price:.5f}")
        print(f"Stop Loss: {stop_price:.5f} (-{riesgo_pct:.2f}%)")
        print(f"Take Profit: {take_profit:.5f} (+{recompensa_pct:.2f}%)")
        print(f"Valor Esperado (EV): {ev:.2f}% — {ev_res}")
        print("=========================\n")

        return {
            "action":action,
            "score":round(score,3),
            "position_size":round(position_size,4),
            "stop_loss_price":round(stop_price,6),
            "take_profit_price":round(take_profit,6)
        }
