# Archivo: trading_bot/src/utils/decision_logger.py

import yaml
from pathlib import Path

class DecisionLogger:
    def __init__(self, log_path="logs/decisions.yaml"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, symbol, date, pred_price, pred_news, final_decision, reasoning, acc_price, acc_news):
        entry = {
            "symbol": symbol,
            "date": str(date),
            "predictions": {
                "price_model": float(pred_price),
                "news_model": float(pred_news),
                "final_decision": int(final_decision),
                "accuracy_price": float(acc_price),
                "accuracy_news": float(acc_news),
                "selected_reasoning": reasoning
            }
        }

        if not self.path.exists():
            with open(self.path, 'w') as f:
                yaml.dump([entry], f)
        else:
            with open(self.path, 'r') as f:
                data = yaml.safe_load(f) or []
            data.append(entry)
            with open(self.path, 'w') as f:
                yaml.dump(data, f)

    def should_use_decision(self, acc_price, acc_news, threshold=0.6):
        return acc_price >= threshold and acc_news >= threshold
