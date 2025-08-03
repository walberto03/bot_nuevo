# Archivo: trading_bot/src/adaptive_retrainer.py

import os
import json
from datetime import datetime, timedelta
from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer
from trading_bot.config import TradingConfig

STATE_FILE = "retraining_state.json"
THRESHOLD = 0.55  # Umbral mínimo de precisión para activar reentrenamiento
DAYS_INTERVAL = 15

class AdaptiveRetrainer:
    def __init__(self):
        self.cfg = TradingConfig()
        self.state_path = self.cfg.models_dir / STATE_FILE

    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {"last_trained": "2020-01-01", "last_accuracy": 1.0}

    def _save_state(self, accuracy):
        state = {
            "last_trained": datetime.today().strftime("%Y-%m-%d"),
            "last_accuracy": accuracy
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f)

    def should_retrain(self):
        state = self._load_state()
        last_trained = datetime.strptime(state["last_trained"], "%Y-%m-%d")
        last_accuracy = state["last_accuracy"]

        time_elapsed = (datetime.today() - last_trained).days
        return time_elapsed >= DAYS_INTERVAL or last_accuracy < THRESHOLD

    def retrain_if_needed(self):
        if not self.should_retrain():
            print("[AdaptiveRetrainer] No es necesario reentrenar hoy.")
            return

        print("[AdaptiveRetrainer] Reentrenando modelo multiescala...")
        trainer = MultiTimescaleTrainer()
        accuracy = trainer.train()  # Debe devolver precisión si se quiere actualizar
        self._save_state(accuracy or 1.0)
        print("[AdaptiveRetrainer] Reentrenamiento finalizado.")

if __name__ == "__main__":
    AdaptiveRetrainer().retrain_if_needed()
