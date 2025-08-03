# Archivo: trading_bot/src/optimizer.py

import optuna  # Asegúrate de instalarlo con: pip install optuna
from trading_bot.brain import TradingBrain
from trading_bot.config import TradingConfig


def objective(trial):
    """
    Función objetivo para Optuna: ajusta parámetros y devuelve -Sharpe para minimizar.
    """
    # 1) Carga la configuración base
    cfg = TradingConfig()

    # 2) Sobrescribe hiperparámetros desde el trial
    cfg.sma_short    = trial.suggest_int("sma_short", 5, 50)
    cfg.sma_long     = trial.suggest_int("sma_long", 20, 200)
    cfg.risk_reward  = trial.suggest_float("risk_reward", 1.0, 5.0)

    # 3) Ejecuta el bot con la configuración modificada
    brain = TradingBrain(cfg)
    results = brain.run(full_refresh=False)

    # 4) Queremos maximizar Sharpe, así que devolvemos su negativo
    return -results.get("sharpe", 0.0)


def optimize(n_trials: int = 50):
    """
    Ejecuta el estudio de Optuna para n_trials iteraciones.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Mejores parámetros:", study.best_params)
    return study.best_params