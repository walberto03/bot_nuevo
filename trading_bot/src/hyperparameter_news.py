# trading_bot/src/hyperparameter_news.py

import optuna
import yaml
from pathlib import Path
from trading_bot.src.news_only_trainer import NewsOnlyTrainer

# (opcional) ruta a config.yaml si quieres guardar los mejores params
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
# Base de datos compartida con el tuning de precios
STUDY_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "optuna_study.db"
STUDY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [32, 48, 64])
    hidden_size   = trial.suggest_categorical("hidden_size", [128, 256, 512])
    dropout       = trial.suggest_float("dropout", 0.1, 0.5)

    trainer = NewsOnlyTrainer(
        learning_rate=learning_rate,
        batch_size=   batch_size,
        hidden_size=  hidden_size,
        dropout=      dropout,
        optuna_mode=  True
    )

    results = trainer.train()
    return results["f1_score"] if results else 0.0

def run_news_hyperparameter_search(n_trials=30, update_config: bool = False):
    print("[Optuna] 📰 Ajustando hiperparámetros (noticias)...")
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{STUDY_DB_PATH}",
        study_name="news_only_tuning",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler()  # Agregado: Mejor sampler para eficiencia
    )
    study.optimize(objective, n_trials=n_trials, timeout=3600)  # Agregado: Timeout de 1 hora para evitar loops infinitos

    best = study.best_params
    print("\n[Optuna] 🚀 Mejor conjunto encontrado para modelo de noticias:")
    print(best)

    if update_config:
        _update_config_yaml(best)
    return best

def _update_config_yaml(best_params: dict):
    if not CONFIG_PATH.exists():
        print(f"[Optuna] ⚠️ No se encontró config en {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    t = cfg.setdefault('news_training', {})
    t.update({
        'learning_rate': float(best_params['learning_rate']),
        'batch_size':    int(best_params['batch_size']),
        'hidden_size':   int(best_params['hidden_size']),
        'dropout':       float(best_params['dropout']),
    })

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[Optuna] 🔄 config.yaml actualizado en {CONFIG_PATH}")

if __name__ == "__main__":
    # Si quieres guardar en config, pasa update_config=True
    run_news_hyperparameter_search(n_trials=30, update_config=False)