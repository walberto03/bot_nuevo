# trading_bot/src/hyperparameter_search.py

import optuna
import yaml
from pathlib import Path
from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer

# Rutas a config y al almacenamiento del estudio Optuna
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
STUDY_DB   = Path(__file__).resolve().parent.parent.parent / "data"   / "optuna_study.db"
STUDY_DB.parent.mkdir(parents=True, exist_ok=True)

def objective(trial):
    # --- Arquitectura ---
    learning_rate    = trial.suggest_float("learning_rate",        1e-6, 1e-2, log=True)
    batch_size       = trial.suggest_categorical("batch_size",    [8,16,32,48,64,128])
    sequence_length  = trial.suggest_categorical("sequence_length",[20,48,72,100,144])
    hidden_size      = trial.suggest_categorical("hidden_size",   [32,64,100,128,256,512])
    num_layers       = trial.suggest_int("num_layers", 1, 5)
    attention_heads  = trial.suggest_categorical("attention_heads",[2,4,6,8,16])
    attention_ff_dim = trial.suggest_categorical("attention_ff_dim",[64,128,256,512])

    # --- Entrenamiento ---
    patience      = trial.suggest_int("early_stopping_patience", 3, 15)
    weight_decay  = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

    # --- Regularización y balanceo de clases ---
    dropout           = trial.suggest_float("dropout",            0.0, 0.7)
    pos_weight_factor = trial.suggest_float("pos_weight_factor",  0.1, 5.0)
    focal_gamma       = trial.suggest_float("focal_gamma",        0.0, 5.0)

    # Montamos el trainer sin pasar classification_threshold
    trainer = MultiTimescaleTrainer(
        learning_rate              = learning_rate,
        batch_size                 = batch_size,
        sequence_length            = sequence_length,
        hidden_size                = hidden_size,
        num_layers                 = num_layers,
        attention_heads            = attention_heads,
        attention_feedforward_dim  = attention_ff_dim,
        early_stopping_patience    = patience,
        weight_decay               = weight_decay,
        dropout                    = dropout,
        pos_weight_factor          = pos_weight_factor,
        focal_gamma                = focal_gamma,
        optuna_mode                = True
    )
    metrics = trainer.train()
    return metrics.get("f1_score", 0.0) if metrics else 0.0

def run_hyperparameter_search(n_trials: int = 50, update_config: bool = True):
    print("[Optuna] 🚀 Iniciando búsqueda de hiperparámetros...")
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{STUDY_DB}",
        study_name="multi_lstm_attention_v2",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler()  # Agregado: Mejor sampler para eficiencia
    )
    study.optimize(objective, n_trials=n_trials, timeout=3600 * 2)  # Agregado: Timeout de 2 horas total para evitar loops infinitos

    best = study.best_params
    print("\n✅ Mejor conjunto encontrado:")
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

    t = cfg.setdefault('training', {})
    t.update({
        'learning_rate':            float(best_params['learning_rate']),
        'batch_size':               int(best_params['batch_size']),
        'sequence_length':          int(best_params['sequence_length']),
        'hidden_size':              int(best_params['hidden_size']),
        'num_layers':               int(best_params['num_layers']),
        'attention_heads':          int(best_params['attention_heads']),
        'attention_ff_dim':         int(best_params['attention_ff_dim']),
        'early_stopping_patience':  int(best_params['early_stopping_patience']),
        'weight_decay':             float(best_params['weight_decay']),
        'dropout':                  float(best_params['dropout']),
        'pos_weight_factor':        float(best_params['pos_weight_factor']),
        'focal_gamma':              float(best_params['focal_gamma']),
    })

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[Optuna] 🔄 config.yaml actualizado en {CONFIG_PATH}")

if __name__ == "__main__":
    run_hyperparameter_search()