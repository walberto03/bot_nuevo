#!/usr/bin/env python3 
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
import optuna

from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer
from trading_bot.src.hyperparameter_search import run_hyperparameter_search
from trading_bot.src.hyperparameter_news import run_news_hyperparameter_search
from trading_bot.src.utils.metric_watcher import MetricWatcher
from trading_bot.src.news_only_trainer import NewsOnlyTrainer
from trading_bot.src.data.db_manager import load_all_data

# ----------------------------------------
# Configuración de Optuna (ubicación .db)
# ----------------------------------------
PROJECT_ROOT      = Path(__file__).resolve().parent
OPTUNA_DB_PATH    = f"sqlite:///{PROJECT_ROOT}/data/optuna_study.db"
OPTUNA_STUDY_NAME = "multi_lstm_attention_v2"


def _post_optuna(metrics):
    watcher = MetricWatcher(tolerance=0.05,
                             min_accuracy=0.60,
                             min_f1_score=0.55)
    if watcher.should_trigger(metrics):
        if input("🚨 Métricas bajas. Ejecutar Optuna? (s/n): ").strip().lower() == 's':
            best = run_hyperparameter_search(n_trials=30)
            print(f"[Optuna] 🧠 Mejores params: {best}")
            if input("🔄 Reentrenar con ellos? (s/n): ").strip().lower() == 's':
                MultiTimescaleTrainer(**best).train()
                print("[MAIN] ✅ Reentrenamiento completado.")


def seleccionar_menu():
    print("\n¿Qué deseas hacer?")
    print("1) Sólo Entrenamiento Multiescala (usando mejores hiperparámetros guardados)")
    print("2) Sólo Simulación desde fecha")
    print("3) Sólo Noticias")
    print("4) Sólo Precios")
    print("5) Flujo Completo (simulación + entrenamiento + predicción futura)")
    print("6) Ajustar hiperparámetros (precios) con Optuna")
    print("7) Ajustar hiperparámetros (noticias) con Optuna")
    print("0) Salir")
    return input("Selecciona (0-7): ").strip()


def cargar_optuna_si_existe():
    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME,
                                  storage=OPTUNA_DB_PATH)
        best_params = study.best_params
        print(f"[Optuna] 🧠 Cargando mejores hiperparámetros almacenados...")
        if 'attention_ff_dim' in best_params:
            best_params['attention_feedforward_dim'] = best_params.pop('attention_ff_dim')
        return best_params
    except Exception as e:
        print(f"[Optuna] ⚠️ No se pudieron cargar los parámetros guardados: {e}")
        return {}


def opcion_1_train_multiescala(best_params, no_optuna):
    print("\n[MAIN] 🚀 Entrenamiento multiescala (modo solo)...")
    trainer = MultiTimescaleTrainer(**best_params)
    report = trainer.train()
    if report:
        f1 = report.get("f1_score")
        print(f"[MAIN] 🎯 Métricas finales: f1_score={f1:.3f}")
        if not no_optuna:
            _post_optuna(report)
    print("[MAIN] ✅ Entrenamiento completado.\n")


def opcion_2_simulacion_desde_fecha(best_params):
    fecha_str = input("Fecha de inicio para simular (YYYY-MM-DD): ").strip()
    try:
        fecha_inicio = datetime.strptime(fecha_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[MAIN] ⚠️ Fecha inválida: {fecha_str}. Formato esperado: YYYY-MM-DD\n")
        return

    hoy = date.today()
    if fecha_inicio >= hoy:
        print(f"[MAIN] ⚠️ La fecha ({fecha_inicio}) debe ser anterior a hoy ({hoy}).\n")
        return

    print(f"\n[MAIN] 🔄 Simulación histórica desde {fecha_inicio} hasta hoy, aprendiendo y prediciendo día a día...")
    from trading_bot.src.orchestrator import Orchestrator as SimOrchestrator

    resultados_sim = []
    fecha_actual = fecha_inicio

    while fecha_actual < hoy:
        siguiente = fecha_actual + timedelta(days=1)

        # (1) Descargar lo que falte
        orch = SimOrchestrator()
        orch.start(full_refresh=False, simulate_from=None)

        # (2) Entrenar/refinar
        trainer = MultiTimescaleTrainer(**best_params)
        trainer.train()

        # (3) Predecir t+1
        try:
            prob, etiqueta_pred, sl, tp, entry, prob_mayor = trainer.predict_next_day()
            print(f"[Simulación] t={fecha_actual} → Predijo {etiqueta_pred} (p={prob:.3f}), SL={sl:.4f}, TP={tp:.4f}")
            print(f"Mayor éxito {prob_mayor:.3f} si {entry:.4f}")
        except Exception as e:
            print(f"[Simulación] ⚠️ No se pudo predecir para {siguiente}: {e}")
            prob, etiqueta_pred = None, None

        # (4) Obtener real t+1
        df_p = load_all_data("1d")
        col_s = "symbol" if "symbol" in df_p.columns else "Symbol"
        col_d = "date" if "date" in df_p.columns else "Date"
        col_c = "close" if "close" in df_p.columns else "Close"
        sym   = trainer.cfg.symbols[0]

        hoy_vals = df_p[(df_p[col_s]==sym) & (df_p[col_d].dt.date==fecha_actual)][col_c].values
        sig_vals = df_p[(df_p[col_s]==sym) & (df_p[col_d].dt.date==siguiente)][col_c].values

        real_mov = None
        if len(hoy_vals) and len(sig_vals):
            real_mov = "SUBE" if sig_vals[0] > hoy_vals[0] else "BAJA"

        resultados_sim.append((fecha_actual, siguiente, prob, etiqueta_pred, real_mov))
        print(f"[Simulación] t={fecha_actual} → Predijo {etiqueta_pred} (p={prob}), real={real_mov}")

        fecha_actual = siguiente

    import pprint; pprint.pprint(resultados_sim)
    print("[MAIN] ✅ Simulación histórica finalizada.\n")


def opcion_3_solo_noticias():
    print("\n[MAIN] 📰 Solo análisis con noticias...")
    res = NewsOnlyTrainer().train()
    if not res:
        print("[MAIN] ⚠️ No se pudo entrenar el modelo solo de noticias.\n")
    else:
        print(f"[MAIN] ✅ News-only f1_score={res['weighted avg']['f1-score']:.3f}\n")


def opcion_4_solo_precios():
    print("\n[MAIN] 💹 Solo análisis con precios...")
    rep = MultiTimescaleTrainer().train()
    if rep:
        print(f"[MAIN] ✅ Prices-only f1_score={rep['f1_score']:.3f}\n")


def opcion_5_flujo_completo(best_params, no_optuna):
    from trading_bot.src.orchestrator import Orchestrator
    orchestrator = Orchestrator()

    respuesta = input("Descargar histórico completo? (s/n): ").strip().lower()
    if respuesta == 's':
        print("[MAIN] ✅ Descargando todo el histórico de precios y noticias hasta hoy...")
        orchestrator.start(full_refresh=True, simulate_from=None)
    else:
        print("[MAIN] ℹ️ Descargando sólo lo que falte históricamente (desde 2022)…")
        orchestrator.start(full_refresh=False, simulate_from=None)

    # entrenamiento + evaluación final
    trainer = MultiTimescaleTrainer(**best_params)
    report  = trainer.train()
    if report:
        f1 = report.get("f1_score")
        print(f"[MAIN] 🎯 Flujo completo f1_score={f1:.3f}")
        if not no_optuna:
            _post_optuna(report)

    # predicción única para mañana
    try:
        p, et = trainer.predict_next_day()
        print(f"[MAIN] 🔮 Predicción para el próximo día: prob_subida={p:.3f} → {et}")
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo predecir el día siguiente: {e}")

    print("[MAIN] ✅ Proceso terminado.\n")


def opcion_6_optuna_precios():
    print("\n[Optuna] 🔍 Ajustando hiperparámetros (precios)...")
    best = run_hyperparameter_search(n_trials=30)
    print(f"[Optuna] ✅ Mejores parámetros: {best}\n")


def opcion_7_optuna_noticias():
    print("\n[Optuna] 📰 Ajustando hiperparámetros (noticias)...")
    best = run_news_hyperparameter_search(n_trials=30)
    print(f"[Optuna] ✅ Mejores parámetros (noticias): {best}\n")


if __name__ == "__main__":
    load_dotenv()
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    no_optuna = False
    best_params = cargar_optuna_si_existe()

    while True:
        opt = seleccionar_menu()
        if opt == '0':
            print("👋 Saliendo..."); sys.exit(0)
        elif opt == '1':
            opcion_1_train_multiescala(best_params, no_optuna)
        elif opt == '2':
            opcion_2_simulacion_desde_fecha(best_params)
        elif opt == '3':
            opcion_3_solo_noticias()
        elif opt == '4':
            opcion_4_solo_precios()
        elif opt == '5':
            opcion_5_flujo_completo(best_params, no_optuna)
        elif opt == '6':
            opcion_6_optuna_precios()
            best_params = cargar_optuna_si_existe()
        elif opt == '7':
            opcion_7_optuna_noticias()
        else:
            print("⚠️ Opción inválida. Elige un número entre 0 y 7.\n")