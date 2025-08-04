#!/usr/bin/env python3
import os
import sys
import requests
import time
import argparse
import MetaTrader5 as mt5  # Para trades demo
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
import optuna
from trading_bot.config import cfg

from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer
from trading_bot.src.hyperparameter_search import run_hyperparameter_search
from trading_bot.src.hyperparameter_news import run_news_hyperparameter_search
from trading_bot.src.utils.metric_watcher import MetricWatcher
from trading_bot.src.news_only_trainer import NewsOnlyTrainer
from trading_bot.src.data.db_manager import load_all_data, save_trade_result
from trading_bot.src.adapters.telegram_reader import TelegramReader

PROJECT_ROOT = Path(__file__).resolve().parent
OPTUNA_DB_PATH = f"sqlite:///{PROJECT_ROOT}/data/optuna_study.db"
OPTUNA_STUDY_NAME = "multi_lstm_attention_v2"

def send_notification(message: str):
    if cfg["notifications"].get("enable_notifications", False):
        telegram_token = cfg["notifications"].get("telegram_token")
        telegram_chat_id = cfg["notifications"].get("telegram_chat_id")
        if telegram_token and telegram_chat_id:
            telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            try:
                requests.post(telegram_url, json={"chat_id": telegram_chat_id, "text": message})
                print("[Notification] ✅ Mensaje enviado a Telegram")
            except Exception as e:
                print(f"[Notification] ⚠️ Error enviando a Telegram: {e}")

def _post_optuna(metrics):
    watcher = MetricWatcher(tolerance=0.05, min_accuracy=0.60, min_f1_score=0.55)
    if watcher.should_trigger(metrics):
        if input("🚨 Métricas bajas. Ejecutar Optuna? (s/n): ").strip().lower() == 's':
            best = run_hyperparameter_search(n_trials=30)
            print(f"[Optuna] 🧠 Mejores params: {best}")
            if input("🔄 Reentrenar con ellos? (s/n): ").strip().lower() == 's':
                MultiTimescaleTrainer(**best).train()
                print("[MAIN] ✅ Reentrenamiento completado.")

def seleccionar_menu():
    print("\n¿Qué deseas hacer?")
    print("1) Sólo Entrenamiento Multiescala")
    print("2) Sólo Simulación desde fecha")
    print("3) Sólo Noticias")
    print("4) Sólo Precios")
    print("5) Flujo Completo (simulación + entrenamiento + predicción)")
    print("6) Ajustar hiperparámetros (precios) con Optuna")
    print("7) Ajustar hiperparámetros (noticias) con Optuna")
    print("0) Salir")
    return input("Selecciona (0-7): ").strip()

def cargar_optuna_si_existe():
    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_DB_PATH)
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

        orch = SimOrchestrator()
        orch.start(full_refresh=False, simulate_from=None)

        trainer = MultiTimescaleTrainer(**best_params)
        trainer.train()

        try:
            prob, etiqueta_pred, sl, tp, entry, prob_mayor = trainer.predict_next_day()
            message = (f"[Simulación] t={fecha_actual} → Predijo {etiqueta_pred} (p={prob*100:.1f}%), "
                       f"Mayor éxito {prob_mayor*100:.1f}% si precio alcanza {entry:.4f}, "
                       f"SL={sl:.4f}, TP={tp:.4f}")
            print(message)
            send_notification(message)
            execute_mt5_trade(etiqueta_pred, sl, tp, entry, prob)  # Nuevo: Trade demo
        except Exception as e:
            print(f"[Simulación] ⚠️ No se pudo predecir para {siguiente}: {e}")
            prob, etiqueta_pred = None, None

        df_p = load_all_data("1d")
        col_s = "Symbol" if "Symbol" in df_p.columns else "symbol"
        col_d = "Date" if "Date" in df_p.columns else "date"
        col_c = "Close" if "Close" in df_p.columns else "close"
        sym = "XAUUSD"  # Foco en oro

        hoy_vals = df_p[(df_p[col_s]==sym) & (df_p[col_d].dt.date==fecha_actual)][col_c].values
        sig_vals = df_p[(df_p[col_s]==sym) & (df_p[col_d].dt.date==siguiente)][col_c].values

        real_mov = None
        if len(hoy_vals) and len(sig_vals):
            real_mov = "SUBE" if sig_vals[0] > hoy_vals[0] else "BAJA"
            save_trade_result(sym, fecha_actual, etiqueta_pred, real_mov, prob, sl, tp)  # Aprendizaje

        resultados_sim.append((fecha_actual, siguiente, prob, etiqueta_pred, real_mov))
        print(f"[Simulación] t={fecha_actual} → Real={real_mov}")

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

    trainer = MultiTimescaleTrainer(**best_params)
    report = trainer.train()
    if report:
        f1 = report.get("f1_score")
        print(f"[MAIN] 🎯 Flujo completo f1_score={f1:.3f}")
        if not no_optuna:
            _post_optuna(report)

    try:
        prob, etiqueta_pred, sl, tp, entry, prob_mayor = trainer.predict_next_day()
        message = (f"[MAIN] 🔮 Predicción para el próximo día: {etiqueta_pred} con {prob*100:.1f}% precisión.\n"
                   f"Mayor éxito {prob_mayor*100:.1f}% si precio alcanza {entry:.4f}.\n"
                   f"SL: {sl:.4f}, TP: {tp:.4f}")
        print(message)
        send_notification(message)
        execute_mt5_trade(etiqueta_pred, sl, tp, entry, prob)  # Nuevo: Trade demo
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

def execute_mt5_trade(etiqueta, sl, tp, entry, prob):
    if not mt5.initialize():
        print("MT5 no inicializado")
        return
    symbol = "XAUUSD"
    lot = 0.1  # Tamaño demo
    price = mt5.symbol_info_tick(symbol).ask if etiqueta == "SUBE" else mt5.symbol_info_tick(symbol).bid
    if etiqueta == "SUBE":
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": f"Bot Prediction p={prob:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
    else:  # BAJA
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": f"Bot Prediction p={prob:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Error en trade: {result.comment}")
    else:
        print(f"Trade demo ejecutado: {etiqueta} at {price}, SL={sl}, TP={tp}")

def save_trade_result(symbol, date, pred, real, prob, sl, tp):
    from trading_bot.src.data.db_manager import save_trade_result  # Ajusta db_manager.py
    outcome = 1 if pred == real else 0
    save_trade_result(symbol, date, pred, real, prob, sl, tp, outcome)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot de trading 24/5 con MT5 demo")
    parser.add_argument('--run-24-5', action='store_true', help="Ejecuta en modo 24/5")
    args = parser.parse_args()

    load_dotenv()
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    no_optuna = False
    best_params = cargar_optuna_si_existe()

    if mt5.initialize(login=123456, password="demo", server="ICMarketsSC-Demo"):  # Configura demo
        print("MT5 inicializado")
    else:
        print("MT5 no inicializado, verifica login/password/server")

    if args.run_24_5:
        print("[MAIN] 🚀 Modo 24/5 activado. Iniciando loop continuo...")
        telegram_reader = TelegramReader()
        import threading
        t = threading.Thread(target=telegram_reader.start_polling, daemon=True)
        t.start()
        while True:
            opcion_5_flujo_completo(best_params, no_optuna)
            time.sleep(3600)  # Retrain cada hora
    else:
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
