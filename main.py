
import os
import sys
import requests
import time
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
import optuna
import MetaTrader5 as mt5
from threading import Thread
from trading_bot.config import cfg
from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer
from trading_bot.src.hyperparameter_search import run_hyperparameter_search
from trading_bot.src.hyperparameter_news import run_news_hyperparameter_search
from trading_bot.src.utils.metric_watcher import MetricWatcher
from trading_bot.src.news_only_trainer import NewsOnlyTrainer
from trading_bot.src.data.db_manager import load_all_data, save_trade_result
from trading_bot.src.orchestrator import Orchestrator
from trading_bot.src.adapters.telegram_reader import TelegramReader
from bs4 import BeautifulSoup

# Configuración de Optuna (ubicación .db)
PROJECT_ROOT = Path(__file__).resolve().parent  # bot_nuevo/
OPTUNA_DB_PATH = f"sqlite:///{PROJECT_ROOT}/trading_bot/data/optuna_study.db"
OPTUNA_STUDY_NAME = "multi_lstm_attention_v2"

def get_vip_signals_alternative():
    """Scraping alternativo para señales de @Sureshotfx_Signals_Freefx."""
    try:
        response = requests.get("https://t.me/s/Sureshotfx_Signals_Freefx")
        soup = BeautifulSoup(response.text, 'html.parser')
        messages = soup.find_all('div', class_='tgme_widget_message_text')
        pattern = r"(BUY|SELL) GOLD @(\d+\.?\d*) TP1: (\d+\.?\d*) TP2: (\d+\.?\d*)( \+{3})? SL: (\d+\.?\d*|PREMIUM)"
        signals = []
        for msg in messages:
            match = re.search(pattern, msg.text, re.IGNORECASE)
            if match:
                signals.append({
                    "action": match.group(1),
                    "entry": float(match.group(2)),
                    "tp1": float(match.group(3)),
                    "tp2": float(match.group(4)),
                    "over": bool(match.group(5)),
                    "sl": float(match.group(6)) if match.group(6) != "PREMIUM" else None
                })
        return signals
    except Exception as e:
        print(f"[VIP Alternative] Error: {e}")
        return []

def send_notification(message: str):
    """Envía notificaciones a Telegram si está habilitado en config."""
    if cfg.get("notifications", {}).get("enable_notifications", False):
        telegram_token = cfg["notifications"].get("telegram_token")
        telegram_chat_id = cfg["notifications"].get("telegram_chat_id")
        if telegram_token and telegram_chat_id:
            telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            try:
                response = requests.post(telegram_url, json={"chat_id": telegram_chat_id, "text": message})
                response.raise_for_status()
                print("[Notification] ✅ Mensaje enviado a Telegram")
            except Exception as e:
                print(f"[Notification] ⚠️ Error enviando a Telegram: {e}")

def _post_optuna(metrics):
    """Verifica métricas y ejecuta Optuna si son bajas."""
    watcher = MetricWatcher(tolerance=0.05, min_accuracy=0.60, min_f1_score=0.55)
    if watcher.should_trigger(metrics):
        if input("🚨 Métricas bajas. Ejecutar Optuna? (s/n): ").strip().lower() == 's':
            best = run_hyperparameter_search(n_trials=30)
            print(f"[Optuna] 🧠 Mejores params: {best}")
            if input("🔄 Reentrenar con ellos? (s/n): ").strip().lower() == 's':
                MultiTimescaleTrainer(**best).train()
                print("[MAIN] ✅ Reentrenamiento completado.")

def seleccionar_menu():
    """Menú interactivo para pruebas manuales."""
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
    """Carga hiperparámetros óptimos de Optuna si existen."""
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
    """Entrenamiento solo con datos multiescala."""
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
    """Simulación histórica día a día desde una fecha dada."""
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
    print(f"\n[MAIN] 🔄 Simulación histórica desde {fecha_inicio} hasta hoy...")
    resultados_sim = []
    fecha_actual = fecha_inicio
    while fecha_actual < hoy:
        siguiente = fecha_actual + timedelta(days=1)
        orch = Orchestrator()
        orch.start(full_refresh=False, simulate_from=None)
        trainer = MultiTimescaleTrainer(**best_params)
        report = trainer.train()
        if report:
            print(f"[MAIN] 🎯 F1-score: {report.get('f1_score', 0):.3f}")
        try:
            prob, etiqueta_pred, sl, tp, entry, prob_mayor = trainer.predict_next_day()
            message = (f"[Simulación] t={fecha_actual} → Predijo {etiqueta_pred} (p={prob*100:.1f}%), "
                       f"Mayor éxito {prob_mayor*100:.1f}% si precio alcanza {entry:.4f}, "
                       f"SL={sl:.4f}, TP={tp:.4f}")
            print(message)
            send_notification(message)
            execute_mt5_trade(etiqueta_pred, sl, tp, entry, prob)
        except Exception as e:
            print(f"[Simulación] ⚠️ No se pudo predecir para {siguiente}: {e}")
            prob, etiqueta_pred = None, None
        df_p = load_all_data("1d")
        col_s = "symbol" if "symbol" in df_p.columns else "Symbol"
        col_d = "date" if "date" in df_p.columns else "Date"
        col_c = "close" if "close" in df_p.columns else "Close"
        sym = trainer.cfg.symbols[0]  # Ej. XAUUSD
        hoy_vals = df_p[(df_p[col_s] == sym) & (df_p[col_d].dt.date == fecha_actual)][col_c].values
        sig_vals = df_p[(df_p[col_s] == sym) & (df_p[col_d].dt.date == siguiente)][col_c].values
        real_mov = None
        if len(hoy_vals) and len(sig_vals):
            real_mov = "SUBE" if sig_vals[0] > hoy_vals[0] else "BAJA"
            save_trade_result(sym, fecha_actual.isoformat(), etiqueta_pred, real_mov, prob, sl, tp)
        resultados_sim.append((fecha_actual, siguiente, prob, etiqueta_pred, real_mov))
        print(f"[Simulación] t={fecha_actual} → Real={real_mov}")
        fecha_actual = siguiente
    import pprint
    pprint.pprint(resultados_sim)
    print("[MAIN] ✅ Simulación histórica finalizada.\n")

def opcion_3_solo_noticias():
    """Entrenamiento solo con noticias."""
    print("\n[MAIN] 📰 Solo análisis con noticias...")
    res = NewsOnlyTrainer().train()
    if not res:
        print("[MAIN] ⚠️ No se pudo entrenar el modelo solo de noticias.\n")
    else:
        f1 = res['weighted avg']['f1-score']
        print(f"[MAIN] ✅ News-only f1_score={f1:.3f}\n")
        send_notification(f"[MAIN] News-only f1_score={f1:.3f}")

def opcion_4_solo_precios():
    """Entrenamiento solo con precios."""
    print("\n[MAIN] 💹 Solo análisis con precios...")
    rep = MultiTimescaleTrainer().train()
    if rep:
        f1 = rep.get("f1_score")
        print(f"[MAIN] ✅ Prices-only f1_score={f1:.3f}\n")
        send_notification(f"[MAIN] Prices-only f1_score={f1:.3f}")

def opcion_5_flujo_completo(best_params, no_optuna):
    """Flujo completo: descarga, entrenamiento, predicción."""
    orchestrator = Orchestrator()
    respuesta = input("Descargar histórico completo? (s/n): ").strip().lower()
    if respuesta == 's':
        print("[MAIN] ✅ Descargando todo el histórico de precios y noticias...")
        orchestrator.start(full_refresh=True, simulate_from=None)
    else:
        print("[MAIN] ℹ️ Descargando solo lo que falte (desde 2022)...")
        orchestrator.start(full_refresh=False, simulate_from=None)
    trainer = MultiTimescaleTrainer(**best_params)
    report = trainer.train()
    if report:
        f1 = report.get("f1_score")
        print(f"[MAIN] 🎯 Flujo completo f1_score={f1:.3f}")
        send_notification(f"[MAIN] Flujo completo f1_score={f1:.3f}")
        if not no_optuna:
            _post_optuna(report)
    try:
        prob, etiqueta, sl, tp, entry, prob_mayor = trainer.predict_next_day()
        message = (f"[MAIN] 🔮 Predicción para el próximo día: {etiqueta} con {prob*100:.1f}% precisión.\n"
                   f"Mayor éxito {prob_mayor*100:.1f}% si precio alcanza {entry:.4f}.\n"
                   f"SL: {sl:.4f}, TP: {tp:.4f}")
        print(message)
        send_notification(message)
        execute_mt5_trade(etiqueta, sl, tp, entry, prob)
    except Exception as e:
        print(f"[MAIN] ⚠️ No se pudo predecir el día siguiente: {e}")
    print("[MAIN] ✅ Proceso terminado.\n")

def opcion_6_optuna_precios():
    """Ajuste de hiperparámetros para precios con Optuna."""
    print("\n[Optuna] 🔍 Ajustando hiperparámetros (precios)...")
    best = run_hyperparameter_search(n_trials=30)
    print(f"[Optuna] ✅ Mejores parámetros: {best}\n")
    send_notification(f"[Optuna] Mejores parámetros (precios): {best}")

def opcion_7_optuna_noticias():
    """Ajuste de hiperparámetros para noticias con Optuna."""
    print("\n[Optuna] 📰 Ajustando hiperparámetros (noticias)...")
    best = run_news_hyperparameter_search(n_trials=30)
    print(f"[Optuna] ✅ Mejores parámetros (noticias): {best}\n")
    send_notification(f"[Optuna] Mejores parámetros (noticias): {best}")

def execute_mt5_trade(etiqueta, sl, tp, entry, prob):
    """Ejecuta trade demo en MT5."""
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_password = os.getenv("MT5_PASSWORD")
    mt5_server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    if not mt5.initialize(login=int(mt5_login) if mt5_login else 0, password=mt5_password, server=mt5_server):
        print("[MT5] ⚠️ Error al inicializar MT5")
        return
    symbol = "XAUUSD"  # Foco en oro
    lot = 0.1  # Tamaño demo
    price = mt5.symbol_info_tick(symbol).ask if etiqueta == "SUBE" else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if etiqueta == "SUBE" else mt5.ORDER_TYPE_SELL,
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
        print(f"[MT5] ⚠️ Error en trade: {result.comment}")
    else:
        print(f"[MT5] ✅ Trade demo ejecutado: {etiqueta} at {price}, SL={sl}, TP={tp}")
        send_notification(f"[MT5] Trade demo ejecutado: {etiqueta} at {price}, SL={sl}, TP={tp}")
        save_trade_result(symbol, datetime.now().isoformat(), etiqueta, None, prob, sl, tp)

def run_24_5():
    """Modo 24/5: Descarga, entrena, predice, usa Telegram y MT5."""
    load_dotenv()
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    best_params = cargar_optuna_si_existe()
    trainer = MultiTimescaleTrainer(**best_params)
    orchestrator = Orchestrator()
    telegram_reader = TelegramReader()
    Thread(target=telegram_reader.start_polling, daemon=True).start()
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_password = os.getenv("MT5_PASSWORD")
    mt5_server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    if not mt5.initialize(login=int(mt5_login) if mt5_login else 0, password=mt5_password, server=mt5_server):
        print("[MT5] ⚠️ Error al inicializar MT5")
        sys.exit(1)
    while True:
        now = datetime.now()
        if now.weekday() < 5:  # Lunes a viernes
            print(f"[MAIN] 🚀 Ciclo 24/5: {now}")
            try:
                orchestrator.start(full_refresh=False, simulate_from=None)
                report = trainer.train()
                if report:
                    f1 = report.get("f1_score")
                    print(f"[MAIN] 🎯 F1-score: {f1:.3f}")
                    send_notification(f"[MAIN] F1-score: {f1:.3f}")
                prob, etiqueta, sl, tp, entry, prob_mayor = trainer.predict_next_day()
                sentiment = telegram_reader.get_signals_sentiment()
                vip_signals = get_vip_signals_alternative()
                vip_sentiment = sum(1 for s in vip_signals if s['action'] == 'BUY') - sum(1 for s in vip_signals if s['action'] == 'SELL')
                vip_sentiment = vip_sentiment / max(1, len(vip_signals)) if vip_signals else 0
                final_prob = prob + (sentiment + vip_sentiment) * 0.05  # Combinar sentiments
                message = (f"[MAIN] 🔮 Predicción: {etiqueta}, Prob: {final_prob*100:.1f}% (sentiment: {sentiment:.2f}, vip: {vip_sentiment:.2f}), "
                           f"SL: {sl:.4f}, TP: {tp:.4f}, Mayor éxito {prob_mayor*100:.1f}% si entra en {entry:.4f}")
                print(message)
                send_notification(message)
                if final_prob > 0.6:  # Umbral para trade
                    execute_mt5_trade(etiqueta, sl, tp, entry, final_prob)
            except Exception as e:
                print(f"[MAIN] ⚠️ Error en ciclo: {e}")
                send_notification(f"[MAIN] Error en ciclo: {e}")
            print("[MAIN] ⏳ Esperando 1 hora...")
            time.sleep(3600)
        else:
            print("[MAIN] 🛌 Fin de semana, esperando...")
            send_notification("[MAIN] Fin de semana, bot en espera")
            time.sleep(3600)

if __name__ == "__main__":
    load_dotenv()
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    parser = argparse.ArgumentParser(description="Bot de trading 24/5 con MT5 demo")
    parser.add_argument('--run-24-5', action='store_true', help="Ejecuta en modo 24/5")
    args = parser.parse_args()
    if args.run_24_5:
        run_24_5()
    else:
        no_optuna = False
        best_params = cargar_optuna_si_existe()
        while True:
            opt = seleccionar_menu()
            if opt == '0':
                print("👋 Saliendo...")
                sys.exit(0)
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
