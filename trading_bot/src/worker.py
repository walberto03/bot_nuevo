import pandas as pd
from trading_bot.config import TradingConfig
from trading_bot.src.adapters.market_adapter import MarketAdapter
from trading_bot.src.adapters.sentiment_adapter import SentimentAdapter
from trading_bot.src.adapters.news_adapter import NewsAdapter
from trading_bot.src.data.db_manager import init_db, save_prices
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.multi_timescale_decision_maker import MultiTimescaleDecisionMaker
from trading_bot.src.risk_manager import RiskManager
from trading_bot.src.backtester import Backtester
from trading_bot.trading_brain import TradingBrain
from datetime import datetime
import time
import socket
from plyer import notification
import os
import json

TWITTER_RESET_DAY = 19
NEWSAPI_LIMIT = 100
MAX_FAILS = 3
MAX_TOTAL_FAILS = 6
PROGRESS_PATH = "progreso_descarga.json"

class Worker:
    def __init__(self):
        self.cfg = TradingConfig()
        init_db()

        self.market = MarketAdapter()
        self.sent = SentimentAdapter()
        self.news = NewsAdapter()
        self.pa = PatternAnalyzer()
        self.dm = MultiTimescaleDecisionMaker()
        self.risk = RiskManager(self.cfg)
        self.bt = Backtester()
        self.brain = TradingBrain()

        self.cache: dict[str, pd.DataFrame] = {}
        self.fail_counter = 0
        self.total_fails = 0

        if self._has_internet():
            self._mostrar_popup("El bot ha iniciado ejecución automática.")

    def _has_internet(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False

    def _mostrar_popup(self, mensaje):
        try:
            notification.notify(
                title="Notificación del bot de trading",
                message=mensaje,
                app_name="TradingBot",
                timeout=10
            )
        except Exception as e:
            print(f"[Notificación] Error mostrando mensaje: {e}")

    def _guardar_progreso(self, fecha):
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"ultimo_dia": fecha.strftime("%Y-%m-%d")}, f)

    def update_single_day(self, date, timeframe="1d"):
        resumen = []
        any_success = False

        for sym in self.cfg.symbols:
            try:
                df = self.market.fetch(sym, date, date, interval=timeframe)
                source = getattr(self.market, "last_source_used", "desconocida")

                if df is not None and not df.empty:
                    save_prices(sym, df, source=source)
                    self.cache[(sym, timeframe)] = df
                    resumen.append((sym, source, len(df), timeframe))
                    print(f"[MarketAdapter] ✅ {sym} ({source}) descargado {len(df)} velas para {date.date()} ({timeframe}).")
                    any_success = True
                else:
                    resumen.append((sym, source, 0, timeframe))
                    print(f"[MarketAdapter] ⚠️ {sym} ({source}) sin datos en {date.date()} ({timeframe}).")

            except Exception as e:
                resumen.append((sym, f"Error: {e}", 0, timeframe))
                print(f"[MarketAdapter] ❌ {sym} Error: {e}")

        if any_success:
            self.fail_counter = 0
            self._guardar_progreso(date)
        else:
            self.fail_counter += 1
            self.total_fails += 1

            if self.fail_counter == MAX_FAILS:
                msg = f"🚧 {MAX_FAILS} días consecutivos fallidos. Esperando 6 horas antes de reintentar..."
                print(f"[Alerta] {msg}")
                self._mostrar_popup(msg)
                time.sleep(6 * 60 * 60)

            if self.fail_counter > MAX_FAILS:
                print(f"[Alerta] ⚠️ Se reintentó pero aún no hay datos. Continuando con siguiente día...")

            if self.total_fails >= MAX_TOTAL_FAILS:
                msg = f"❌ Se acumularon {MAX_TOTAL_FAILS} días fallidos. Verificar conectividad o límites de API."
                print(f"[Diagnóstico] {msg}")
                self._mostrar_popup(msg)

        return resumen

    def generate_signals_for_day(self, date, remaining_news_calls=100):
        signals = []
        twitter_scores = {}
        used_news_calls = 0

        if datetime.utcnow().day < TWITTER_RESET_DAY:
            print(f"[Twitter] 🚫 Límite mensual alcanzado. Twitter bloqueado hasta el {TWITTER_RESET_DAY} de mayo.")
        else:
            twitter_scores = self.sent.fetch_batch_twitter_sentiment(
                self.cfg.symbols, max_tweets=100
            )
            print("[Twitter] ✔️ Sentimiento de Twitter obtenido.")

        for sym in self.cfg.symbols:
            df_hour = self.cache.get((sym, "1h"))
            df_day = self.cache.get((sym, "1d"))
            if df_hour is None or df_hour.empty or df_day is None or df_day.empty:
                continue

            df_day_filtered = df_day.loc[df_day.index.date == date.date()]
            if df_day_filtered.empty:
                continue

            vol = (df_day_filtered["High"] - df_day_filtered["Low"]).mean()
            tw_score = twitter_scores.get(sym, 0.0)

            if used_news_calls >= NEWSAPI_LIMIT:
                print(f"[NewsAPI] 🚫 Límite diario alcanzado para {sym} en {date.date()}.")
                nw_score = 0.0
            else:
                articles = self.news.fetch_news(
                    symbol=sym,
                    start=date.strftime("%Y-%m-%d"),
                    end=date.strftime("%Y-%m-%d"),
                    max_pages=3,
                )
                used_news_calls += 1

                if articles:
                    print(f"[NewsAPI] ✔️ {len(articles)} artículos para {sym} en {date.date()}.")
                else:
                    print(f"[NewsAPI] ⚠️ Sin artículos para {sym} en {date.date()}.")

                pos = {"gain", "up", "bull", "rise", "strong", "surge", "rally"}
                neg = {"loss", "down", "bear", "fall", "weak", "slump", "crash"}
                news_scores = []
                for art in articles:
                    words = (art["title"] + " " + art.get("description", "")).lower().split()
                    p = sum(w in pos for w in words)
                    n = sum(w in neg for w in words)
                    news_scores.append(0.0 if (p + n) == 0 else (p - n) / (p + n))
                nw_score = sum(news_scores) / len(news_scores) if news_scores else 0.0

            sentiment = (tw_score + nw_score) / 2

            patterns = self.pa.find_patterns(df_day_filtered, sentiment)
            historical_insights = self.brain.learner.memory.get(sym, [])
            order = self.dm.make_decision(patterns, sentiment, historical_insights, df_hour)

            stop_pct = self.risk.compute_stop(vol)
            order["size"] = self.risk.adjust_size(100_000, stop_pct)
            order["date"] = date
            order["symbol"] = sym
            signals.append(order)

        return signals, used_news_calls
