# trading_bot/src/orchestrator.py

from datetime import datetime, timedelta
import pandas as pd
from trading_bot.src.data.db_manager import init_db, load_all_data, save_prices, load_all_news, save_news
from trading_bot.src.adapters.twelve_adapter import TwelveDataAdapter
from trading_bot.src.adapters.news_adapter_transformer import NewsAdapterTransformer
from trading_bot.src.multi_timescale_trainer import MultiTimescaleTrainer
import time

class Orchestrator:
    def __init__(self):
        self.adapter      = TwelveDataAdapter()
        self.trainer      = MultiTimescaleTrainer()
        self.news_manager = NewsAdapterTransformer()
        self.symbols      = ["EURUSD", "GBPUSD"]

    def datos_ya_existentes(self, symbol, start, end, resolution):
        """
        Comprueba si en la BD existen ya precios para 'symbol' y 'resolution'
        en todo el rango [start,end].
        """
        try:
            df = load_all_data()
            col_symbol = "Symbol" if "Symbol" in df.columns else "symbol"
            df_symbol  = df[(df[col_symbol] == symbol) & (df["resolution"] == resolution)]
            if df_symbol.empty:
                return False

            fechas_existentes = set(df_symbol["date"].dt.date)
            step = {
                "15min": timedelta(minutes=15),
                "30min": timedelta(minutes=30),
                "1h":    timedelta(hours=1),
                "4h":    timedelta(hours=4)
            }.get(resolution, timedelta(days=1))

            fecha = start
            while fecha <= end:
                if fecha.date() not in fechas_existentes:
                    return False
                fecha += step
            return True

        except Exception as e:
            print(f"[Orchestrator] ⚠️ Error verificando existencia de datos: {e}")
            return False

    def download_prices(self, start: datetime, end: datetime, intervals):
        for interval in intervals:
            print(f"[Orchestrator] Descargando precios para intervalo: {interval}")
            delta_days = {"15min":45,"30min":90,"1h":180,"4h":720,"1d":1000}.get(interval,30)
            delta = timedelta(days=delta_days)
            for symbol in self.symbols:
                current = start
                while current < end:
                    block_end = min(current + delta, end)
                    if self.datos_ya_existentes(symbol, current, block_end, interval):
                        print(f"[Orchestrator] ⏭️ Ya existen {symbol} {interval} {current}–{block_end}")
                    else:
                        df = self.adapter.fetch(symbol, current, block_end, interval)
                        if df is not None:
                            df["resolution"] = interval
                            save_prices(symbol, df, source="TwelveData", resolution=interval)
                            print(f"[Orchestrator] ✅ Guardados {len(df)} filas de {symbol} {interval} {current}–{block_end}")
                        else:
                            print(f"[Orchestrator] ❌ Error descarga {symbol} {interval} {current}–{block_end}")
                    current = block_end + timedelta(seconds=1)
                    time.sleep(10)

    def noticias_ya_existentes(self, symbol, date):
        """
        Comprueba si ya hay noticias en la tabla 'news' para ese símbolo y fecha exacta.
        """
        try:
            fecha_str = date.strftime("%Y-%m-%d")
            df_news   = load_all_news(symbol=symbol, start=fecha_str, end=fecha_str)
            return not df_news.empty
        except Exception as e:
            print(f"[Orchestrator] ⚠️ Error verificando noticias: {e}")
            return False

    def download_news(self, start: datetime, end: datetime):
        print("[Orchestrator] Descargando noticias históricas...")
        fecha = start
        while fecha.date() <= end.date():
            fecha_str = fecha.strftime("%Y-%m-%d")
            for symbol in self.symbols:
                if self.noticias_ya_existentes(symbol, fecha):
                    print(f"[Orchestrator] ⏭️ Noticias ya existen para {symbol} en {fecha_str}")
                else:
                    arts  = self.news_manager.fetch_news(symbol, fecha_str, fecha_str)
                    score = self.news_manager.analyze_news_sentiment(arts)
                    save_news(symbol, fecha_str, arts, score)
                    print(f"[Orchestrator] 📅 Guardadas {len(arts)} noticias de {symbol} en {fecha_str}")
            fecha += timedelta(days=1)
            time.sleep(1)

    def start(self, full_refresh=False, simulate_from=None):
        init_db()
        end_date = datetime.now()
        start_all = datetime(2022, 1, 1)

        if full_refresh:
            # Fuerza descargas completas
            self.download_prices(start_all, end_date, ["1d","15min","1h","4h"])
            self.download_news(start_all, end_date)
        else:
            # Para cada resolución y símbolo, descarga desde el último timestamp disponible
            data = load_all_data()
            for interval in ["15min","1h","4h","1d"]:
                for symbol in self.symbols:
                    df_sym = data[
                        (data["symbol"] == symbol) & (data["resolution"] == interval)
                    ]
                    if df_sym.empty:
                        start_dt = start_all
                    else:
                        last = df_sym["date"].max()
                        step = {
                            "15min": timedelta(minutes=15),
                            "30min": timedelta(minutes=30),
                            "1h":    timedelta(hours=1),
                            "4h":    timedelta(hours=4)
                        }.get(interval, timedelta(days=1))
                        start_dt = last + step
                    if start_dt < end_date:
                        print(f"[Orchestrator] 📈 {symbol} {interval}: descargando desde {start_dt}")
                        self.download_prices(start_dt, end_date, [interval])

            # Noticias: desde el día siguiente de la última noticia
            for symbol in self.symbols:
                df_news = load_all_news(symbol=symbol, start="2022-01-01", end=end_date.strftime("%Y-%m-%d"))
                if df_news.empty:
                    news_start = start_all
                else:
                    max_date = pd.to_datetime(df_news["date"]).dt.date.max()
                    news_start = datetime.combine(max_date + timedelta(days=1), datetime.min.time())
                if news_start < end_date:
                    print(f"[Orchestrator] 📰 {symbol}: noticias desde {news_start.date()}")
                    self.download_news(news_start, end_date)

        if simulate_from:
            print(f"[Orchestrator] Simulando desde {simulate_from}")

        print("[Orchestrator] Entrenamiento multiescala en marcha…")
        self.trainer.train()
        print("[Orchestrator] Flujo completo finalizado ✅")