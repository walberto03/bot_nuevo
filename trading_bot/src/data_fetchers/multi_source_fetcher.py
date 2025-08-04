# Mostrando el archivo actualizado `multi_source_fetcher.py`
updated_code = r'''
import os
import json
import time
from typing import List, Tuple, Optional, Dict
import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
from trading_bot.config import ALPHA_VANTAGE_API_KEY, FMP_API_KEY, FINNHUB_API_KEY, cfg

class MultiSourceDataFetcher:
    def __init__(self):
        self.api_keys = {
            'finnhub': FINNHUB_API_KEY,
            'fmp': FMP_API_KEY,
            'alpha_vantage': ALPHA_VANTAGE_API_KEY
        }

        # Configuración de directorios
        self.data_directory = 'data'
        self.market_data_cache = os.path.join(self.data_directory, 'market_data_cache')
        self.progress_file = os.path.join(self.data_directory, 'learning_progress.json')

        # Configuración de límites de API
        self.api_limits = {
            'fmp': {'daily_calls': 0, 'last_reset': None},
            'alpha_vantage': {'daily_calls': 0, 'last_reset': None},
            'finnhub': {'minute_calls': 0, 'last_reset': None}
        }

        # Inicialización
        self.setup_directories()
        self.load_progress()

    def setup_directories(self):
        """Crear estructura de directorios necesaria"""
        self.data_directory = os.path.abspath(self.data_directory)
        self.market_data_cache = os.path.abspath(self.market_data_cache)
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.market_data_cache, exist_ok=True)

    def load_progress(self):
        """Cargar progreso del aprendizaje"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'last_processed_date': None,
                'last_symbol': None,
                'total_days_processed': 0
            }

    def save_progress(self):
        """Guardar progreso del aprendizaje"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)

    # -------------------------------
    #  NUEVAS FUNCIONES DE ACTUALIZACIÓN
    # -------------------------------
    def fetch_price_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Descarga histórico OHLCV desde yfinance para el rango dado.
        """
        return yf.download(symbol, start=start, end=end, interval='1h', progress=False)

    def update_prices(self, symbol: str, full_refresh: bool = False):
        """
        Descarga histórico de los últimos 6 meses y actualizaciones incrementales.
        Divide día a día y guarda en caché.
        """
        start_cfg = cfg["date"]["start_cache"]
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        if full_refresh or not self.progress.get('last_processed_date'):
            start_date = start_cfg
        else:
            last = datetime.fromisoformat(self.progress['last_processed_date'])
            start_date = (last + timedelta(days=1)).strftime("%Y-%m-%d")

        if start_date >= today_str:
            return  # Nada que actualizar

        df = self.fetch_price_history(symbol, start_date, today_str)
        # Particionar por día y guardar cada uno
        for single_date, group in df.groupby(df.index.date):
            dt_obj = datetime.combine(single_date, datetime.min.time())
            self.save_to_cache(symbol, dt_obj, group)
            self.progress['last_processed_date'] = single_date.strftime("%Y-%m-%d")

        self.save_progress()

    # -------------------------------
    #  MÉTODOS EXISTENTES
    # -------------------------------
    def check_api_limits(self, api_name: str) -> bool:
        if api_name not in self.api_limits:
            return True

        limit_info = self.api_limits[api_name]
        current_time = datetime.now()

        if limit_info['last_reset'] is None:
            limit_info['last_reset'] = current_time
            limit_info['daily_calls'] = 0
            return True

        # Reset diario o por minuto
        if api_name == 'alpha_vantage':
            if (current_time - limit_info['last_reset']).days >= 1:
                limit_info['daily_calls'] = 0
                limit_info['last_reset'] = current_time
        else:
            if (current_time - limit_info['last_reset']).total_seconds() >= 60:
                limit_info['daily_calls'] = 0
                limit_info['last_reset'] = current_time

        if api_name == 'alpha_vantage':
            return limit_info['daily_calls'] < 500
        elif api_name == 'fmp':
            return limit_info['daily_calls'] < 300
        else:
            return limit_info['daily_calls'] < 30

    def get_cached_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        cache_file = os.path.join(self.market_data_cache, f"{symbol}_{date.strftime('%Y-%m-%d')}.csv")
        if os.path.exists(cache_file):
            try:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

    def save_to_cache(self, symbol: str, date: datetime, data: pd.DataFrame) -> None:
        cache_file = os.path.join(self.market_data_cache, f"{symbol}_{date.strftime('%Y-%m-%d')}.csv")
        data.to_csv(cache_file)

    def get_fmp_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        if not self.check_api_limits('fmp'):
            return None
        try:
            url = (
                f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}'
                f'?apikey={self.api_keys["fmp"]}'
            )
            resp = requests.get(url)
            self.api_limits['fmp']['daily_calls'] += 1
            if resp.status_code == 200:
                arr = resp.json()
                df = pd.DataFrame(arr)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df.sort_index()
        except Exception:
            return None
        return None

    def get_alpha_vantage_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        if not self.check_api_limits('alpha_vantage'):
            return None
        try:
            from_sym, to_sym = symbol[:3], symbol[3:]
            url = (
                'https://www.alphavantage.co/query'
                f'?function=FX_INTRADAY'
                f'&from_symbol={from_sym}'
                f'&to_symbol={to_sym}'
                f'&interval=60min'
                f'&outputsize=full'
                f'&apikey={self.api_keys["alpha_vantage"]}'
            )
            resp = requests.get(url)
            self.api_limits['alpha_vantage']['daily_calls'] += 1
            data = resp.json().get('Time Series FX (60min)', {})
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['Open','High','Low','Close']
            df = df.apply(pd.to_numeric, errors='coerce')
            return df[df.index.date == date.date()].sort_index()
        except Exception:
            return None

    def get_yfinance_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        try:
            ticker = f"{symbol}=X"
            end_dt = date + timedelta(days=1)
            df = yf.download(ticker, start=date, end=end_dt, interval='1h', progress=False)
            return df if not df.empty else None
        except Exception:
            return None

    def get_market_data(self, symbol: str, date: datetime) -> Tuple[Optional[pd.DataFrame], str]:
        # 1) Caché
        cached = self.get_cached_data(symbol, date)
        if cached is not None:
            return cached, 'cache'
        # 2) Proveedores
        for name, fn in [
            ('alpha_vantage', self.get_alpha_vantage_data),
            ('fmp',            self.get_fmp_data),
            ('yfinance',       self.get_yfinance_data)
        ]:
            df = fn(symbol, date)
            if df is not None and not df.empty:
                self.save_to_cache(symbol, date, df)
                return df, name
        return None, 'none'

    def get_news_from_fmp(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        try:
            url = (
                'https://financialmodelingprep.com/api/v3/stock_news'
                f'?tickers={symbol}'
                f'&limit=1000'
                f'&apikey={self.api_keys["fmp"]}'
            )
            resp = requests.get(url)
            if resp.status_code == 200:
                articles = resp.json()
                return [
                    a for a in articles
                    if start_date <= datetime.strptime(a['publishedDate'], '%Y-%m-%d %H:%M:%S') <= end_date
                ]
        except Exception:
            pass
        return []

    def save_progress_and_ask_continue(self, current_date: datetime, symbol: str) -> bool:
        self.progress['last_processed_date'] = current_date.strftime('%Y-%m-%d')
        self.progress['last_symbol'] = symbol
        self.progress['total_days_processed'] += 1
        self.save_progress()
        ans = input(f"Procesados {self.progress['total_days_processed']} días. Continuar? (s/n): ")
        return ans.lower().startswith('s')
'''
with open('/mnt/data/multi_source_fetcher_updated.py', 'w') as f:
    f.write(updated_code)

# Mostrar ruta de archivo generado
'/mnt/data/multi_source_fetcher_updated.py'
