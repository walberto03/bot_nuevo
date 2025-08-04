import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

class MultiSourceDataFetcher:
    def __init__(self):
        self.api_keys = {
            'finnhub': 'd0087i1r01qud9qm5r2gd0087i1r01qud9qm5r30',
            'fmp': 'rz7oCaBfuJrB5B2K1sq0Y1rhLzaF2EdZ',
            'alpha_vantage': 'PP5KHR7ZHR2SG3ZY'
        }
        # ... (resto del __init__ se mantiene igual)

    def get_fmp_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Obtener datos de Financial Modeling Prep"""
        if not self.check_api_limits('fmp'):
            return None
            
        try:
            # URL actualizada con el formato correcto para FMP
            url = (f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}'
                  f'?apikey={self.api_keys["fmp"]}')
            
            params = {
                'from': date.strftime('%Y-%m-%d'),
                'to': (date + timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params)
            self.api_limits['fmp']['calls'] += 1
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()  # Asegurar orden cronológico
                    return df
            return None
        except Exception as e:
            print(f"Error en FMP: {e}")
            return None

    def get_alpha_vantage_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Obtener datos de Alpha Vantage"""
        if not self.check_api_limits('alpha_vantage'):
            return None
            
        try:
            from_currency = symbol[:3]
            to_currency = symbol[3:]
            
            url = ('https://www.alphavantage.co/query'
                  f'?function=FX_INTRADAY'
                  f'&from_symbol={from_currency}'
                  f'&to_symbol={to_currency}'
                  f'&interval=60min'
                  f'&outputsize=full'
                  f'&apikey={self.api_keys["alpha_vantage"]}')
            
            response = requests.get(url)
            self.api_limits['alpha_vantage']['calls'] += 1
            
            if response.status_code == 200:
                data = response.json()
                if 'Time Series FX (60min)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series FX (60min)'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Renombrar columnas
                    df.columns = ['Open', 'High', 'Low', 'Close']
                    
                    # Convertir a valores numéricos
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Filtrar solo los datos del día solicitado
                    mask = (df.index.date == date.date())
                    return df[mask]
            return None
        except Exception as e:
            print(f"Error en Alpha Vantage: {e}")
            return None

    def get_news_from_fmp(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Obtener noticias de Financial Modeling Prep"""
        try:
            url = (f'https://financialmodelingprep.com/api/v3/stock_news'
                  f'?tickers={symbol}'
                  f'&limit=1000'
                  f'&apikey={self.api_keys["fmp"]}')
            
            response = requests.get(url)
            
            if response.status_code == 200:
                news = response.json()
                # Filtrar noticias por fecha
                filtered_news = [
                    item for item in news
                    if start_date <= datetime.strptime(item['publishedDate'], '%Y-%m-%d %H:%M:%S')
                    <= end_date
                ]
                return filtered_news
            return []
        except Exception as e:
            print(f"Error obteniendo noticias de FMP: {e}")
            return []

    def get_market_data(self, symbol: str, date: datetime) -> Tuple[Optional[pd.DataFrame], str]:
        """Obtener datos del mercado usando múltiples fuentes"""
        # Verificar caché primero
        cached_data = self.get_cached_data(symbol, date)
        if cached_data is not None:
            return cached_data, 'cache'
            
        # Intentar cada API en orden de preferencia
        apis_to_try = [
            ('fmp', self.get_fmp_data),
            ('alpha_vantage', self.get_alpha_vantage_data),
            ('finnhub', self.get_finnhub_data),
            ('yfinance', self.get_yfinance_data)
        ]
        
        for api_name, api_func in apis_to_try:
            print(f"Intentando obtener datos de {api_name} para {symbol}...")
            data = api_func(symbol, date)
            
            if data is not None and not data.empty:
                print(f"Datos obtenidos exitosamente de {api_name}")
                self.save_to_cache(symbol, date, data)
                return data, api_name
            
            if not self.check_api_limits(api_name):
                print(f"Límite de API alcanzado para {api_name}")
                continue
                
        print(f"No se pudieron obtener datos para {symbol} de ninguna fuente")
        return None, 'none'

    def save_progress_and_ask_continue(self, current_date: datetime, symbol: str) -> bool:
        """Guardar progreso y preguntar si continuar"""
        self.progress['last_processed_date'] = current_date.strftime('%Y-%m-%d')
        self.progress['last_symbol'] = symbol
        self.save_progress()
        
        print("\nEstado actual del proceso:")
        print(f"Última fecha procesada: {current_date.date()}")
        print(f"Último símbolo: {symbol}")
        print(f"Total de días procesados: {self.progress['total_days_processed']}")
        
        response = input("\n¿Desea continuar con el proceso de aprendizaje? (s/n): ")
        return response.lower() == 's'

class APIUsageTracker:
    def __init__(self):
        self.usage_file = 'data/api_usage.json'
        self.load_usage()
        
    def load_usage(self):
        if os.path.exists(self.usage_file):
            with open(self.usage_file, 'r') as f:
                self.usage = json.load(f)
        else:
            self.usage = {
                'fmp': {'daily_calls': 0, 'last_reset': ''},
                'alpha_vantage': {'daily_calls': 0, 'last_reset': ''},
                'finnhub': {'minute_calls': 0, 'last_reset': ''}
            }
            
    def save_usage(self):
        with open(self.usage_file, 'w') as f:
            json.dump(self.usage, f)
            
    def record_call(self, api_name: str):
        current_time = datetime.now()
        
        if api_name not in self.usage:
            return
            
        # Resetear contadores según corresponda
        if api_name in ['fmp', 'alpha_vantage']:
            last_reset = datetime.strptime(self.usage[api_name]['last_reset'], '%Y-%m-%d') if self.usage[api_name]['last_reset'] else current_time
            if current_time.date() > last_reset.date():
                self.usage[api_name]['daily_calls'] = 0
                self.usage[api_name]['last_reset'] = current_time.strftime('%Y-%m-%d')
        else:  # finnhub
            last_reset = datetime.fromtimestamp(float(self.usage[api_name]['last_reset'])) if self.usage[api_name]['last_reset'] else current_time
            if (current_time - last_reset).total_seconds() > 60:
                self.usage[api_name]['minute_calls'] = 0
                self.usage[api_name]['last_reset'] = str(current_time.timestamp())
                
        # Incrementar contador
        if api_name in ['fmp', 'alpha_vantage']:
            self.usage[api_name]['daily_calls'] += 1
        else:
            self.usage[api_name]['minute_calls'] += 1
            
        self.save_usage()

def main():
    data_fetcher = MultiSourceDataFetcher()
    api_tracker = APIUsageTracker()
    
    print("Iniciando proceso de obtención de datos...")
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    for symbol in symbols:
        print(f"\nProcesando {symbol}...")
        current_date = start_date
        
        while current_date <= end_date:
            print(f"\nObteniendo datos para {current_date.date()}")
            
            data, source = data_fetcher.get_market_data(symbol, current_date)
            
            if data is not None:
                print(f"Datos obtenidos de {source}")
                if source != 'cache':
                    api_tracker.record_call(source)
            else:
                print("No se pudieron obtener datos")
                if not data_fetcher.save_progress_and_ask_continue(current_date, symbol):
                    print("Proceso pausado por el usuario")
                    return
                    
            current_date += timedelta(days=1)
            
    print("\nProceso completado exitosamente")

if __name__ == "__main__":
    main()
