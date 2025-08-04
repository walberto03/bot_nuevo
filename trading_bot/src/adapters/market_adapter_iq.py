import pandas as pd
import os
import time
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option

class IQOptionAdapter:
    def __init__(self):
        self.email = os.getenv("IQ_EMAIL")
        self.password = os.getenv("IQ_PASSWORD")

        if not self.email or not self.password:
            raise Exception("[IQOptionAdapter] Faltan credenciales IQ Option en las variables de entorno.")

        self.api = IQ_Option(self.email, self.password)
        self.api.connect()

        if not self.api.check_connect():
            raise Exception("[IQOptionAdapter] Error de conexión con IQ Option")

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            if not self.api.check_connect():
                print("[IQOptionAdapter] Reintentando conexión...")
                self.api.connect()

            asset = symbol.upper()
            candles = []
            current = int(time.mktime(end.timetuple()))

            while current > int(time.mktime(start.timetuple())):
                print(f"[IQOptionAdapter] Descargando {asset} desde {datetime.utcfromtimestamp(current)}")
                data = self.api.get_candles(asset, 86400, 1000, current)
                if not data:
                    break
                candles.extend(data)
                current = data[-1]['from'] - 86400  # ir atrás un día
                time.sleep(1)  # evita bloqueo

            if not candles:
                print(f"[IQOptionAdapter] No se obtuvieron velas para {symbol}")
                return None

            df = pd.DataFrame(candles)
            df['datetime'] = pd.to_datetime(df['from'], unit='s')
            df.set_index('datetime', inplace=True)
            df = df[['open', 'max', 'min', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.sort_index()
            df = df.loc[start:end]

            return df if not df.empty else None

        except Exception as e:
            print(f"[IQOptionAdapter] Error: {e}")
            return None
