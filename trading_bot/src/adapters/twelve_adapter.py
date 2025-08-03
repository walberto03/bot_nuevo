# trading_bot/src/adapters/twelve_adapter.py

import requests
import pandas as pd
from datetime import datetime
from trading_bot.config import TWELVE_DATA_API_KEY
from tenacity import retry, stop_after_attempt, wait_fixed
import time  # Agregado: import time para time.sleep

class TwelveDataAdapter:
    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self):
        if not TWELVE_DATA_API_KEY:
            raise ValueError("[TwelveDataAdapter] ❌ TWELVE_DATA_API_KEY no está definido en config.yaml")
        self.daily_credits_used = 0  # Reset manual o por fecha

    def _normalize_interval(self, interval: str) -> str:
        """
        Convierte '1d' en '1day' para que la API de TwelveData lo acepte.
        Dejamos pasar el resto sin cambios.
        """
        if interval.lower() == "1d":
            return "1day"
        return interval

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def fetch(self, symbol: str, start: datetime, end: datetime, interval: str = "1h"):
        if self.daily_credits_used >= 800:
            print("[TwelveData] ⚠️ Límite diario free alcanzado (800 credits). Usando datos existentes.")
            return None

        interval_api = self._normalize_interval(interval)

        params = {
            "symbol": f"{symbol[:3]}/{symbol[3:]}",
            "interval": interval_api,
            "apikey":  TWELVE_DATA_API_KEY,
            "format":  "JSON",
            "outputsize": 5000,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   end.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "values" not in data:
                print(f"[TwelveDataAdapter] ❌ Sin datos para {symbol} ({interval_api}): "
                      f"{data.get('message', 'Respuesta inválida')}")
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df = df.sort_index()

            df = df.rename(columns={
                "open":   "Open",
                "high":   "High",
                "low":    "Low",
                "close":  "Close",
                "volume": "Volume"
            })

            # 🔵 Asegurar que todas las columnas existan
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                if col not in df.columns:
                    if col == "Volume":
                        df["Volume"] = 0.0  # Si falta volumen, llenamos con ceros
                    else:
                        raise ValueError(f"[TwelveDataAdapter] ❌ Columna faltante inesperada: {col}")

            df = df[expected_cols].astype(float)
            self.daily_credits_used += 1
            time.sleep(8)  # ~7-8 calls/min max para límite free
            return df

        except requests.exceptions.RequestException as e:
            print(f"[TwelveDataAdapter] ⚠️ Error al obtener datos de {symbol} ({interval_api}): {e}")
            return None