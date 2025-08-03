# trading_bot/src/adapters/finnhub_adapter.py

import pandas as pd
import requests
import time
from datetime import datetime
from trading_bot.config import FINNHUB_API_KEY

class FinnhubAdapter:
    BASE_URL = "https://finnhub.io/api/v1/forex/candle"

    def __init__(self):
        if not FINNHUB_API_KEY:
            raise ValueError("[FinnhubAdapter] ⚠️ Falta la FINNHUB_API_KEY en el entorno o config.yaml")

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        try:
            fx_from, fx_to = symbol[:3], symbol[3:]
            resolution = "D"  # daily candles
            from_unix = int(start.timestamp())
            to_unix = int(end.timestamp())

            params = {
                "symbol": f"OANDA:{fx_from}_{fx_to}",
                "resolution": resolution,
                "from": from_unix,
                "to": to_unix,
                "token": FINNHUB_API_KEY
            }

            resp = requests.get(self.BASE_URL, params=params)
            data = resp.json()

            if data.get("s") != "ok":
                print(f"[FinnhubAdapter] ⚠️ No se pudieron obtener velas para {symbol}")
                return None

            df = pd.DataFrame({
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"],
                "datetime": pd.to_datetime(data["t"], unit="s")
            })

            df.set_index("datetime", inplace=True)
            df = df.sort_index()
            df = df.loc[start:end]
            print(f"[FinnhubAdapter] ✅ Datos cargados para {symbol} ({len(df)} filas)")
            time.sleep(1)
            return df if not df.empty else None

        except Exception as e:
            print(f"[FinnhubAdapter] ❌ Error al obtener datos de Finnhub para {symbol}: {e}")
            return None
