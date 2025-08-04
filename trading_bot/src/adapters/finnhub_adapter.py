# trading_bot/src/adapters/finnhub_adapter.py

import pandas as pd
import requests
from datetime import datetime
from trading_bot.config import FINNHUB_API_KEY

class FinnhubAdapter:
    URL = "https://finnhub.io/api/v1/forex/candle"

    def fetch(self, symbol: str, start: datetime, end: datetime):
        fx_from, fx_to = symbol[:3], symbol[3:]
        resolution = "D"  # Diario

        try:
            payload = {
                "symbol": f"OANDA:{fx_from}_{fx_to}",
                "resolution": resolution,
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
                "token": FINNHUB_API_KEY,
            }
            res = requests.get(self.URL, params=payload)
            data = res.json()

            if data.get("s") != "ok":
                print(f"[FinnhubAdapter] No hay datos válidos para {symbol}")
                return None

            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["t"], unit="s"),
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"]
            }).set_index("datetime")

            return df if not df.empty else None

        except Exception as e:
            print(f"[FinnhubAdapter] Error al obtener datos: {e}")
            return None
