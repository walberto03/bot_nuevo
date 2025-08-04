import time
from datetime import datetime
from typing import Optional
import pandas as pd

from trading_bot.src.adapters.twelve_adapter import TwelveDataAdapter
from trading_bot.src.adapters.market_adapter_iq import IQOptionAdapter
from trading_bot.src.adapters.finnhub_adapter import FinnhubAdapter

class MarketAdapter:
    """
    Fuente principal: Twelve Data
    Fallback: IQ Option → Finnhub
    """

    def __init__(self):
        self.last_source_used = "ninguna"
        self.twelve_adapter = TwelveDataAdapter()
        self.iq_adapter = IQOptionAdapter()
        self.fh_adapter = FinnhubAdapter()

    def fetch(self, symbol: str, start: datetime | str, end: datetime | str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        # Intentar Twelve Data como fuente principal
        try:
            df = self.twelve_adapter.fetch(symbol, start, end, timeframe)
            if df is not None and not df.empty:
                self.last_source_used = "TwelveData"
                return df
            print(f"[MarketAdapter] ⚠️ Twelve Data no devolvió datos para {symbol}")
        except Exception as e:
            print(f"[MarketAdapter] ❌ Error Twelve Data en {symbol}: {e}")

        # Fallbacks
        return self._fallback_iq_then_fh(symbol, start, end, timeframe)

    def _fallback_iq_then_fh(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        print(f"[MarketAdapter] 🕵️‍♂️ Intentando IQ Option como fallback para {symbol}")
        iq_df = self.iq_adapter.fetch(symbol, start, end)
        if iq_df is not None and not iq_df.empty:
            self.last_source_used = "IQ Option"
            return iq_df

        print(f"[MarketAdapter] 🔄 IQ Option falló. Intentando Finnhub para {symbol}")
        fh_df = self.fh_adapter.fetch(symbol, start, end)
        if fh_df is not None and not fh_df.empty:
            self.last_source_used = "Finnhub"
            return fh_df

        print(f"[MarketAdapter] 🚫 Todas las fuentes fallaron para {symbol}")
        self.last_source_used = "ninguna"
        return None
