import pandas as pd
import numpy as np
from trading_bot.config import cfg

class PatternAnalyzer:
    def __init__(self):
        self.use_deep_learning = cfg.get("use_deep_learning_patterns", False)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = cfg["parameters"]
        df = df.copy()

        # Estandarizar columnas a minúsculas para consistencia
        df.columns = [col.lower() for col in df.columns]

        # Asegurar columnas necesarias existen y son numéricas
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"[PatternAnalyzer] Columna faltante: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop NaN tempranos
        df = df.dropna(subset=required_cols)

        # SMA (ya estaba)
        df[f"sma_{p['sma_short']}"] = df["close"].rolling(p["sma_short"]).mean()
        df[f"sma_{p['sma_long']}"] = df["close"].rolling(p["sma_long"]).mean()

        # RSI (ya estaba)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(p["rsi_window"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p["rsi_window"]).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD (ya estaba)
        exp1 = df["close"].ewm(span=p["macd_fast"], adjust=False).mean()
        exp2 = df["close"].ewm(span=p["macd_slow"], adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["signal"] = df["macd"].ewm(span=p["macd_signal"], adjust=False).mean()

        # Bollinger Bands (ya estaba)
        mid = df["close"].rolling(p["bollinger_window"]).mean()
        std = df["close"].rolling(p["bollinger_window"]).std()
        df["bb_up"] = mid + 2 * std
        df["bb_dn"] = mid - 2 * std

        # ATR (ya estaba)
        tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())))
        df["atr"] = tr.rolling(p["atr_window"]).mean()

        # Nuevo: EMA (Exponential Moving Average)
        df["ema_short"] = df["close"].ewm(span=p["sma_short"], adjust=False).mean()  # EMA basada en sma_short window
        df["ema_long"] = df["close"].ewm(span=p["sma_long"], adjust=False).mean()  # EMA basada en sma_long window

        # Nuevo: VWAP (Volume Weighted Average Price) - Calculado diario (reset por día si needed)
        df["cum_volume"] = df["volume"].cumsum()
        df["cum_price_volume"] = (df["close"] * df["volume"]).cumsum()
        df["vwap"] = df["cum_price_volume"] / df["cum_volume"].replace(0, np.nan)  # Evitar división por 0

        # Nuevo: Niveles de Fibonacci (basado en high/low recientes, ej. último swing)
        # Calcula niveles de retracement simples (0.382, 0.618) basado en max high/min low en window
        fib_window = 20  # Ajusta si needed
        df["fib_high"] = df["high"].rolling(fib_window).max()
        df["fib_low"] = df["low"].rolling(fib_window).min()
        df["fib_range"] = df["fib_high"] - df["fib_low"]
        df["fib_382"] = df["fib_high"] - 0.382 * df["fib_range"]  # Nivel de soporte/resistencia
        df["fib_618"] = df["fib_high"] - 0.618 * df["fib_range"]

        return df.dropna()  # Drop NaN finales para datos limpios

    def find_patterns(self, df: pd.DataFrame, sentiment: float = 0.0) -> list:
        if self.use_deep_learning:
            return self._find_patterns_with_model(df)

        df = self.calculate_technical_indicators(df)
        pats = []

        sma_s = f"sma_{cfg['parameters']['sma_short']}"
        sma_l = f"sma_{cfg['parameters']['sma_long']}"
        if len(df) >= 2 and df[sma_s].iloc[-1] > df[sma_l].iloc[-1] and df[sma_s].iloc[-2] <= df[sma_l].iloc[-2]:
            pats.append({"type": "GOLDEN_CROSS", "confidence": 0.7})

        r = df["rsi"].iloc[-1]
        if r < 30:
            pats.append({"type": "OVERSOLD", "confidence": 0.65})
        elif r > 70:
            pats.append({"type": "OVERBOUGHT", "confidence": 0.65})

        if df["macd"].iloc[-1] > df["signal"].iloc[-1] and df["macd"].iloc[-2] <= df["signal"].iloc[-2]:
            pats.append({"type": "MACD_BULLISH", "confidence": 0.6})

        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        if body < 0.1 * (last["high"] - last["low"]):
            pats.append({"type": "DOJI", "confidence": 0.6})

        for p in pats:
            p["confidence"] = min(1.0, max(0.0, p["confidence"] * (1 + sentiment)))  # Clamp entre 0-1

        return pats

    def _find_patterns_with_model(self, df: pd.DataFrame) -> list:
        """
        Método futuro para usar redes neuronales o modelos ML.
        Por ahora devuelve un patrón ficticio si la condición se cumple.
        """
        df = self.calculate_technical_indicators(df)
        if df["rsi"].iloc[-1] > 50 and df["macd"].iloc[-1] > 0:
            return [{"type": "NN_BULLISH", "confidence": 0.75}]
        return []