# trading_bot/src/trading_brain.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_bot.config import TradingConfig
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.adapters.news_adapter import NewsAdapter
from trading_bot.src.adapters.sentiment_adapter import SentimentAdapter
from trading_bot.src.adapters.market_adapter import MarketAdapter
from trading_bot.src.decision_maker import DecisionMaker
from trading_bot.src.risk_manager import RiskManager
from trading_bot.src.data.db_manager import save_prices
import joblib

class TradingBrain:
    def __init__(self):
        self.cfg        = TradingConfig()
        self.market     = MarketAdapter()
        self.analyzer   = PatternAnalyzer()
        self.news       = NewsAdapter()
        self.twitter    = SentimentAdapter()
        self.decider    = DecisionMaker()
        self.risk       = RiskManager(self.cfg)
        self.model_path = self.cfg.model_save_path

    def analyze_and_learn(self, symbol: str, start_date: datetime, end_date: datetime):
        """
        Backtest de precios + noticias para entrenar y generar señales.
        """
        current = start_date
        X, y, signals = [], [], []

        while current < end_date:
            df = self.market.fetch(symbol, current, current + timedelta(days=1))
            if df is None or df.empty:
                current += timedelta(days=1)
                continue

            # Volatilidad diaria para riesgo
            vol = (df['High'] - df['Low']).mean()

            # Sentimiento Twitter + Noticias
            tw_score = self.twitter.fetch_batch_twitter_sentiment([symbol]).get(symbol, 0.0)
            articles = self.news.fetch_news(symbol,
                                            current.strftime('%Y-%m-%d'),
                                            current.strftime('%Y-%m-%d'))
            nw_score = self._score_news(articles)
            sentiment = (tw_score + nw_score) / 2

            # Detección de patrones técnicos
            patterns = self.analyzer.find_patterns(df, sentiment)

            # —> DECISION MAKER + RIESGO + NIVELES DE TRADING
            order = self.decider.make_decision(patterns, sentiment, [])
            if order and order.get('side') in ('buy', 'sell'):
                # Calcular stop_pct y tamaño de posición
                stop_pct = self.risk.compute_stop(vol)
                size     = self.risk.adjust_size(self.cfg.initial_balance, stop_pct)
                order['size'] = size

                # Precios de entrada, SL y TP
                entry_price = df['Close'].iloc[-1]
                order['entry_price'] = entry_price
                if order['side'] == 'buy':
                    order['stop_loss']  = entry_price * (1 - stop_pct)
                    tp_pct = stop_pct * self.cfg.take_profit_ratio
                    order['take_profit'] = entry_price * (1 + tp_pct)
                else:  # sell
                    order['stop_loss']  = entry_price * (1 + stop_pct)
                    tp_pct = stop_pct * self.cfg.take_profit_ratio
                    order['take_profit'] = entry_price * (1 - tp_pct)

                order['symbol'] = symbol
                order['date']   = current.strftime('%Y-%m-%d')
                signals.append(order)
                print(f"[Signal] {order}")

            # Preparar datos de entrenamiento
            feature_row = self._prepare_features(df, sentiment, patterns)
            try:
                next_day = self.market.fetch(symbol,
                                             current + timedelta(days=1),
                                             current + timedelta(days=2))
                if next_day is None or next_day.empty:
                    current += timedelta(days=1)
                    continue

                current_price = df['Close'].iloc[-1]
                next_price    = next_day['Close'].iloc[-1]
                label = 1 if next_price > current_price else 0

                X.append(feature_row)
                y.append(label)

                # Guardar precios en base de datos
                save_prices(symbol, df)

            except Exception as e:
                print(f"[TradingBrain] Error procesando {current.date()}: {e}")

            current += timedelta(days=1)

        # Entrenamiento si hay datos
        if X and y:
            self._train_model(X, y)

        # Opcional: devolver señales generadas en backtest
        return signals

    def _prepare_features(self, df: pd.DataFrame, sentiment: float, patterns: list) -> list:
        sma_20     = df['Close'].rolling(20).mean().iloc[-1]
        sma_50     = df['Close'].rolling(50).mean().iloc[-1]
        rsi        = self._rsi(df['Close'])
        volatility = (df['High'] - df['Low']).mean()
        tech_score = self.decider._tech(patterns)
        return [sma_20, sma_50, rsi, volatility, sentiment, tech_score]

    def _rsi(self, series: pd.Series, window: int = 14):
        delta = series.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss  = -delta.where(delta < 0, 0.0).rolling(window=window).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def _score_news(self, articles: list) -> float:
        pos = {"gain", "up", "bull", "rise", "strong", "surge", "rally"}
        neg = {"loss", "down", "bear", "fall", "weak", "slump", "crash"}
        scores = []
        for art in articles:
            text  = (art.get("title", "") + " " + art.get("description", "")).lower().split()
            p     = sum(w in pos for w in text)
            n     = sum(w in neg for w in text)
            scores.append(0.0 if (p + n) == 0 else (p - n) / (p + n))
        return sum(scores) / len(scores) if scores else 0.0

    def _train_model(self, X, y):
        from sklearn.ensemble    import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics       import classification_report

        print("[ModelTrainer] Entrenando modelo...")
        X_df = pd.DataFrame(X)
        y_arr = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_arr, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("[ModelTrainer] Reporte de Clasificación:")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, self.model_path)
        print(f"[ModelTrainer] Modelo guardado en {self.model_path}")
