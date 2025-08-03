import pandas as pd
import numpy as np

class Backtester:
    """
    Clase para ejecutar backtests sobre señales de trading.
    """
    def run_backtest(self, signals: pd.DataFrame) -> dict:
        """
        Ejecuta un backtest simple calculando métricas clave:
        - Win rate
        - Profit factor
        - Sharpe ratio anualizado
        - Drawdown máximo
        """
        # Win rate
        total = len(signals)
        wins = signals['hit'].sum() if 'hit' in signals else 0
        win_rate = wins / total if total else 0.0

        # Profit factor
        profit = signals['profit'].sum() if 'profit' in signals else None
        loss = signals['loss'].sum() if 'loss' in signals else None
        if profit is not None and loss not in (None, 0):
            profit_factor = profit / abs(loss)
        else:
            profit_factor = None

        # Sharpe ratio anualizado
        if 'returns' in signals and not signals['returns'].empty:
            returns = signals['returns']
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = None

        # Max drawdown
        drawdown = None
        if 'returns' in signals and not signals['returns'].empty:
            equity_curve = (1 + signals['returns']).cumprod()
            peak = equity_curve.cummax()
            drawdown = ((equity_curve - peak) / peak).min()

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'max_drawdown': drawdown
        }
