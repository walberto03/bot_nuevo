# Archivo: trading_bot/src/strategy_validator.py

import pandas as pd
from typing import List, Dict

class StrategyValidator:
    def __init__(self):
        pass

    def validate_signals(self, daily_signals: List[Dict], hourly_signals: List[Dict]) -> List[Dict]:
        """
        Filtra las decisiones si hay coherencia entre señales horarias y diarias.
        Solo se toman decisiones si ambas están alineadas.
        """
        validated = []
        hourly_index = {(s['date'], s['symbol']): s for s in hourly_signals}

        for d in daily_signals:
            key = (d['date'], d['symbol'])
            h = hourly_index.get(key)

            if h and d['action'] == h['action']:
                validated.append(d)
            else:
                print(f"[StrategyValidator] Señal descartada por conflicto: {d['symbol']} en {d['date']}")

        return validated

    def coherence_score(self, daily_signals: List[Dict], hourly_signals: List[Dict]) -> float:
        """
        Mide la coherencia general entre señales diarias y horarias.
        """
        if not daily_signals or not hourly_signals:
            return 0.0

        match_count = 0
        total_count = 0
        hourly_index = {(s['date'], s['symbol']): s for s in hourly_signals}

        for d in daily_signals:
            key = (d['date'], d['symbol'])
            h = hourly_index.get(key)
            if h:
                total_count += 1
                if d['action'] == h['action']:
                    match_count += 1

        return match_count / total_count if total_count else 0.0
