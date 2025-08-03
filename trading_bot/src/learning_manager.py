import pandas as pd
from datetime import datetime, timedelta
import json
import os

class LearningManager:
    def __init__(self):
        self.performance_file = "data/performance_history.json"
        self.load_performance_history()
        
    def load_performance_history(self):
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                self.performance_history = json.load(f)
        else:
            self.performance_history = {
                'daily_results': [],
                'overall_stats': {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0
                }
            }
            
    def save_performance_history(self):
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_history, f)
            
    def record_daily_performance(self, date, predictions, actual_results):
        """Registra el rendimiento diario durante el aprendizaje"""
        daily_success = 0
        daily_total = len(predictions)
        daily_profit = 0
        
        for pred, actual in zip(predictions, actual_results):
            if self.validate_prediction(pred, actual):
                daily_success += 1
                daily_profit += self.calculate_profit(pred, actual)
                
        daily_result = {
            'date': date.strftime('%Y-%m-%d'),
            'total_trades': daily_total,
            'successful_trades': daily_success,
            'win_rate': (daily_success/daily_total*100) if daily_total > 0 else 0,
            'profit': daily_profit
        }
        
        self.performance_history['daily_results'].append(daily_result)
        self.update_overall_stats()
        self.save_performance_history()
        
        return daily_result
        
    def update_overall_stats(self):
        """Actualiza estadísticas generales"""
        total_trades = 0
        successful_trades = 0
        total_profit = 0
        
        for day in self.performance_history['daily_results']:
            total_trades += day['total_trades']
            successful_trades += day['successful_trades']
            total_profit += day['profit']
            
        self.performance_history['overall_stats'] = {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'win_rate': (successful_trades/total_trades*100) if total_trades > 0 else 0,
            'avg_profit': total_profit/total_trades if total_trades > 0 else 0
        }
