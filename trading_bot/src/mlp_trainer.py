import sys, os
# Añadir raíz del proyecto al PYTHONPATH para permitir imports absolutos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE
from trading_bot.config import TradingConfig
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.pattern_analyzer import PatternAnalyzer

class ModelTrainer:
    def __init__(self):
        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()

    def prepare_dataset(self):
        df = load_all_data()
        df = df[df['Close'].notna()].copy()
        df = df.sort_values(['symbol', 'date'])

        all_rows = []
        for _, group in df.groupby('symbol'):
            group = self.pattern.calculate_technical_indicators(group)
            group['target'] = (group['Close'].shift(-1) > group['Close']).astype(int)
            group.dropna(inplace=True)
            all_rows.append(group)

        df_all = pd.concat(all_rows)

        features = [
            'RSI', 'MACD', 'Signal',
            'BB_up', 'BB_dn', 'ATR',
            f'SMA_{self.cfg.sma_short}', f'SMA_{self.cfg.sma_long}'
        ]

        X = df_all[features]
        y = df_all['target']

        print(f"[ModelTrainer] Distribución de clases original: {Counter(y)}")
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        print("[ModelTrainer] Preparando conjunto de datos...")
        X_train, X_test, y_train, y_test = self.prepare_dataset()

        print(f"[ModelTrainer] Distribución de clases en entrenamiento: {Counter(y_train)}")
        # Balancear clases con SMOTE en training set
        if len(set(y_train)) > 1:
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"[ModelTrainer] Distribución tras SMOTE: {Counter(y_train)}")

        print("[ModelTrainer] Entrenando RandomForestClassifier con class_weight='balanced'...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        rf_score = cross_val_score(rf, X_train, y_train, cv=5).mean()

        print("[ModelTrainer] Entrenando MLPClassifier...")
        mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        mlp_score = cross_val_score(mlp, X_train, y_train, cv=5).mean()

        print(f"[ModelTrainer] Score RF : {rf_score:.4f}")
        print(f"[ModelTrainer] Score MLP: {mlp_score:.4f}")

        best_model = rf if rf_score > mlp_score else mlp
        pred = best_model.predict(X_test)

        print("[ModelTrainer] Reporte de Clasificación:")
        print(classification_report(y_test, pred, zero_division=0))

        model_name = "trading_model.pkl" if best_model == rf else "mlp_model.pkl"
        path = self.cfg.models_dir / model_name
        joblib.dump(best_model, path)
        print(f"[ModelTrainer] ✅ Modelo guardado en {path}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
