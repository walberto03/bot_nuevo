# trading_bot/src/analysis/feature_ranker.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from pathlib import Path

print("[FeatureRanker] Cargando datos...")

# Cargar datos de la base de datos
df = load_all_data()
df = df[df["Symbol"] == "EURUSD"]
df = df.dropna()

# Añadir indicadores técnicos
pa = PatternAnalyzer()
df = pa.calculate_technical_indicators(df)
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

# Seleccionar características
features = [
    "RSI", "MACD", "Signal", "BB_up", "BB_dn",
    "ATR", "SMA_20", "SMA_50"
]

X = df[features]
y = df["target"]

# Entrenamiento
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Interpretabilidad con SHAP (modo Tree)
print("[FeatureRanker] Calculando importancia con SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Visualizar
Path("trading_bot/plots").mkdir(parents=True, exist_ok=True)
shap.summary_plot(shap_values[1], X_train, show=False)
plt.savefig("trading_bot/plots/feature_importance_shap.png", bbox_inches="tight")
plt.close()
print("[FeatureRanker] Gráfico guardado en plots/feature_importance_shap.png")
