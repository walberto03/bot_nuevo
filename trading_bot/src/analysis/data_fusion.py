# trading_bot/src/analysis/data_fusion.py

import pandas as pd
import matplotlib.pyplot as plt
from trading_bot.src.data.db_manager import load_all_data

def run_data_fusion(symbol="EURUSD"):
    print("[DataFusion] Comparando fuentes: Alpha Vantage vs IQ Option...")

    df = load_all_data()

    # Filtrar por símbolo
    df = df[df["Symbol"] == symbol.upper()]

    # Dividir por fuente
    alpha = df[df["Source"] == "alpha"].copy()
    iq = df[df["Source"] == "iq"].copy()

    if alpha.empty or iq.empty:
        print("[DataFusion] No hay suficientes datos para ambas fuentes.")
        return

    # Asegurar índices por fecha
    alpha.set_index("Date", inplace=True)
    iq.set_index("Date", inplace=True)

    # Unir por fecha
    merged = pd.merge(
        alpha[["Close"]],
        iq[["Close"]],
        left_index=True,
        right_index=True,
        suffixes=("_alpha", "_iq")
    )

    # Calcular diferencias absolutas y relativas
    merged["diff"] = (merged["Close_alpha"] - merged["Close_iq"]).abs()
    merged["rel_diff_pct"] = 100 * merged["diff"] / merged[["Close_alpha", "Close_iq"]].mean(axis=1)

    # Mostrar algunas métricas clave
    print(f"\nDiferencia promedio absoluta: {merged['diff'].mean():.5f}")
    print(f"Diferencia relativa promedio (%): {merged['rel_diff_pct'].mean():.3f}%")

    # Visualizar diferencias
    plt.figure(figsize=(12, 6))
    plt.plot(merged.index, merged["Close_alpha"], label="Alpha Vantage", linewidth=2)
    plt.plot(merged.index, merged["Close_iq"], label="IQ Option", linewidth=2)
    plt.title(f"Comparación de Precios: {symbol}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("trading_bot/logs/data_fusion_plot.png")
    plt.close()
    print("[DataFusion] Gráfico guardado en: trading_bot/logs/data_fusion_plot.png")
