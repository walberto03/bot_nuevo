# trading_bot/src/utils/yaml_logger.py

import yaml
import os
from datetime import datetime

def guardar_resultado_yaml(data: dict, nombre_archivo: str = "news_only_metrics.yaml"):
    """
    Guarda un diccionario de resultados en formato YAML con timestamp.
    Si el archivo ya existe, añade una nueva entrada y actualiza el resumen top 3 por f1_score.
    """
    ruta = os.path.join("trading_bot", "results", nombre_archivo)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entrada = {"timestamp": timestamp, **data}

    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = yaml.safe_load(f) or {}
    else:
        contenido = {}

    # Histórico de resultados
    historial = contenido.get("resultados", [])
    historial.append(entrada)
    contenido["resultados"] = historial

    # Top 3 por f1_score (manejar si no existe f1_score)
    top3 = sorted(
        [r for r in historial if "f1_score" in r],
        key=lambda r: r.get("f1_score", 0),
        reverse=True
    )[:3]
    contenido["summary"] = {
        "top_3_by_f1_score": [
            {
                "timestamp": r["timestamp"],
                "accuracy": r.get("accuracy"),
                "f1_score": r.get("f1_score"),
                "params": r.get("params")
            }
            for r in top3
        ]
    }

    with open(ruta, "w", encoding="utf-8") as f:
        yaml.safe_dump(contenido, f, allow_unicode=True)

    print(f"[YAML Logger] Resultado guardado en {ruta}")