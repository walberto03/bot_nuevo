# ii_researcher/api.py
from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.get("/search")
def search(query: str, max_results: int = 10):
    # Aquí podrías devolver una lista simulada de resultados
    return [
        {"title": f"Simulación: {query}", "snippet": "Este es un artículo de prueba."}
        for _ in range(max_results)
    ]
