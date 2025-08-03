# trading_bot/src/adapters/news_adapter_transformer.py

import requests
import time
import random
import torch
from transformers import pipeline
from trading_bot.config import cfg
import subprocess
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

class NewsAdapterTransformer:
    """
    Adaptador de noticias que combina NewsAPI con un micro-servicio II-Researcher,
    construye consultas dinámicas según temas relevantes y analiza sentimiento con FinBERT.
    """
    def __init__(self):
        # NewsAPI
        self.newsapi_key = cfg["api"]["newsapi_key"]
        if not self.newsapi_key:
            raise ValueError("[NewsAdapterTransformer] ❌ NEWSAPI_KEY no está definido en config.yaml")
        self.newsapi_url = "https://newsapi.org/v2/everything"
        # FinBERT
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
        # II-Researcher HTTP endpoint (arranca: uvicorn ii_researcher.api:app …)
        self.ii_host = "127.0.0.1"
        self.ii_port = 8000
        self.ii_url = f"http://{self.ii_host}:{self.ii_port}/search"

        # Temas importantes por símbolo
        self.related_topics = {
            "EURUSD": [
                "Eurozone economy", "US economy", "ECB decision", "Fed decision",
                "Trade war Europe US", "US tariffs Europe", "Interest rates Euro", "US inflation"
            ],
            "GBPUSD": [
                "UK economy", "US economy", "Bank of England decision", "Fed decision",
                "Brexit", "Trade agreements UK US", "UK inflation", "US inflation"
            ],
            # añade más símbolos si lo necesitas…
        }

    def _build_dynamic_query(self, symbol: str) -> str:
        topics = self.related_topics.get(symbol.upper(), [])
        if not topics:
            return f"{symbol} forex"
        picks = random.sample(topics, min(2, len(topics)))
        return f"{symbol} {' '.join(picks)}"

    def _ii_running(self):
        try:
            requests.get(self.ii_url, timeout=5)
            return True
        except requests.exceptions.ConnectionError:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def fetch_news(self, symbol: str, start: str, end: str, max_pages: int = 1) -> list:
        """
        1) Intenta descarga vía NewsAPI.
        2) Si quedan <3 artículos, hace fallback a II-Researcher HTTP.
        Devuelve lista de dicts con keys 'title' y 'description'.
        """
        articles: list[dict] = []

        # — 1) NewsAPI oficial —
        try:
            for page in range(1, max_pages + 1):
                params = {
                    "q": symbol,
                    "from": start,
                    "to": end,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 100,
                    "page": page,
                    "apiKey": self.newsapi_key
                }
                resp = requests.get(self.newsapi_url, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json().get("articles", [])
                if not batch:
                    break
                articles.extend(batch)
                time.sleep(1.0)  # Respetamos rate-limit
        except Exception as e:
            print(f"[NewsAdapterTransformer] ⚠️ NewsAPI error: {e}")

        # — 2) Fallback con II-Researcher si pocos resultados —
        if len(articles) < 3:
            print("[NewsAdapterTransformer] 🔄 Pocos resultados, usando II-Researcher fallback…")
            query = self._build_dynamic_query(symbol)
            # Intentar llamada a II-Researcher
            try:
                resp = requests.get(
                    self.ii_url,
                    params={"query": query, "max_results": 10},
                    timeout=60
                )
                resp.raise_for_status()
                extra = resp.json()
                for item in extra:
                    articles.append({
                        "title":       item.get("title", ""),
                        "description": item.get("snippet", "") or item.get("description", "")
                    })
                print(f"[NewsAdapterTransformer] 📰 Total artículos tras II-Researcher: {len(articles)}")

            except requests.exceptions.ConnectionError as conn_e:
                # Si la conexión fue rechazada, probablemente no está corriendo II-Researcher
                print("[NewsAdapterTransformer] ⚠️ II-Researcher no está corriendo, iniciando servicio local…")
                try:
                    # Ruta al directorio de ii_researcher
                    ii_path = Path(__file__).resolve().parent / "ii_researcher"
                    cmd = [
                        "uvicorn",
                        "api:app",
                        "--host", self.ii_host,
                        "--port", str(self.ii_port)
                    ]
                    # Arrancamos uvicorn en background (stdout y stderr a DEVNULL)
                    subprocess.Popen(
                        cmd,
                        cwd=ii_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    # Esperamos unos segundos a que el servicio suba
                    time.sleep(10)

                    # Reintentamos llamar a II-Researcher
                    resp = requests.get(
                        self.ii_url,
                        params={"query": query, "max_results": 10},
                        timeout=60
                    )
                    resp.raise_for_status()
                    extra = resp.json()
                    for item in extra:
                        articles.append({
                            "title":       item.get("title", ""),
                            "description": item.get("snippet", "") or item.get("description", "")
                        })
                    print(f"[NewsAdapterTransformer] 📰 Artículos tras iniciar II-Researcher: {len(articles)}")
                except Exception as e2:
                    print(f"[NewsAdapterTransformer] ⚠️ II-Researcher sigue inaccesible: {e2}")

            except Exception as e:
                print(f"[NewsAdapterTransformer] ⚠️ II-Researcher error: {e}")

        # Filtrar artículos vacíos o duplicados
        seen = set()
        unique_articles = []
        for art in articles:
            title = art.get("title", "").strip()
            desc = art.get("description", "").strip()
            if title or desc:
                key = (title, desc)
                if key not in seen:
                    seen.add(key)
                    unique_articles.append(art)
        return unique_articles

    def analyze_news_sentiment(self, articles: list) -> float:
        """
        Devuelve score promedio de sentimiento entre –1 y +1.
        """
        if not articles:
            return 0.0

        texts = [
            a.get("title", "") + " " + a.get("description", "")
            for a in articles
            if a.get("title")
        ]
        if not texts:
            return 0.0

        results = self.analyzer(texts, truncation=True, max_length=512)
        scores = []
        for r in results:
            lbl = r["label"].upper()
            val = r["score"]
            scores.append(val if lbl == "POSITIVE" else -val)

        return sum(scores) / len(scores)