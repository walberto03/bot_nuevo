# trading_bot/src/adapters/news_adapter.py

import requests
from transformers import pipeline
import torch
from trading_bot.config import cfg

class NewsAdapter:
    def __init__(self):
        self.key = cfg["api"]["newsapi_key"]
        self.base = "https://newsapi.org/v2/everything"
        self.pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

    def fetch_news(self, symbol: str, start: str, end: str, max_pages: int = 5) -> list:
        """
        Descarga y analiza noticias para un símbolo entre start y end (formato YYYY-MM-DD).
        Retorna artículos con sentimiento.
        """
        all_articles = []

        for page in range(1, max_pages + 1):
            params = {
                "q": symbol,
                "from": start,
                "to": end,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 100,
                "page": page,
                "apiKey": self.key
            }
            resp = requests.get(self.base, params=params).json()
            arts = resp.get("articles", [])
            if not arts:
                break
            all_articles.extend(arts)

        return all_articles

    def compute_sentiment_score(self, articles: list) -> float:
        """
        Aplica transformers a los titulares/descripciones para calcular una media de sentimiento.
        """
        if not articles:
            return 0.0

        texts = []
        for art in articles:
            text = f"{art['title']} {art.get('description', '')}".strip()
            if text:
                texts.append(text)

        results = self.pipe(texts, truncation=True, max_length=512)

        scores = []
        for r in results:
            val = r["score"] if r["label"] == "POSITIVE" else -r["score"]
            scores.append(val)

        return sum(scores) / len(scores) if scores else 0.0
