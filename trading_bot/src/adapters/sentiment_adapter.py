# trading_bot/src/adapters/sentiment_adapter.py

import os
import json
from datetime import datetime
from pathlib import Path
import tweepy
import torch
from transformers import pipeline
from trading_bot.config import TradingConfig

class SentimentAdapter:
    def __init__(self):
        cfg = TradingConfig()
        self.client = tweepy.Client(bearer_token=cfg.twitter_bearer)
        self.cache_dir = Path(cfg.cache_dir) / "twitter"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

    def fetch_batch_twitter_sentiment(self, symbols: list, max_tweets: int = 100) -> dict:
        # 1) Cargar desde caché si existe
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"twitter_{date_str}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                tweets = json.load(f)
        else:
            # 2) Hacer búsqueda
            query = " OR ".join(symbols) + " -is:retweet lang:en"
            resp = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),
                tweet_fields=["text"]
            )
            tweets = [{"text": t.text} for t in (resp.data or [])]

            # 3) Guardar en caché
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(tweets, f, ensure_ascii=False)

        # 4) Calcular sentimientos con Transformers
        texts = [t["text"] for t in tweets]
        if not texts:
            return {s: 0.0 for s in symbols}

        results = self.pipe(texts, truncation=True, max_length=512)

        scores = {s: [] for s in symbols}
        for i, res in enumerate(results):
            sentiment = res["label"]
            score = res["score"]
            compound = score if sentiment == "POSITIVE" else -score
            text_lower = texts[i].lower()
            for sym in symbols:
                if f"${sym.lower()}" in text_lower or sym.lower() in text_lower:
                    scores[sym].append(compound)

        # 5) Promediar
        return {s: (sum(v) / len(v)) if v else 0.0 for s, v in scores.items()}
