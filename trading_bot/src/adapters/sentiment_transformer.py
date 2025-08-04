# trading_bot/src/adapters/sentiment_adapter_transformer.py

from transformers import pipeline
from pathlib import Path
import json
from datetime import datetime
from trading_bot.config import TradingConfig
import tweepy

class SentimentAdapterTransformer:
    def __init__(self):
        cfg = TradingConfig()
        self.client = tweepy.Client(bearer_token=cfg.twitter_bearer)
        self.cache_dir = Path(cfg.cache_dir) / "twitter_transformer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def fetch_batch_twitter_sentiment(self, symbols: list, max_tweets: int = 100) -> dict:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"tweets_{date_str}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                tweets = json.load(f)
        else:
            query = " OR ".join(symbols) + " -is:retweet lang:en"
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),
                tweet_fields=["text"]
            )
            tweets = [t.text for t in (response.data or [])]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(tweets, f, ensure_ascii=False)

        sentiments = self.model(tweets)

        scores = {s: [] for s in symbols}
        for tweet, result in zip(tweets, sentiments):
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            for sym in symbols:
                if f"${sym.lower()}" in tweet.lower() or sym.lower() in tweet.lower():
                    scores[sym].append(score)

        return {s: (sum(v) / len(v)) if v else 0.0 for s, v in scores.items()}
