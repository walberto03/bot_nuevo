# Archivo: trading_bot/src/adapters/news_adapter_ii.py

from ii_researcher.core import search

class NewsAdapterIIResearcher:
    def __init__(self):
        pass

    def fetch_news(self, symbol: str, start: str, end: str, max_results: int = 5) -> list:
        """
        Busca noticias usando ii-researcher (scraping de internet)
        """
        query = f"{symbol} news between {start} and {end}"
        try:
            results = search(query, num_results=max_results)
            articles = []
            for res in results:
                article = {
                    "title": res.get("title", ""),
                    "description": res.get("snippet", ""),
                    "url": res.get("link", "")
                }
                articles.append(article)
            return articles
        except Exception as e:
            print(f"[NewsAdapterIIResearcher] ⚠️ Error al buscar noticias: {e}")
            return []
