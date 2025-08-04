# Archivo: trading_bot/src/utils/relevance_filter.py

class RelevanceFilter:
    def __init__(self):
        # 🔵 Diccionario de palabras clave importantes (se irá ampliando o ajustando en el futuro)
        self.keywords = [
            "inflation", "interest rates", "tariffs", "central bank", "ECB", "Fed",
            "recession", "GDP", "economic slowdown", "employment data", "jobs report",
            "unemployment", "Brexit", "trade war", "banking crisis", "monetary policy",
            "fiscal policy", "credit rating", "currency intervention"
        ]

    def is_relevant(self, title: str, description: str = "") -> bool:
        """
        Retorna True si al menos una palabra clave aparece en el título o la descripción.
        """
        text = (title + " " + description).lower()
        return any(keyword.lower() in text for keyword in self.keywords)

    def filter_articles(self, articles: list) -> list:
        """
        Filtra artículos dejando solo los relevantes.
        """
        filtered = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            if self.is_relevant(title, description):
                filtered.append(article)
        return filtered
