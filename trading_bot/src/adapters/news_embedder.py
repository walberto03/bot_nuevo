# Archivo: trading_bot/src/adapters/news_embedder.py

from sentence_transformers import SentenceTransformer
import torch

class NewsEmbedder:
    def __init__(self, device=None):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed_texts(self, texts: list) -> torch.Tensor:
        """
        Convierte una lista de textos en embeddings (vectores numéricos).
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings
