from langchain.embeddings.base import Embeddings
from typing import List
import numpy as np

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        print(f"🔍 Using Hugging Face embedder from: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

def get_embedder(local_model_path=None):
    if local_model_path:
        return SentenceTransformerEmbeddings(local_model_path)
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        print("🧠 Using Ollama embedder: nomic-embed-text")
        return OllamaEmbeddings(model="nomic-embed-text")