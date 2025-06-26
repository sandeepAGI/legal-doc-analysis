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

def get_embedder(local_model_path=None, model_name=None):
    """
    Get embedder based on model path or model name.
    
    Args:
        local_model_path: Path to local model files (for BGE models)
        model_name: Name of the model to use (for Hugging Face models or Ollama)
    """
    if local_model_path:
        return SentenceTransformerEmbeddings(local_model_path)
    elif model_name == "nomic-embed-text":
        from langchain_community.embeddings import OllamaEmbeddings
        print("🧠 Using Ollama embedder: nomic-embed-text")
        return OllamaEmbeddings(model="nomic-embed-text")
    elif model_name == "arctic-embed-33m":
        local_path = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/arctic-embed-33m"
        print("❄️ Using Arctic Embed 33m from local path")
        return SentenceTransformerEmbeddings(local_path)
    elif model_name == "all-minilm-l6-v2":
        local_path = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/all-minilm-l6-v2"
        print("⚡ Using all-MiniLM-L6-v2 from local path")
        return SentenceTransformerEmbeddings(local_path)
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        print("🧠 Using Ollama embedder: nomic-embed-text")
        return OllamaEmbeddings(model="nomic-embed-text")