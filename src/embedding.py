from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None 
        self.load_model()

    def load_model(self):
        print("Loading embedding model:", self.model_name)
        self.model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        print("Model loaded.")

    def get_model(self):
        return self.model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.embed_documents(texts)
    
    def embed_one(self, text: str) -> np.ndarray:
        return self.model.embed_query(text)