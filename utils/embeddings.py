import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings

class RESTGeminiEmbeddings(Embeddings):
    """
    A custom LangChain embeddings class that uses the Gemini REST API directly.
    This bypasses gRPC which is known to cause Deadline Exceeded hangs on
    certain environments (like macOS + Python 3.13).
    """
    def __init__(self, google_api_key: str, model: str = "models/gemini-embedding-001"):
        self.api_key = google_api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"
        requests_list = [{"model": self.model, "content": {"parts": [{"text": t}]}} for t in texts]
        payload = {"requests": requests_list}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return [r["values"] for r in resp.json()["embeddings"]]

    def embed_query(self, text: str) -> List[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}"
        payload = {"model": self.model, "content": {"parts": [{"text": text}]}}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
