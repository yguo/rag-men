from typing import List, Dict
import requests
import logging

class ContextualEmbeddings:
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.api_url = f"http://127.0.0.1:11434/api/embeddings"
        self.logger = logging.getLogger(__name__)

    def generate_embeddings(self, texts: List[str], context: str) -> List[list[float]]:
        embeddings = []
        for text in texts:
            prompt = f"Context: {context}\n\nText: {text}"
            try:
                response = requests.post(self.api_url, json={"model": self.model_name, "prompt": prompt})
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Ollama API failed to generate embeddings for text: {text}")
                self.logger.error(f"Error: {e}")
            except KeyError as e:
                self.logger.error(f"Ollama API returned an invalid response: {response.text}")
                self.logger.error(f"Error: {e}")                
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                continue
        return embeddings

