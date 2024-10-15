from typing import Dict, List
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, collection_name: str = "local_knowledge_base"):
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)        

    def add_documents(self, texts: list[str], embeddings: list[list[float]], metadata: list[dict] = None, ids: list[str] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata if metadata else [{}] * len(texts),
            ids=ids
        )

    def query(self, query_embedding: list[float], n_results: int = 5):
        print(f"INFO: Collection: {self.collection}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def remove_documents(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def update_document(self, id: str, text: str, embedding: List[float], metadata: Dict = None):
        self.collection.update(
            ids=[id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None
        )


    def get_all_documents(self):
        return self.collection.get()
    
    def get_document_by_id(self, id: str):
        return self.collection.get(ids=[id])
