from typing import Dict, List
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, 
                 collection_name: str = "local_knowledge_base", 
                 persist_directory: str = "./chroma_db", embedding_provider=None):
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True)) 
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_provider = embedding_provider

    def add_documents(self, texts: list[str],  metadata: list[dict] = None, ids: list[str] = None):
        print(f"DEBUG VECTOR STORE ADD DOCUMENTS: Adding {len(texts)} documents to the collection")
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = self.embedding_provider.generate_embeddings(texts, "")
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata if metadata else [{}] * len(texts),
            ids=ids
        )

    # similarity search using cosine similarity. 
    def similarity_search(self, query:str, context:str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedding_provider.generate_embeddings([query], context)[0]
        # Calculate cosine similarity between query embedding and all document embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [{"text": doc, "score": score} for doc, score in zip(results["documents"][0], results["distances"][0])]

    def query(self, query_embedding: list[float], n_results: int = 5):
        print(f"DEBUG: Collection: {self.collection}")
        print(f"DEBUG: Query embedding shape: {len(query_embedding)}")
        print(f"DEBUG: Number of results requested: {n_results}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def remove_documents(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def update_document(self, id: str, text: str, metadata: Dict = None):
        embedding = self.embedding_provider.generate_embeddings([text], "")
        self.collection.update(
            ids=[id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None
        )


    def get_all_documents(self):
        return self.collection.get()
    
    def get_document_by_id(self, id: str):
        result = self.collection.get(ids=[id])
        print(f"DEBUG: get_document_by_id result for {id}: {result}")  # Add this debug line
        return result
    
    def clear_database(self):
        self.client.reset()
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
