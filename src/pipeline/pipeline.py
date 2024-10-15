from typing import List, Dict

from ..utils.text_chunker import TextChunker
from ..retriever.contextual_embeddings import ContextualEmbeddings, OllamaEmbeddings
from ..retriever.contextual_bm25 import ContextualBM25
from ..search.web_search import WebSearch
from ..reranker.reranker_base import Reranker
from ..retriever.vector_store import VectorStore
from ..generator.answer_generator import AnswerGenerator

class ContextualRAGPipeline:
    def __init__(self):        
        ollama_provider = OllamaEmbeddings(model_name="llama3.1")
        self.contextual_embeddings = ContextualEmbeddings(provider=ollama_provider)
        self.contextual_bm25 = ContextualBM25()
        self.web_search = WebSearch()
        self.reranker = Reranker()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.text_chunker = TextChunker()
    
    def add_document_chunk(self, chunk: str, metadata: Dict):
        try:
            embedding = self.contextual_embeddings.generate_embeddings([chunk], "")[0]
            if not embedding:
                print("Error: No embedding generated to add document. The embedding service might be unavailable.")
                return False
            # add more metadata 
            chunk_metadata =  metadata.copy()
            chunk_metadata["content_summary"] = chunk[:100] #placeholder for now
            chunk_metadata["is_chunk"] = True
            chunk_metadata["chunk_index"] = metadata.get("chunk_index", 0) 

            self.vector_store.add_documents([chunk], [embedding], [chunk_metadata])
            self.contextual_bm25.add_documents([chunk])
            return True
        except Exception as e:
            print(f"Error adding document chunk: {e}")
            return False

            self.vector_store.add_documents([text], [embedding], metadata)

    def add_document(self, text: str,metadata):
        chunks = self.text_chunker.chunk_text(text)
        success = True
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            if not self.add_document_chunk(chunk, chunk_metadata):
                success = False
        return success

    def process_query(self, query: str) -> Dict:
        # Step 1: Retrieve relevant local documents
        query_embedding = self.contextual_embeddings.generate_embeddings([query], "")[0]
        local_results = self.vector_store.query(query_embedding, n_results=20)
        local_texts = local_results['documents'][0]
        local_scores = local_results['distances'][0]

        # Step 2: Perform web search
        web_results = self.web_search.search(query)
        web_texts = [result['snippet'] for result in web_results]

        # Step 3: Combine local and web results and initialize scores
        all_texts = local_texts + web_texts
        all_scores = local_scores + [0] * len(web_results)  # Initialize web scores to 0

        # Step 4: Perform contextual BM25 scoring
        self.contextual_bm25.add_documents(all_texts)
        bm25_scores = self.contextual_bm25.score(query, " ".join(all_texts))

        # Step 5: Combine vector similarity and BM25 scores
        combined_results = []
        for i, (text, vector_score, bm25_score) in enumerate(zip(all_texts, all_scores, bm25_scores)):
            combined_score = (vector_score + bm25_score) / 2
            result = {
                "text": text,
                "combined_score": combined_score,
                "is_local": i < len(local_texts)
            }
            if not result["is_local"]:
                result.update(web_results[i - len(local_texts)])
            combined_results.append(result)

        # Step 6: Rerank results
        reranked_results = self.reranker.rerank(query, " ".join(all_texts), combined_results)

        # Step 7: Generate answer
        context = " ".join([r['text'] for r in reranked_results[:3]])
        answer = self.answer_generator.generate_answer(query, context, reranked_results[:3])

        return {
            "answer": answer,
            "sources": reranked_results[:5]
        }