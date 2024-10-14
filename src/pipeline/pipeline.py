from typing import List, Dict
from ..retriever.contextual_embeddings import ContextualEmbeddings
from ..retriever.contextual_bm25 import ContextualBM25
from ..search.web_search import WebSearch
from ..reranker.reranker_base import Reranker
from ..retriever.vector_store import VectorStore
from ..generator.answer_generator import AnswerGenerator

class ContextualRAGPipeline:
    def __init__(self):
        self.contextual_embeddings = ContextualEmbeddings()
        self.contextual_bm25 = ContextualBM25()
        self.web_search = WebSearch()
        self.reranker = Reranker()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()

    def add_document(self, text: str):
        embedding = self.contextual_embeddings.generate_embeddings([text], "")[0]
        self.vector_store.add_documents([text], [embedding])
        self.contextual_bm25.add_documents([text])

    def process_query(self, query: str) -> Dict:
        # Step 1: Retrieve relevant local documents
        query_embedding = self.contextual_embeddings.generate_embeddings([query], "")[0]
        local_results = self.vector_store.query(query_embedding, n_results=5)
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