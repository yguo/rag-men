from typing import List, Dict, Tuple
import numpy as np
from ..context.context_manager import ContextManager
from ..utils.text_chunker import TextChunker
from ..retriever.contextual_embeddings import ContextualEmbeddings, OllamaEmbeddings
from ..retriever.contextual_bm25 import ContextualBM25
from ..search.web_search import WebSearch
from ..reranker.reranker_base import Reranker
from ..retriever.vector_store import VectorStore
from ..generator.answer_generator import AnswerGenerator
from ..context.query_processing.query_expander import QueryExpander

class ContextualRAGPipeline:
    def __init__(self):        
        ollama_provider = OllamaEmbeddings(model_name="llama3.1")
        self.contextual_embeddings = ContextualEmbeddings(provider=ollama_provider)
        self.contextual_bm25 = ContextualBM25()
        self.web_search = WebSearch()
        self.reranker = Reranker()
        self.vector_store = VectorStore(persist_directory="./chroma_db", embedding_provider=self.contextual_embeddings)
        self.answer_generator = AnswerGenerator(model_name="llama3.1")
        self.text_chunker = TextChunker()
        self.query_expander = QueryExpander()

        self.context_manager = ContextManager()
        self.context_window_size = 5

    def generate_context(self,query:str) -> str:
        query_embedding = self.contextual_embeddings.generate_embeddings([query], "")[0]
            
        # Get recent contexts
        recent_contexts = self.context_manager.get_recent_contexts(self.context_window_size - 1)
        
        # Calculate relevance scores
        relevance_scores = self.calculate_relevance_scores(query_embedding, [emb for _, _, emb in recent_contexts])
        
        # Generate weighted context
        weighted_context = self.generate_weighted_context(query, relevance_scores, recent_contexts)
        
        return weighted_context


    def calculate_relevance_scores(self, current_query_embedding: List[float], historical_embeddings: List[List[float]]) -> List[float]:
        if not historical_embeddings:
            return []
        similarities = [
            np.dot(current_query_embedding,hist_emb) / (np.linalg.norm(current_query_embedding) * np.linalg.norm(hist_emb))
            for hist_emb in historical_embeddings
        ]
        recent_weights = np.linspace(0.5,1, len(similarities))
        weighted_similarities = np.array(similarities) * recent_weights
        return list(weighted_similarities / np.sum(weighted_similarities)) if weighted_similarities.size>  0 else []

    def generate_weighted_context(self, current_query: str, relevance_scores: List[float], recent_contexts: List[Tuple[str, str, List[float]]]) -> str:
        # Combine historical queries and responses based on their relevance scores
        weighted_contexts = [
            f"{score:.2f} * Q: {query} A: {response}"
            for score, (query, response, _) in zip(relevance_scores, recent_contexts)
        ]
        
        # Add the current query with full weight
        weighted_contexts.append(f"1.00 * Q: {current_query}")
        
        return " | ".join(weighted_contexts)


    def add_document_chunk(self, chunk: str, metadata: Dict):
        try:
             # Generate a context summary from the metadata
            context = f"Title: {metadata.get('title', 'Unknown')}\n"
            context += f"Author: {metadata.get('author', 'Unknown')}\n"
            context += f"Subject: {metadata.get('subject', 'Unknown')}\n"
            context += f"Chunk {metadata.get('chunk_index', 0)} of {metadata.get('total_chunks', 1)}\n"

           # embedding = self.contextual_embeddings.generate_embeddings([chunk], context)[0]
                   
            #if not embedding:
             #   print("Error: No embedding generated to add document. The embedding service might be unavailable.")
            #    return False
            # add more metadata 
            chunk_metadata =  metadata.copy()
            chunk_metadata["content_summary"] = chunk[:100] #placeholder for now
            chunk_metadata["is_chunk"] = True
            chunk_metadata["chunk_index"] = metadata.get("chunk_index", 0) 

            # Generate a unique ID for the chunk
            chunk_id = f"doc_{metadata.get('file_name', 'unknown')}_{metadata.get('chunk_index', 0)}"
             # Check if the document already exists
            existing_doc = self.vector_store.get_document_by_id(chunk_id)
            print(f"DEBUG: existing_doc: {existing_doc}")
            if existing_doc['ids']:
                print(f"Document with ID {chunk_id} already exists. Updating...")
                self.vector_store.update_document(chunk_id, chunk, chunk_metadata)
            else:
                print(f"DEBUG: Adding new document with ID {chunk_id}")
                self.vector_store.add_documents([chunk], [chunk_metadata], [chunk_metadata], [chunk_id])

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
        expanded_query = self.query_expander.expand_query_with_pos(query)
        context = self.generate_context(expanded_query)
        print(f"DEBUG: Context: {context}")

        
        
        # Step 1: Perform web search
        web_results = self.web_search.search(query)        
        web_texts = []
        for result in web_results:
            if 'description' in result:
                web_texts.append(result['description'])
            elif 'body' in result:
                web_texts.append(result['body'])
            elif 'title' in result:
                web_texts.append(result['title'])
            else:   
                print(f"Warning: 'snippet' or 'body' or 'title' not found in web result: {result}")

        # Step 2: Retrieve relevant local documents
        # query_embedding = self.contextual_embeddings.generate_embeddings([query], "")[0]

       
        local_results = self.vector_store.similarity_search(query, context, top_k=20)
        local_texts = local_results['documents'][0] if local_results['ids'][0] else []
        local_scores = local_results['distances'][0] if local_results['ids'][0] else []
       
      
        # Step 3: Combine local and web results and initialize scores
        all_texts = local_texts + web_texts
        if not all_texts:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information to answer your query.",
                "sources": [],
                "web_texts": web_texts
            }
        all_scores = local_scores + [0] * len(web_results)  # Initialize web scores to 0

        # Step 4: Perform contextual BM25 scoring on all texts
        self.contextual_bm25.add_documents(all_texts)
        bm25_scores = self.contextual_bm25.score(query, context)

        def normalize(scores):
            min_score = min(scores)
            max_score = max(scores)
            return [(score - min_score) / (max_score - min_score) if max_score - min_score > 0 else 0.5 for score in scores]
        vector_scores = normalize(local_scores + [0] * len(web_texts))
        bm25_scores = normalize(bm25_scores)



        # Step 5: Normalize Combine vector similarity and BM25 scores
        combined_results = []
        for i, (text, vector_score, bm25_score) in enumerate(zip(all_texts, vector_scores, bm25_scores)):
            combined_score = (vector_score + bm25_score) / 2
            result = {
                "text": text,
                "bm25_score": bm25_score,
                "combined_score": combined_score,
                "is_local": i < len(local_texts)
            }
            if not result["is_local"]:
                result.update(web_results[i - len(local_texts)])
            combined_results.append(result)       
        # Step 6: Rerank results
        # print(f"DEBUG: Combined results structure: {combined_results[:2]}")  # Print first two items for brevity
        reranked_results = self.reranker.rerank(query, " ".join(all_texts), combined_results)

        # Step 7: Generate answer
        answer = self.answer_generator.generate_answer(query, context, reranked_results[:3])

        # step 8 : store the query and answer in the context manager
        query_embedding = self.contextual_embeddings.generate_embeddings([query], "")[0]
        self.context_manager.add_entry(query, answer, query_embedding)


        return {
            "answer": answer,
            "sources": reranked_results[:5]
        }
    def __del__(self):
        self.context_manager.close()
