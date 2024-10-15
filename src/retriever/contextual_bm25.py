import math
from typing import List


class ContextualBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.idf = {}

    def add_documents(self, documents: List[str]):
        new_docs = [doc.split() for doc in documents if doc.strip()]  # Skip empty documents
        if not new_docs:
            print("Warning: No valid documents to add.")
            return

        self.corpus.extend(new_docs)
        self.doc_lengths.extend(len(doc) for doc in new_docs)
        
        if self.corpus:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.corpus)
        else:
            self.avg_doc_length = 0

        self._calculate_idf()

    # Calculate IDF for each term in the corpus. IDF is the inverse document frequency of a term, 
    # which measures how important the term is in the corpus. 
    def _calculate_idf(self):
        if not self.corpus:
            print("Warning:Corpus is empty. Unbale to calculate IDF")
            return
        N = len(self.corpus)
        for doc in self.corpus:
            for term in set(doc):
                if term not in self.idf:
                    df = sum(1 for doc in self.corpus if term in doc) # calculate document frequency
                    self.idf[term] = math.log(N - df + 0.5) / (df + 0.5) +1

    # Calculate the score of a query for a given context. 
    # See https://www.elastic.co/blog/practical-bm25-part-2-the-score-function
    def score(self, query: str, context: str) -> List[float]:
        query_terms = query.split()
        context_terms = context.split()
        scores = []
        for i, doc in enumerate(self.corpus):
            score = 0
            for term in query_terms:
                if term in doc:
                    tf = doc.count(term)
                    context_boost = 1 + (context_terms.count(term) / len(context_terms))
                    numerator = self.idf[term] * tf * (self.k1 + 1) * context_boost
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length)                    
                    score += numerator / denominator
            scores.append(score)
        return scores
                    

    def generate_embeddings(self, texts: List[str], context: str) -> List[list[float]]:
        pass

