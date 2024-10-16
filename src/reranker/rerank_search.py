from reranker.reranker_base import Reranker

class SearchReranker(Reranker):
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)

    def rerank_search_results(self, query: str, results: list[dict]) -> list[dict]:
        # Implement search-specific reranking logic here
        pass
