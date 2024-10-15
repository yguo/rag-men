from typing import List, Dict
from ..pipeline.pipeline import ContextualRAGPipeline

class Evaluator:
    def __init__(self, pipeline: ContextualRAGPipeline):
        self.pipeline = pipeline

    def evaluate(self, test_queries: List[str], ground_truth: List[str]) -> Dict:
        results = []
        for query, truth in zip(test_queries, ground_truth):
            response = self.pipeline.process_query(query)
            results.append({
                "query": query,
                "ground_truth": truth,
                "generated_answer": response["answer"],
                "sources": response["sources"]
            })
        
        # Implement your evaluation metrics here
        # For example, you could use BLEU score, ROUGE, or custom metrics

        return {
            "results": results,
            "metrics": {
                "accuracy": 0.0,  # Placeholder for actual metric
                "relevance": 0.0  # Placeholder for actual metric
            }
        }