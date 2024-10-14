from typing import List, Dict
import ollama

class Reranker:
    def __init__(self, model_name: str = "llama3.1"):
        self.model = model_name

    def rerank(self, query: str, context: str, results: List[Dict]) -> List[Dict]:
        reranked_results = []
        for result in results:
            prompt = f"""
            Query: {query}
            Context: {context}
            Document: {result['snippet']}

            Rate the relevance of this document to the query and context on a scale of 0 to 10, where 0 is completely irrelevant and 10 is highly relevant.
            Only respond with a number between 0 and 10.
            """
            response = ollama.generate(model=self.model, prompt=prompt)
            try:
                relevance_score = float(response['response'].strip())
                result['relevance_score'] = relevance_score
                reranked_results.append(result)
            except ValueError:
                print(f"Error parsing relevance score for result: {result['title']}")

        return sorted(reranked_results, key=lambda x: x['relevance_score'], reverse=True)