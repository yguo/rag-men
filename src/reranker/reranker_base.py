from typing import List, Dict
import ollama
from ollama._types import ResponseError
import re

class Reranker:
    def __init__(self, model_name: str = "llama2"):
        self.model = model_name

    def rerank(self, query: str, context: str, results: List[Dict]) -> List[Dict]:
        reranked_results = []
        for i, result in enumerate(results):
            prompt = f"""
            Query: {query}
            Context: {context}
            Document: {result['text'][:500]}  # Limit document text to 500 characters

            Rate the relevance of this document to the query and context on a scale of 0 to 10, where 0 is completely irrelevant and 10 is highly relevant.
            Only respond with a number between 0 and 10. DO NOT include any other text. ONLY THE SCORE.  NO extra explanation. OUTPUT a number.
            """
            try:
                response = ollama.generate(model=self.model, prompt=prompt)
                response_text = response['response'].strip()
                print(f"DEBUG: Raw model response for result {i}: {response_text}")

                # Try to extract a number from the response
                match = re.search(r'\b(\d+(?:\.\d+)?)\b', response_text)
                if match:
                    relevance_score = float(match.group(1))
                    if 0 <= relevance_score <= 10:
                        result['relevance_score'] = relevance_score
                        reranked_results.append(result)
                    else:
                        print(f"Warning: Relevance score {relevance_score} out of range for result {i}")
                        result['relevance_score'] = 5  # Assign a neutral score
                        reranked_results.append(result)
                else:
                   # print(f"Error: Could not parse relevance score from response for result {i}")
                    result['relevance_score'] = 5  # Assign a neutral score
                    reranked_results.append(result)
            except Exception as e:
                print(f"Unexpected error when processing result {i}: {e}")
                result['relevance_score'] = 5  # Assign a neutral score
                reranked_results.append(result)
        
        return sorted(reranked_results, key=lambda x: x['relevance_score'], reverse=True)
