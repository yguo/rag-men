import ollama
from typing import List, Dict

class AnswerGenerator:
    def __init__(self, model_name: str = "llama2"):
        self.model = model_name

    def generate_answer(self, query: str, context: str, reranked_results: List[Dict]) -> str:
        prompt = self._construct_prompt(query, context, reranked_results)
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    def _construct_prompt(self, query: str, context: str, reranked_results: List[Dict]) -> str:
        sources = "\n".join([f"Source {i+1}: {result['text']}" for i, result in enumerate(reranked_results)])
        
        prompt = f"""
        Query: {query}

        Context: {context}

        Relevant Sources:
        {sources}

        Based on the query, context, and relevant sources provided, please generate a comprehensive and accurate answer. 
        Ensure that your response directly addresses the query and incorporates information from the given sources. 
        If the sources contain conflicting information, please mention this and provide a balanced view.

        Answer:
        """
        
        return prompt