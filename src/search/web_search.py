from serpapi import GoogleSearch
from typing import Dict, List
from config import config

class WebSearch:
    def __init__(self):
        self.api_key = config.get('API', 'SERPAPI_API_KEY')
        
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY is not set in the configuration")

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": max_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            formatted_results = []
            for item in results.get("organic_results", [])[:max_results]:
                formatted_results.append({
                    "title": item.get("title"),
                    "description": item.get("snippet"),
                    "url": item.get("link")
                })

            print(f"INFO: Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []