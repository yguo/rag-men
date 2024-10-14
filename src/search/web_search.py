from typing import Dict, List
from duckduckgo_search import DDGS


class WebSearch:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        results = []
        for r in self.ddgs.text(query, max_results=num_results):
            results.append({
                "title": r['title'],
                "description": r['description'],
                "url": r['href']
            })
        return results