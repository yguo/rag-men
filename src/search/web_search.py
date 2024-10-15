import time
from typing import Dict, List
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException


class WebSearch:
    def __init__(self):       
        self.max_retries = 5
        self.base_wait_time = 1 # start with 1 second wait time

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        for attempt in range(self.max_retries):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                    for r in results:
                        results.append({
                            "title": r['title'],
                            "description": r['description'],
                            "url": r['href']
                    })
                return results
            except DuckDuckGoSearchException as e:
                print(f"DuckDuckGoSearchException: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.base_wait_time * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Unable to complete the search.")
                    return []
            except Exception as e:
                print (f"Unexpected error during search: {e} ")                
        print("Max retries reached. Unable to complete the search.  ")    
        return results