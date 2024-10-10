import unittest
from rag_men.retriever import Retriever
from rag_men.retriever.vector_store import VectorStore

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.vector_store = VectorStore()
        self.retriever = Retriever(vector_store=self.vector_store)

    def test_retriever_initialization(self):
        self.assertIsInstance(self.retriever, Retriever)
        self.assertIsInstance(self.vector_store, VectorStore)

    def test_retrieve_documents(self):
        query = "What is the meaning of life?"
        results = self.retriever.retrieve(query)
        self.assertIsInstance(results, list)
    
    def test_retrieve_empty_query(self):
        query = ""
        results = self.retriever.retrieve(query)
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()