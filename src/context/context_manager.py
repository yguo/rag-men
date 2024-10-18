import sqlite3
from typing import List, Tuple
import json

class ContextManager:
    def __init__(self, db_path: str = 'context_history.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS context_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            embedding TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()

    def add_entry(self, query: str, response: str, embedding: List[float]):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO context_history (query, response, embedding)
        VALUES (?, ?, ?)
        ''', (query, response, json.dumps(embedding)))
        self.conn.commit()

    def get_recent_contexts(self, limit: int = 5) -> List[Tuple[str, str, List[float]]]:
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT query, response, embedding
        FROM context_history
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        return [(query, response, json.loads(embedding)) for query, response, embedding in results]

    def close(self):
        self.conn.close()