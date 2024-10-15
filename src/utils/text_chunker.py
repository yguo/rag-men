import nltk
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        nltk.download('punkt', quiet=True)

    def chunk_text(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_size = sum(len(s) for s in current_chunk[-2:])
                current_chunk = current_chunk[-2:] if overlap_size <= self.overlap else []
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    