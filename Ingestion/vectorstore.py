import numpy as np
import pickle
from vectorizer import SimpleTokenizer, TFIDFVectorizer


class DocumentChunker:
    """ Splits documents into chunks of specified token size"""

    def __init__(self, chunk_size=512, overlap=50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = SimpleTokenizer()

    def chunk_document(self, text):
        """Split a document into overlapping chunks"""
        tokens = self.tokenizer.tokenize(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size-self.overlap

        return chunks

class VectorStore:
    """Store and search document chunks using vector embeddings"""

    def __init__(self, chunk_size=512, overlap=50, max_features=2000) -> None:
        self.chunker = DocumentChunker(chunk_size, overlap)
        self.vectorizer = TFIDFVectorizer(max_features)
        self.chunks = []
        self.vectors = None
        self.metadata = []

    def add_documents(self, documents, doc_names=None):
        """Add documents to the vector store"""

        if doc_names is None:
            doc_names = [f"doc_{i}" for i in range(len(documents))]

        # Chunk all documents
        for doc_idx, (doc, doc_name) in enumerate(zip(documents, doc_names)):
            doc_chunks = self.chunker.chunk_document(doc)
            for chunk_idx, chunk in enumerate(doc_chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    'doc_name':doc_name,
                    'doc_idx':doc_idx,
                    'chunk_idx':chunk_idx
                })


        self.vectors = self.vectorizer.fit_transform(self.chunks)

        print(f"Added {len(documents)} documents")
        print(f"Created {len(self.chunks)} chunks")
        print(f"Vector dimensions: {self.vectors.shape[1]}")

    def search (self, query, top_k=5):
        """Search for the most similar chunks to query"""

        if self.vectors is None or len(self.chunks) == 0:
            return []
        

        # Transform query to vector
        query_vector = self.vectorizer.transform([query])

        similarities = np.dot(self.vectors, query_vector.T).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_indices = np.atleast_1d(top_indices).tolist()
        

        results = []

        for idx in top_indices:
            results.append({
                'chunk':self.chunks[idx],
                'score':float(similarities[idx]),
                'metadata':self.metadata[idx]
            })

        return results
    
    def save(self, filepath):
        """Save vector store to disk"""
        data = {
            'chunk': self.chunks,
            'vectors': self.vectors,
            'metadata': self.metadata,
            'vectorizer': self.vectorizer
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to {filepath}")

    def load(self, filepath):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.vectors = data['vectors']
        self.metadata = data['metadata']
        self.vectorizer = data['vectorizer']
        print(f"Vector store loaded from {filepath}")