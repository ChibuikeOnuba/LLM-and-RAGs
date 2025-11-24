import numpy as np
import re
from collections import defaultdict
import pickle

class SimpleTokenizer:
    """ Basic tokenizer for text processing"""

    def tokenize(self, text):

        text = text.lower()

        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
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
    
class TFIDFVectorizer:
        """Simple TF-IDF vectorizer without external libraries"""

        def __init__(self, max_features=2000) -> None:
             self.max_features = max_features
             self.vocabulary = {}
             self.idf = {}
             self.tokenizer = SimpleTokenizer()


        def fit(self, documents):
             """Build vocabulary and calculate IDF"""

             df = defaultdict(int)
             all_tokens = set()

             for doc in documents:
                tokens = set(self.tokenizer.tokenize(doc))
                for token in tokens:
                    df[token] += 1
                    all_tokens.add(token)

             sorted_terms = sorted(df.items(), key=lambda x:x[1], reverse=True)
             top_terms = sorted_terms[:self.max_features]

             # Create vocabulary
             self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}

             num_docs = len(documents)
             for term, idx in self.vocabulary.items():
                 self.idf[term] = np.log((num_docs +1)/(df[term] + 1)) + 1 

        def transform(self, documents):
            """Transform documents to TF-IDF vectors"""
            vectors = np.zeros((len(documents), len(self.vocabulary)))

            for doc_idx, doc in enumerate(documents):
                tokens = self.tokenizer.tokenize(doc)

                # Calculate term frequency
                tf = defaultdict(int)
                for token in tokens:
                    if token in self.vocabulary:
                        tf[token] += 1

                doc_length = len(tokens) if tokens else 1

                for term, freq in tf.items():
                    if term in self.vocabulary:
                        vocab_idx = self.vocabulary[term]
                        tf_normalized = freq / doc_length
                        vectors[doc_idx, vocab_idx] = tf_normalized * self.idf[term]
            
            # L2 normalization
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms

            return vectors
        
        def fit_transform(self, documents):
            """Fit and transform in one step"""
            self.fit(documents)
            return self.transform(documents)
        

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

        top_indices = np.argsort(similarities)[::-1][top_k]

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

                 
