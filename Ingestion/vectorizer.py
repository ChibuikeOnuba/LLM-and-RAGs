import numpy as np
import re
from collections import defaultdict

class SimpleTokenizer:
    """ Basic tokenizer for text processing"""

    def tokenize(self, text):
        self.text = text
        text = text.lower()

        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens

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