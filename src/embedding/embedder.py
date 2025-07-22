"""Text embedding using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import time
import cachetools


class TextEmbedder:
    """Handles text embedding using sentence-transformers with caching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """Initialize with specified model and cache."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.cache = cachetools.LRUCache(maxsize=cache_size)
        
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts into vectors with caching."""
        if not texts:
            return np.array([])
        
        embeddings = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                embedding = self.model.encode(
                    [text],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )[0]
                self.cache[text] = embedding
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text into a vector with caching."""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.embed_texts([text])[0]
        self.cache[text] = embedding
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim or 384  # Default fallback
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarities between query and multiple candidates."""
        # Normalize all embeddings
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        
        # Avoid division by zero
        valid_candidates = candidate_norms > 0
        similarities = np.zeros(len(candidate_embeddings))
        
        if query_norm > 0 and np.any(valid_candidates):
            # Compute cosine similarities for valid candidates
            similarities[valid_candidates] = (
                np.dot(candidate_embeddings[valid_candidates], query_embedding) / 
                (candidate_norms[valid_candidates] * query_norm)
            )
            
        return similarities


class TagEmbedder:
    """Handles embedding of tag descriptions and examples."""
    
    def __init__(self, embedder: TextEmbedder):
        """Initialize with a text embedder."""
        self.embedder = embedder
        
    def embed_tag_definition(self, description: str, examples: List[str]) -> np.ndarray:
        """Embed a tag definition by combining description and examples."""
        # Combine description and examples into a single text
        combined_text = description + " " + " ".join(examples)
        return self.embedder.embed_single_text(combined_text)
    
    def embed_tag_examples(self, examples: List[str]) -> np.ndarray:
        """Embed tag examples and return average embedding."""
        if not examples:
            return np.zeros(self.embedder.get_embedding_dimension())
            
        example_embeddings = self.embedder.embed_texts(examples)
        return np.mean(example_embeddings, axis=0)
    
    def embed_tag_description(self, description: str) -> np.ndarray:
        """Embed just the tag description."""
        return self.embedder.embed_single_text(description) 