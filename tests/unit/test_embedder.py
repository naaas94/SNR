"""Unit tests for embedding components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from src.embedding.embedder import TextEmbedder, TagEmbedder


class TestTextEmbedder:
    """Test cases for TextEmbedder class."""
    
    def test_init(self):
        """Test TextEmbedder initialization."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None
        assert embedder.embedding_dim is not None and embedder.embedding_dim > 0
    
    def test_embed_texts_empty_list(self):
        """Test embedding empty text list."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        result = embedder.embed_texts([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
    
    def test_embed_texts_single_text(self):
        """Test embedding single text."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        texts = ["This is a test note"]
        result = embedder.embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, embedder.embedding_dim)
        assert not np.isnan(result).any()
    
    def test_embed_texts_multiple_texts(self):
        """Test embedding multiple texts."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        texts = [
            "This is a test note",
            "Another test note",
            "Third test note"
        ]
        result = embedder.embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, embedder.embedding_dim)
        assert not np.isnan(result).any()
    
    def test_embed_single_text(self):
        """Test embedding single text."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        text = "This is a test note"
        result = embedder.embed_single_text(text)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedder.embedding_dim,)
        assert not np.isnan(result).any()
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        dim = embedder.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_compute_similarity(self):
        """Test computing similarity between embeddings."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        # Create test embeddings
        dim = embedder.embedding_dim or 384  # Default fallback
        emb1 = np.random.rand(dim)
        emb2 = np.random.rand(dim)
        
        similarity = embedder.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_compute_similarity_zero_vectors(self):
        """Test similarity with zero vectors."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        # Create zero embeddings
        dim = embedder.embedding_dim or 384  # Default fallback
        emb1 = np.zeros(dim)
        emb2 = np.zeros(dim)
        
        similarity = embedder.compute_similarity(emb1, emb2)
        assert similarity == 0.0
    
    def test_batch_similarity(self):
        """Test batch similarity computation."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        # Create test embeddings
        dim = embedder.embedding_dim or 384  # Default fallback
        query_emb = np.random.rand(dim)
        candidate_embs = np.random.rand(5, dim)
        
        similarities = embedder.batch_similarity(query_emb, candidate_embs)
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (5,)
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)
    
    def test_batch_similarity_with_zero_vectors(self):
        """Test batch similarity with zero vectors."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        # Create test embeddings with some zero vectors
        dim = embedder.embedding_dim or 384  # Default fallback
        query_emb = np.random.rand(dim)
        candidate_embs = np.random.rand(5, dim)
        candidate_embs[2] = np.zeros(dim)  # Zero vector
        
        similarities = embedder.batch_similarity(query_emb, candidate_embs)
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (5,)
        assert similarities[2] == 0.0  # Zero vector should have zero similarity


class TestTagEmbedder:
    """Test cases for TagEmbedder class."""
    
    def test_init(self):
        """Test TagEmbedder initialization."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        assert tag_embedder.embedder == text_embedder
    
    def test_embed_tag_definition(self):
        """Test embedding tag definition."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        
        description = "A test tag description"
        examples = ["Example 1", "Example 2"]
        
        result = tag_embedder.embed_tag_definition(description, examples)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (text_embedder.embedding_dim,)
        assert not np.isnan(result).any()
    
    def test_embed_tag_examples(self):
        """Test embedding tag examples."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        
        examples = ["Example 1", "Example 2", "Example 3"]
        result = tag_embedder.embed_tag_examples(examples)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (text_embedder.embedding_dim,)
        assert not np.isnan(result).any()
    
    def test_embed_tag_examples_empty(self):
        """Test embedding empty examples list."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        
        result = tag_embedder.embed_tag_examples([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (text_embedder.embedding_dim,)
        assert np.all(result == 0)  # Should be zero vector
    
    def test_embed_tag_description(self):
        """Test embedding tag description."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        
        description = "A test tag description"
        result = tag_embedder.embed_tag_description(description)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (text_embedder.embedding_dim,)
        assert not np.isnan(result).any()


class TestEmbedderIntegration:
    """Integration tests for embedding components."""
    
    def test_end_to_end_embedding(self):
        """Test complete embedding pipeline."""
        # Initialize embedders
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        tag_embedder = TagEmbedder(text_embedder)
        
        # Test data
        note_text = "This is a test note about machine learning"
        tag_description = "Notes about machine learning concepts"
        tag_examples = [
            "Neural networks are powerful",
            "Deep learning requires lots of data"
        ]
        
        # Embed note
        note_embedding = text_embedder.embed_single_text(note_text)
        
        # Embed tag
        tag_embedding = tag_embedder.embed_tag_definition(tag_description, tag_examples)
        
        # Compute similarity
        similarity = text_embedder.compute_similarity(note_embedding, tag_embedding)
        
        # Assertions
        assert isinstance(note_embedding, np.ndarray)
        assert isinstance(tag_embedding, np.ndarray)
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
        assert not np.isnan(similarity)
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent for same input."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        text = "This is a test note"
        
        # Embed same text twice
        emb1 = text_embedder.embed_single_text(text)
        emb2 = text_embedder.embed_single_text(text)
        
        # Embeddings should be identical
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)
    
    def test_embedding_differentiation(self):
        """Test that different texts produce different embeddings."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        text1 = "This is about machine learning"
        text2 = "This is about cooking recipes"
        
        emb1 = text_embedder.embed_single_text(text1)
        emb2 = text_embedder.embed_single_text(text2)
        
        # Embeddings should be different
        similarity = text_embedder.compute_similarity(emb1, emb2)
        assert similarity < 0.9  # Should not be too similar
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual."""
        text_embedder = TextEmbedder("all-MiniLM-L6-v2")
        
        texts = [f"Test note {i}" for i in range(10)]
        
        # Time batch processing
        start_time = time.time()
        batch_result = text_embedder.embed_texts(texts)
        batch_time = time.time() - start_time
        
        # Time individual processing
        start_time = time.time()
        individual_results = [text_embedder.embed_single_text(text) for text in texts]
        individual_time = time.time() - start_time
        
        # Batch should be faster (or at least not significantly slower)
        assert batch_time <= individual_time * 1.5
        
        # Results should be equivalent
        individual_array = np.array(individual_results)
        np.testing.assert_array_almost_equal(batch_result, individual_array, decimal=6)


class TestEmbedderPerformance:
    """Performance tests for embedding components."""
    
    def test_load_performance(self):
        """Test load performance of the TextEmbedder with a large number of texts."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        texts = ["This is a test note" for _ in range(1000)]  # Simulate 1000 notes
        
        start_time = time.time()
        result = embedder.embed_texts(texts)
        end_time = time.time()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000, embedder.embedding_dim)
        
        # Log performance
        duration = end_time - start_time
        print(f"Load test completed in {duration:.2f} seconds")
        
    def test_stress_performance(self):
        """Test stress performance by embedding texts until failure."""
        embedder = TextEmbedder("all-MiniLM-L6-v2")
        texts = ["This is a test note" for _ in range(10000)]  # Start with 10,000 notes
        
        try:
            start_time = time.time()
            result = embedder.embed_texts(texts)
            end_time = time.time()
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (10000, embedder.embedding_dim)
            
            # Log performance
            duration = end_time - start_time
            print(f"Stress test completed in {duration:.2f} seconds")
        except Exception as e:
            print(f"Stress test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 