"""FAISS-based semantic routing for note classification."""

import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

from ..config.tag_loader import TagDefinition
from ..embedding.embedder import TextEmbedder, TagEmbedder


@dataclass
class RoutingResult:
    """Result of routing a note to tags."""
    note_id: str
    text: str
    timestamp: str
    routed_tag: str
    similarity_score: float
    confidence: float
    tag_description: str
    tag_examples: List[str]
    routed_by: str
    version: int
    metadata: Dict[str, Any]


class FAISSRouter:
    """Routes notes to tags using FAISS for efficient similarity search."""
    
    def __init__(self, embedder: TextEmbedder, tag_embedder: TagEmbedder):
        """Initialize with embedders."""
        self.embedder = embedder
        self.tag_embedder = tag_embedder
        self.index: Optional[faiss.IndexFlatIP] = None
        self.tag_definitions: List[TagDefinition] = []
        self.tag_embeddings: List[Any] = []
        
    def build_index(self, tags: List[TagDefinition]) -> None:
        """Build FAISS index from tag definitions."""
        self.tag_definitions = tags
        self.tag_embeddings = []
        
        # Embed each tag definition
        for tag in tags:
            # Create multiple embeddings per tag: description + examples
            tag_embedding = self.tag_embedder.embed_tag_definition(
                tag.description, tag.examples
            )
            self.tag_embeddings.append(tag_embedding)
            
        # Convert to numpy array
        embeddings_array = np.array(self.tag_embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings_array)
        
    def route_note(self, note_id: str, text: str, timestamp: str, 
                   top_k: int = 3, confidence_threshold: float = 0.5) -> RoutingResult:
        """Route a single note to the best matching tag."""
        start_time = time.time()
        
        # Embed the note
        note_embedding = self.embedder.embed_single_text(text)
        note_embedding = note_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(note_embedding)
        
        # Ensure index is built
        if self.index is None:
            raise RuntimeError("FAISS index has not been built. Call build_index() first.")
        
        # Search for similar tags
        similarities, indices = self.index.search(note_embedding, top_k)
        
        # Get best match
        best_idx = indices[0][0]
        best_similarity = similarities[0][0]
        best_tag = self.tag_definitions[best_idx]
        
        # Calculate confidence (simple mapping from similarity to confidence)
        confidence = self._calculate_confidence(best_similarity, best_tag.confidence_threshold)
        
        # Create routing result
        result = RoutingResult(
            note_id=note_id,
            text=text,
            timestamp=timestamp,
            routed_tag=best_tag.tag,
            similarity_score=float(best_similarity),
            confidence=confidence,
            tag_description=best_tag.description,
            tag_examples=best_tag.examples,
            routed_by="faiss-v1",
            version=1,
            metadata={
                "embedding_model": self.embedder.model_name,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "disambiguation_used": False,
                "top_k_candidates": [
                    {
                        "tag": self.tag_definitions[idx].tag,
                        "similarity": float(sim)
                    }
                    for sim, idx in zip(similarities[0], indices[0])
                ]
            }
        )
        
        return result
    
    def route_notes_batch(self, notes: List[Tuple[str, str, str]], 
                         top_k: int = 3, 
                         confidence_threshold: float = 0.5) -> List[RoutingResult]:
        """Route multiple notes in batch."""
        results = []
        
        for note_id, text, timestamp in notes:
            try:
                result = self.route_note(note_id, text, timestamp, top_k, confidence_threshold)
                results.append(result)
            except Exception as e:
                # Log error and continue with next note
                print(f"Error routing note {note_id}: {e}")
                continue
                
        return results
    
    def _calculate_confidence(self, similarity: float, tag_threshold: float) -> float:
        """Calculate confidence score from similarity and tag threshold."""
        # Simple confidence calculation: similarity relative to tag threshold
        if similarity >= tag_threshold:
            # Above threshold: confidence increases with similarity
            confidence = 0.5 + 0.5 * (similarity - tag_threshold) / (1.0 - tag_threshold)
        else:
            # Below threshold: confidence decreases
            confidence = 0.5 * (similarity / tag_threshold)
            
        return max(0.0, min(1.0, confidence))
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tag index."""
        if not self.index:
            return {"error": "Index not built"}
            
        return {
            "num_tags": len(self.tag_definitions),
            "embedding_dimension": self.index.d,
            "index_type": "FlatIP",
            "tags": [tag.tag for tag in self.tag_definitions]
        } 