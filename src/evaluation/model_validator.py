"""Model validation and comparison for embedding models."""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..embedding.embedder import TextEmbedder, TagEmbedder
from ..routing.route_with_faiss import FAISSRouter
from ..config.tag_loader import TagLoader
from .metrics import EvaluationMetrics, calculate_precision_recall


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple embedding models."""
    model_name: str
    precision: float
    recall: float
    f1_score: float
    processing_time_ms: float
    memory_usage_mb: float
    embedding_dimension: int
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "processing_time_ms": self.processing_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "embedding_dimension": self.embedding_dimension,
            "model_size_mb": self.model_size_mb
        }


class ModelValidator:
    """Validates and compares different embedding models."""
    
    def __init__(self, test_data: List[Tuple[str, str, str]], tags_file: str = "tags/default_tags.yaml"):
        """
        Initialize validator with test data.
        
        Args:
            test_data: List of (note_id, text, true_tag) tuples
            tags_file: Path to tag definitions file
        """
        self.test_data = test_data
        self.tag_loader = TagLoader(tags_file)
        self.tags = self.tag_loader.load_tags()
        self.available_tags = [tag.tag for tag in self.tags]
        
    def compare_models(self, model_names: List[str]) -> List[ModelComparisonResult]:
        """
        Compare multiple embedding models on the test dataset.
        
        Args:
            model_names: List of sentence-transformer model names to compare
            
        Returns:
            List of ModelComparisonResult objects
        """
        results = []
        
        for model_name in model_names:
            print(f"Evaluating model: {model_name}")
            try:
                result = self._evaluate_single_model(model_name)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
                
        return results
    
    def _evaluate_single_model(self, model_name: str) -> ModelComparisonResult:
        """Evaluate a single embedding model."""
        start_time = time.time()
        
        # Initialize model and router
        embedder = TextEmbedder(model_name)
        tag_embedder = TagEmbedder(embedder)
        router = FAISSRouter(embedder, tag_embedder)
        
        # Build index
        router.build_index(self.tags)
        
        # Make predictions
        predicted_tags = []
        confidence_scores = []
        processing_times = []
        
        for note_id, text, _ in self.test_data:
            try:
                result = router.route_note(note_id, text, "2025-01-27T00:00:00")
                predicted_tags.append(result.routed_tag)
                confidence_scores.append(result.confidence)
                processing_times.append(result.metadata.get("processing_time_ms", 0))
            except Exception as e:
                print(f"Error routing note {note_id}: {e}")
                predicted_tags.append("unknown")
                confidence_scores.append(0.0)
                processing_times.append(0)
        
        # Calculate metrics
        true_tags = [note[2] for note in self.test_data]
        metrics = calculate_precision_recall(
            true_tags=true_tags,
            predicted_tags=predicted_tags,
            confidence_scores=confidence_scores,
            processing_times=processing_times,
            available_tags=self.available_tags
        )
        
        # Estimate memory usage and model size
        memory_usage = self._estimate_memory_usage(embedder)
        model_size = self._estimate_model_size(model_name)
        
        total_time = time.time() - start_time
        
        return ModelComparisonResult(
            model_name=model_name,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            processing_time_ms=metrics.processing_time_ms,
            memory_usage_mb=memory_usage,
            embedding_dimension=embedder.get_embedding_dimension(),
            model_size_mb=model_size
        )
    
    def _estimate_memory_usage(self, embedder: TextEmbedder) -> float:
        """Estimate memory usage of the model in MB."""
        # Rough estimation based on embedding dimension
        dim = embedder.get_embedding_dimension()
        # Assume 4 bytes per float32, plus some overhead
        estimated_mb = (dim * 4 * 1000) / (1024 * 1024)  # Rough estimate for 1000 tokens
        return min(estimated_mb, 1000.0)  # Cap at 1GB
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in MB."""
        # Rough estimates based on common model sizes
        size_map = {
            "all-MiniLM-L6-v2": 90.0,
            "all-mpnet-base-v2": 420.0,
            "all-MiniLM-L12-v2": 120.0,
            "paraphrase-multilingual-MiniLM-L12-v2": 140.0,
            "multi-qa-MiniLM-L6-cos-v1": 90.0
        }
        return size_map.get(model_name, 200.0)  # Default estimate
    
    def select_best_model(self, results: List[ModelComparisonResult], 
                         criteria: str = "f1_score") -> ModelComparisonResult:
        """
        Select the best model based on specified criteria.
        
        Args:
            results: List of model comparison results
            criteria: Selection criteria ("f1_score", "precision", "recall", "speed")
            
        Returns:
            Best model according to criteria
        """
        if not results:
            raise ValueError("No valid results to select from")
        
        if criteria == "f1_score":
            return max(results, key=lambda x: x.f1_score)
        elif criteria == "precision":
            return max(results, key=lambda x: x.precision)
        elif criteria == "recall":
            return max(results, key=lambda x: x.recall)
        elif criteria == "speed":
            return min(results, key=lambda x: x.processing_time_ms)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
    
    def generate_comparison_report(self, results: List[ModelComparisonResult], 
                                 output_file: str) -> None:
        """Generate a comprehensive model comparison report."""
        # Sort by F1 score
        sorted_results = sorted(results, key=lambda x: x.f1_score, reverse=True)
        
        report = {
            "summary": {
                "total_models_evaluated": len(results),
                "best_model": sorted_results[0].model_name if sorted_results else None,
                "best_f1_score": sorted_results[0].f1_score if sorted_results else 0.0
            },
            "model_comparisons": [result.to_dict() for result in sorted_results],
            "recommendations": self._generate_model_recommendations(sorted_results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_model_recommendations(self, results: List[ModelComparisonResult]) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        if not results:
            return ["No models evaluated successfully"]
        
        best_model = results[0]
        
        # Performance recommendations
        if best_model.f1_score < 0.7:
            recommendations.append("All models performing below target (F1 < 0.7). Consider data quality or model selection.")
        
        # Speed recommendations
        fast_models = [r for r in results if r.processing_time_ms < 50]
        if not fast_models:
            recommendations.append("No models meet speed target (< 50ms). Consider optimization or different models.")
        
        # Memory recommendations
        memory_efficient = [r for r in results if r.memory_usage_mb < 500]
        if not memory_efficient:
            recommendations.append("All models exceed memory target (< 500MB). Consider smaller models.")
        
        # Specific model recommendations
        if best_model.f1_score > 0.8:
            recommendations.append(f"Model '{best_model.model_name}' performs excellently. Consider for production.")
        
        # Trade-off analysis
        if len(results) > 1:
            second_best = results[1]
            if (best_model.f1_score - second_best.f1_score < 0.05 and 
                second_best.processing_time_ms < best_model.processing_time_ms * 0.7):
                recommendations.append(f"Consider '{second_best.model_name}' for better speed/performance trade-off.")
        
        return recommendations


def benchmark_embedding_models(
    test_data_file: str,
    model_names: List[str],
    output_dir: str = "results/"
) -> None:
    """
    Benchmark multiple embedding models and generate reports.
    
    Args:
        test_data_file: Path to test data file (CSV with note_id, text, true_tag columns)
        model_names: List of model names to benchmark
        output_dir: Directory to save results
    """
    from ..ingestion.load_notes import NoteLoader
    import pandas as pd
    
    # Load test data
    df = pd.read_csv(test_data_file)
    test_data = [(str(row['note_id']), str(row['text']), str(row['true_tag'])) 
                 for _, row in df.iterrows()]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize validator
    validator = ModelValidator(test_data)
    
    # Compare models
    results = validator.compare_models(model_names)
    
    # Generate reports
    validator.generate_comparison_report(
        results, 
        f"{output_dir}/model_comparison.json"
    )
    
    # Select best model
    best_model = validator.select_best_model(results)
    
    print(f"Best model: {best_model.model_name}")
    print(f"F1 Score: {best_model.f1_score:.3f}")
    print(f"Processing time: {best_model.processing_time_ms:.1f}ms")
    
    # Save best model recommendation
    with open(f"{output_dir}/best_model.json", 'w') as f:
        json.dump(best_model.to_dict(), f, indent=2) 