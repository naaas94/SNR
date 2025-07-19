"""Evaluation metrics for semantic routing performance."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import json


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray
    per_tag_metrics: Dict[str, Dict[str, float]]
    confidence_correlation: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_tag_metrics": self.per_tag_metrics,
            "confidence_correlation": self.confidence_correlation,
            "processing_time_ms": self.processing_time_ms
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def calculate_precision_recall(
    true_tags: List[str],
    predicted_tags: List[str],
    confidence_scores: List[float],
    processing_times: List[float],
    available_tags: List[str]
) -> EvaluationMetrics:
    """
    Calculate comprehensive evaluation metrics for semantic routing.
    
    Args:
        true_tags: Ground truth tag labels
        predicted_tags: Predicted tag labels
        confidence_scores: Confidence scores for predictions
        processing_times: Processing time for each prediction (ms)
        available_tags: List of all available tags
        
    Returns:
        EvaluationMetrics object with all computed metrics
    """
    # Convert tags to indices for sklearn metrics
    tag_to_idx = {tag: idx for idx, tag in enumerate(available_tags)}
    
    # Convert to numerical labels
    y_true = [tag_to_idx.get(tag, -1) for tag in true_tags]
    y_pred = [tag_to_idx.get(tag, -1) for tag in predicted_tags]
    
    # Filter out invalid predictions
    valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                    if true != -1 and pred != -1]
    
    if not valid_indices:
        raise ValueError("No valid predictions found for evaluation")
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    confidences_valid = [confidence_scores[i] for i in valid_indices]
    
    # Calculate overall metrics
    precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    accuracy = np.mean(np.array(y_true_valid) == np.array(y_pred_valid))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(available_tags)))
    
    # Per-tag metrics
    per_tag_metrics = {}
    for i, tag in enumerate(available_tags):
        if i < len(available_tags):
            tag_precision = precision_score(y_true_valid, y_pred_valid, 
                                          labels=[i], average='binary', zero_division=0)
            tag_recall = recall_score(y_true_valid, y_pred_valid, 
                                    labels=[i], average='binary', zero_division=0)
            tag_f1 = f1_score(y_true_valid, y_pred_valid, 
                             labels=[i], average='binary', zero_division=0)
            
            per_tag_metrics[tag] = {
                "precision": tag_precision,
                "recall": tag_recall,
                "f1_score": tag_f1,
                "support": int(np.sum(np.array(y_true_valid) == i))
            }
    
    # Confidence correlation (how well confidence predicts accuracy)
    correct_predictions = np.array(y_true_valid) == np.array(y_pred_valid)
    confidence_correlation = np.corrcoef(confidences_valid, correct_predictions.astype(float))[0, 1]
    if np.isnan(confidence_correlation):
        confidence_correlation = 0.0
    
    # Average processing time
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        confusion_matrix=cm,
        per_tag_metrics=per_tag_metrics,
        confidence_correlation=confidence_correlation,
        processing_time_ms=avg_processing_time
    )


def cross_validate_routing(
    router_func,
    notes: List[Tuple[str, str, str]],  # (note_id, text, true_tag)
    n_splits: int = 5,
    random_state: int = 42
) -> List[EvaluationMetrics]:
    """
    Perform cross-validation on routing performance.
    
    Args:
        router_func: Function that takes notes and returns predictions
        notes: List of (note_id, text, true_tag) tuples
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of EvaluationMetrics for each fold
    """
    # Extract features and labels
    note_ids = [note[0] for note in notes]
    texts = [note[1] for note in notes]
    true_tags = [note[2] for note in notes]
    
    # Get unique tags for label encoding
    unique_tags = list(set(true_tags))
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, true_tags)):
        print(f"Processing fold {fold_idx + 1}/{n_splits}")
        
        # Split data
        train_notes = [(note_ids[i], texts[i], true_tags[i]) for i in train_idx]
        test_notes = [(note_ids[i], texts[i], true_tags[i]) for i in test_idx]
        
        # Train router (if needed) and make predictions
        try:
            # This assumes router_func can handle training and prediction
            # Implementation depends on specific router interface
            predicted_tags = []
            confidence_scores = []
            processing_times = []
            
            for note_id, text, _ in test_notes:
                # This is a placeholder - actual implementation depends on router interface
                # result = router_func.predict_single(note_id, text)
                # predicted_tags.append(result.routed_tag)
                # confidence_scores.append(result.confidence)
                # processing_times.append(result.metadata.get("processing_time_ms", 0))
                pass
            
            # Calculate metrics for this fold
            fold_metric = calculate_precision_recall(
                true_tags=[true_tags[i] for i in test_idx],
                predicted_tags=predicted_tags,
                confidence_scores=confidence_scores,
                processing_times=processing_times,
                available_tags=unique_tags
            )
            
            fold_metrics.append(fold_metric)
            
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {e}")
            continue
    
    return fold_metrics


def generate_evaluation_report(metrics: EvaluationMetrics, output_file: str) -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        metrics: EvaluationMetrics object
        output_file: Path to save the report
    """
    report = {
        "summary": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "accuracy": metrics.accuracy,
            "confidence_correlation": metrics.confidence_correlation,
            "avg_processing_time_ms": metrics.processing_time_ms
        },
        "per_tag_performance": metrics.per_tag_metrics,
        "recommendations": _generate_recommendations(metrics)
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


def _generate_recommendations(metrics: EvaluationMetrics) -> List[str]:
    """Generate recommendations based on evaluation metrics."""
    recommendations = []
    
    if metrics.precision < 0.8:
        recommendations.append("Precision below target (0.8). Consider adjusting confidence thresholds.")
    
    if metrics.recall < 0.7:
        recommendations.append("Recall below target (0.7). Consider adding more tag examples.")
    
    if metrics.confidence_correlation < 0.3:
        recommendations.append("Low confidence correlation. Implement confidence calibration.")
    
    if metrics.processing_time_ms > 100:
        recommendations.append("Processing time above target (100ms). Consider optimization.")
    
    # Find worst performing tags
    worst_tags = sorted(
        metrics.per_tag_metrics.items(),
        key=lambda x: x[1]['f1_score']
    )[:3]
    
    for tag, tag_metrics in worst_tags:
        if tag_metrics['f1_score'] < 0.5:
            recommendations.append(f"Tag '{tag}' performing poorly (F1={tag_metrics['f1_score']:.2f}). Review examples.")
    
    return recommendations 