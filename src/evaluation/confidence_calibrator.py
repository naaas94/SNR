"""Confidence calibration for semantic routing predictions."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, calibration_curve
import json


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    original_scores: List[float]
    calibrated_scores: List[float]
    calibration_parameters: Dict[str, float]
    brier_score_before: float
    brier_score_after: float
    reliability_diagram: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_scores": self.original_scores,
            "calibrated_scores": self.calibrated_scores,
            "calibration_parameters": self.calibration_parameters,
            "brier_score_before": self.brier_score_before,
            "brier_score_after": self.brier_score_after,
            "reliability_diagram": self.reliability_diagram
        }


class ConfidenceCalibrator:
    """Calibrates confidence scores using Platt scaling."""
    
    def __init__(self, method: str = "platt"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ("platt", "isotonic", "temperature")
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, similarity_scores: List[float], 
            true_labels: List[bool], 
            validation_split: float = 0.2) -> CalibrationResult:
        """
        Fit the calibration model.
        
        Args:
            similarity_scores: Raw similarity scores from model
            true_labels: True binary labels (1 for correct, 0 for incorrect)
            validation_split: Fraction of data to use for validation
            
        Returns:
            CalibrationResult with calibration metrics
        """
        if len(similarity_scores) != len(true_labels):
            raise ValueError("Length of scores and labels must match")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            np.array(similarity_scores).reshape(-1, 1),
            np.array(true_labels),
            test_size=validation_split,
            random_state=42,
            stratify=true_labels
        )
        
        # Fit calibrator
        if self.method == "platt":
            self.calibrator = LogisticRegression(random_state=42)
            self.calibrator.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        
        self.is_fitted = True
        
        # Calculate calibration metrics
        original_scores = similarity_scores
        calibrated_scores = self.predict_proba(similarity_scores)
        
        # Brier scores
        brier_before = brier_score_loss(true_labels, original_scores)
        brier_after = brier_score_loss(true_labels, calibrated_scores)
        
        # Reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, calibrated_scores, n_bins=10
        )
        
        reliability_diagram = {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist()
        }
        
        # Calibration parameters
        if self.method == "platt":
            calibration_params = {
                "intercept": float(self.calibrator.intercept_[0]),
                "coefficient": float(self.calibrator.coef_[0][0])
            }
        else:
            calibration_params = {}
        
        return CalibrationResult(
            original_scores=original_scores,
            calibrated_scores=calibrated_scores,
            calibration_parameters=calibration_params,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            reliability_diagram=reliability_diagram
        )
    
    def predict_proba(self, similarity_scores: List[float]) -> List[float]:
        """
        Predict calibrated probabilities.
        
        Args:
            similarity_scores: Raw similarity scores
            
        Returns:
            Calibrated probability scores
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before prediction")
        
        X = np.array(similarity_scores).reshape(-1, 1)
        calibrated_probs = self.calibrator.predict_proba(X)[:, 1]
        
        return calibrated_probs.tolist()
    
    def calibrate_confidence(self, similarity_score: float, 
                           tag_threshold: float = 0.5) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            similarity_score: Raw similarity score
            tag_threshold: Minimum threshold for the tag
            
        Returns:
            Calibrated confidence score
        """
        if not self.is_fitted:
            # Fallback to simple calibration if not fitted
            return self._simple_calibration(similarity_score, tag_threshold)
        
        calibrated_prob = self.predict_proba([similarity_score])[0]
        
        # Apply tag-specific threshold adjustment
        if similarity_score < tag_threshold:
            # Below threshold: reduce confidence
            calibrated_prob *= (similarity_score / tag_threshold)
        
        return max(0.0, min(1.0, calibrated_prob))
    
    def _simple_calibration(self, similarity_score: float, 
                           tag_threshold: float) -> float:
        """Simple calibration fallback when no training data is available."""
        if similarity_score >= tag_threshold:
            # Above threshold: confidence increases with similarity
            confidence = 0.5 + 0.5 * (similarity_score - tag_threshold) / (1.0 - tag_threshold)
        else:
            # Below threshold: confidence decreases
            confidence = 0.5 * (similarity_score / tag_threshold)
            
        return max(0.0, min(1.0, confidence))
    
    def save_calibrator(self, filepath: str) -> None:
        """Save the fitted calibrator to file."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before saving")
        
        calibrator_data = {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "parameters": {}
        }
        
        if self.method == "platt":
            calibrator_data["parameters"] = {
                "intercept": float(self.calibrator.intercept_[0]),
                "coefficient": float(self.calibrator.coef_[0][0])
            }
        
        with open(filepath, 'w') as f:
            json.dump(calibrator_data, f, indent=2)
    
    def load_calibrator(self, filepath: str) -> None:
        """Load a fitted calibrator from file."""
        with open(filepath, 'r') as f:
            calibrator_data = json.load(f)
        
        self.method = calibrator_data["method"]
        self.is_fitted = calibrator_data["is_fitted"]
        
        if self.method == "platt" and self.is_fitted:
            self.calibrator = LogisticRegression(random_state=42)
            intercept = calibrator_data["parameters"]["intercept"]
            coefficient = calibrator_data["parameters"]["coefficient"]
            
            # Reconstruct the fitted model
            self.calibrator.intercept_ = np.array([intercept])
            self.calibrator.coef_ = np.array([[coefficient]])
            self.calibrator.classes_ = np.array([0, 1])


def create_calibration_dataset(
    routing_results: List[Dict[str, Any]]
) -> Tuple[List[float], List[bool]]:
    """
    Create calibration dataset from routing results.
    
    Args:
        routing_results: List of routing result dictionaries
        
    Returns:
        Tuple of (similarity_scores, true_labels)
    """
    similarity_scores = []
    true_labels = []
    
    for result in routing_results:
        # Extract similarity score
        similarity_score = result.get("similarity_score", 0.0)
        similarity_scores.append(similarity_score)
        
        # Determine if prediction was correct (requires ground truth)
        # This is a placeholder - actual implementation depends on data structure
        predicted_tag = result.get("routed_tag", "")
        true_tag = result.get("true_tag", "")  # Assuming this exists
        
        is_correct = predicted_tag == true_tag
        true_labels.append(is_correct)
    
    return similarity_scores, true_labels


def evaluate_calibration_quality(
    true_labels: List[bool],
    confidence_scores: List[float]
) -> Dict[str, float]:
    """
    Evaluate the quality of confidence calibration.
    
    Args:
        true_labels: True binary labels
        confidence_scores: Confidence scores
        
    Returns:
        Dictionary of calibration quality metrics
    """
    # Brier score (lower is better)
    brier_score = brier_score_loss(true_labels, confidence_scores)
    
    # Expected calibration error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(confidence_scores > bin_lower, 
                               confidence_scores <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Calculate accuracy and confidence in this bin
            bin_accuracy = np.mean(np.array(true_labels)[in_bin])
            bin_confidence = np.mean(np.array(confidence_scores)[in_bin])
            
            # Add to ECE
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    ece /= len(true_labels)
    
    # Reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, confidence_scores, n_bins=10
    )
    
    return {
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "reliability_diagram": {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist()
        }
    } 