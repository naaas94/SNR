"""Data validation for SNR system inputs."""

import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from .logger import MLLogger


@dataclass
class ValidationError:
    """Container for validation error information."""
    field: str
    error_type: str
    message: str
    value: Any
    severity: str = "error"  # error, warning, info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
            "value": str(self.value),
            "severity": self.severity
        }


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    data_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "data_quality_score": self.data_quality_score
        }


class DataValidator:
    """Validates input data for SNR system."""
    
    def __init__(self, logger: Optional[MLLogger] = None):
        """
        Initialize data validator.
        
        Args:
            logger: ML logger instance
        """
        self.logger = logger or MLLogger()
        
        # Validation rules
        self.note_validation_rules = {
            "text": {
                "required": True,
                "min_length": 3,
                "max_length": 10000,
                "pattern": None  # No specific pattern required
            },
            "note_id": {
                "required": True,
                "pattern": r"^[a-zA-Z0-9_-]+$",
                "max_length": 100
            },
            "timestamp": {
                "required": True,
                "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$"
            }
        }
        
        self.tag_validation_rules = {
            "tag": {
                "required": True,
                "pattern": r"^[a-z_][a-z0-9_]*$",
                "max_length": 50
            },
            "description": {
                "required": True,
                "min_length": 10,
                "max_length": 1000
            },
            "examples": {
                "required": True,
                "min_count": 1,
                "max_count": 20,
                "min_length": 5,
                "max_length": 500
            },
            "confidence_threshold": {
                "required": True,
                "min_value": 0.0,
                "max_value": 1.0
            }
        }
    
    def validate_note(self, note_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single note.
        
        Args:
            note_data: Note data dictionary
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Validate required fields
        for field, rules in self.note_validation_rules.items():
            if rules.get("required", False) and field not in note_data:
                errors.append(ValidationError(
                    field=field,
                    error_type="missing_field",
                    message=f"Required field '{field}' is missing",
                    value=None
                ))
                continue
            
            if field in note_data:
                value = note_data[field]
                field_errors, field_warnings = self._validate_field(
                    field, value, rules
                )
                errors.extend(field_errors)
                warnings.extend(field_warnings)
        
        # Additional note-specific validations
        if "text" in note_data:
            text_errors, text_warnings = self._validate_note_text(note_data["text"])
            errors.extend(text_errors)
            warnings.extend(text_warnings)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data_quality_score=quality_score
        )
    
    def validate_tag_definition(self, tag_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a tag definition.
        
        Args:
            tag_data: Tag definition dictionary
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Validate required fields
        for field, rules in self.tag_validation_rules.items():
            if rules.get("required", False) and field not in tag_data:
                errors.append(ValidationError(
                    field=field,
                    error_type="missing_field",
                    message=f"Required field '{field}' is missing",
                    value=None
                ))
                continue
            
            if field in tag_data:
                value = tag_data[field]
                field_errors, field_warnings = self._validate_field(
                    field, value, rules
                )
                errors.extend(field_errors)
                warnings.extend(field_warnings)
        
        # Additional tag-specific validations
        if "examples" in tag_data and isinstance(tag_data["examples"], list):
            example_errors, example_warnings = self._validate_tag_examples(
                tag_data["examples"]
            )
            errors.extend(example_errors)
            warnings.extend(example_warnings)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data_quality_score=quality_score
        )
    
    def validate_batch_notes(self, notes: List[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """
        Validate a batch of notes.
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Dictionary mapping note_id to ValidationResult
        """
        results = {}
        
        for i, note in enumerate(notes):
            note_id = note.get("note_id", f"note_{i}")
            result = self.validate_note(note)
            results[note_id] = result
            
            # Log validation results
            if not result.is_valid:
                self.logger.log_event(
                    event_type="validation",
                    component="validator",
                    message=f"Note {note_id} validation failed",
                    metadata={
                        "note_id": note_id,
                        "error_count": len(result.errors),
                        "warning_count": len(result.warnings),
                        "quality_score": result.data_quality_score
                    }
                )
        
        return results
    
    def _validate_field(self, field: str, value: Any, 
                       rules: Dict[str, Any]) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate a single field according to rules."""
        errors = []
        warnings = []
        
        # Type validation
        if "type" in rules:
            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                errors.append(ValidationError(
                    field=field,
                    error_type="type_error",
                    message=f"Expected type {expected_type}, got {type(value)}",
                    value=value
                ))
                return errors, warnings
        
        # Length validation for strings
        if isinstance(value, str):
            if "min_length" in rules and len(value) < rules["min_length"]:
                errors.append(ValidationError(
                    field=field,
                    error_type="length_error",
                    message=f"Minimum length is {rules['min_length']}, got {len(value)}",
                    value=value
                ))
            
            if "max_length" in rules and len(value) > rules["max_length"]:
                warnings.append(ValidationError(
                    field=field,
                    error_type="length_warning",
                    message=f"Maximum length is {rules['max_length']}, got {len(value)}",
                    value=value,
                    severity="warning"
                ))
            
            # Pattern validation
            if "pattern" in rules and rules["pattern"]:
                if not re.match(rules["pattern"], value):
                    errors.append(ValidationError(
                        field=field,
                        error_type="pattern_error",
                        message=f"Value does not match pattern {rules['pattern']}",
                        value=value
                    ))
        
        # Numeric validation
        if isinstance(value, (int, float)):
            if "min_value" in rules and value < rules["min_value"]:
                errors.append(ValidationError(
                    field=field,
                    error_type="value_error",
                    message=f"Minimum value is {rules['min_value']}, got {value}",
                    value=value
                ))
            
            if "max_value" in rules and value > rules["max_value"]:
                errors.append(ValidationError(
                    field=field,
                    error_type="value_error",
                    message=f"Maximum value is {rules['max_value']}, got {value}",
                    value=value
                ))
        
        # List validation
        if isinstance(value, list):
            if "min_count" in rules and len(value) < rules["min_count"]:
                errors.append(ValidationError(
                    field=field,
                    error_type="count_error",
                    message=f"Minimum count is {rules['min_count']}, got {len(value)}",
                    value=value
                ))
            
            if "max_count" in rules and len(value) > rules["max_count"]:
                warnings.append(ValidationError(
                    field=field,
                    error_type="count_warning",
                    message=f"Maximum count is {rules['max_count']}, got {len(value)}",
                    value=value,
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _validate_note_text(self, text: str) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate note text content."""
        errors = []
        warnings = []
        
        # Check for common issues
        if len(text.strip()) == 0:
            errors.append(ValidationError(
                field="text",
                error_type="content_error",
                message="Note text is empty or contains only whitespace",
                value=text
            ))
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (r"^\s*$", "Note contains only whitespace"),
            (r"^[^\w\s]*$", "Note contains only special characters"),
            (r"^(.)\1{10,}", "Note contains repeated characters"),
        ]
        
        for pattern, message in suspicious_patterns:
            if re.match(pattern, text):
                warnings.append(ValidationError(
                    field="text",
                    error_type="content_warning",
                    message=message,
                    value=text,
                    severity="warning"
                ))
        
        # Check for potential data quality issues
        if len(text) < 10:
            warnings.append(ValidationError(
                field="text",
                error_type="content_warning",
                message="Note is very short, may lack context",
                value=text,
                severity="warning"
            ))
        
        return errors, warnings
    
    def _validate_tag_examples(self, examples: List[str]) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate tag examples."""
        errors = []
        warnings = []
        
        # Check for duplicate examples
        unique_examples = set()
        for i, example in enumerate(examples):
            if example in unique_examples:
                warnings.append(ValidationError(
                    field=f"examples[{i}]",
                    error_type="duplicate_warning",
                    message="Duplicate example found",
                    value=example,
                    severity="warning"
                ))
            unique_examples.add(example)
        
        # Check example quality
        for i, example in enumerate(examples):
            if len(example.strip()) < 5:
                warnings.append(ValidationError(
                    field=f"examples[{i}]",
                    error_type="quality_warning",
                    message="Example is very short",
                    value=example,
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _calculate_quality_score(self, errors: List[ValidationError], 
                                warnings: List[ValidationError]) -> float:
        """Calculate data quality score (0.0 to 1.0)."""
        total_issues = len(errors) + len(warnings)
        
        if total_issues == 0:
            return 1.0
        
        # Weight errors more heavily than warnings
        weighted_issues = len(errors) * 2 + len(warnings)
        
        # Calculate score (higher is better)
        score = max(0.0, 1.0 - (weighted_issues / 10.0))
        
        return round(score, 3)
    
    def detect_outliers(self, notes: List[Dict[str, Any]], 
                       field: str = "text") -> List[Dict[str, Any]]:
        """
        Detect outliers in note data.
        
        Args:
            notes: List of notes
            field: Field to analyze for outliers
            
        Returns:
            List of outlier notes with analysis
        """
        if not notes:
            return []
        
        # Extract field values
        values = []
        for note in notes:
            if field in note:
                value = note[field]
                if isinstance(value, str):
                    values.append(len(value))
                elif isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            return []
        
        # Calculate statistics
        import numpy as np
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        # Define outlier threshold (2 standard deviations)
        threshold = 2 * std
        
        # Find outliers
        outliers = []
        for i, note in enumerate(notes):
            if field in note:
                value = note[field]
                if isinstance(value, str):
                    field_value = len(value)
                elif isinstance(value, (int, float)):
                    field_value = value
                else:
                    continue
                
                if abs(field_value - mean) > threshold:
                    outliers.append({
                        "note_id": note.get("note_id", f"note_{i}"),
                        "field": field,
                        "value": value,
                        "field_value": field_value,
                        "deviation": abs(field_value - mean),
                        "threshold": threshold
                    })
        
        return outliers
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult],
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Dictionary of validation results
            output_file: Optional file to save report
            
        Returns:
            Validation report dictionary
        """
        total_items = len(validation_results)
        valid_items = sum(1 for result in validation_results.values() if result.is_valid)
        invalid_items = total_items - valid_items
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        
        for item_id, result in validation_results.items():
            for error in result.errors:
                error_dict = error.to_dict()
                error_dict["item_id"] = item_id
                all_errors.append(error_dict)
            
            for warning in result.warnings:
                warning_dict = warning.to_dict()
                warning_dict["item_id"] = item_id
                all_warnings.append(warning_dict)
        
        # Calculate average quality score
        quality_scores = [result.data_quality_score for result in validation_results.values()]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Error type distribution
        error_types = {}
        for error in all_errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report = {
            "summary": {
                "total_items": total_items,
                "valid_items": valid_items,
                "invalid_items": invalid_items,
                "validation_rate": valid_items / total_items if total_items > 0 else 0.0,
                "average_quality_score": round(avg_quality_score, 3)
            },
            "errors": {
                "total_errors": len(all_errors),
                "error_types": error_types,
                "error_details": all_errors
            },
            "warnings": {
                "total_warnings": len(all_warnings),
                "warning_details": all_warnings
            },
            "recommendations": self._generate_validation_recommendations(
                total_items, invalid_items, all_errors, all_warnings
            )
        }
        
        # Save report if file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_validation_recommendations(self, total_items: int, invalid_items: int,
                                           errors: List[Dict], warnings: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Overall quality recommendations
        if invalid_items / total_items > 0.1:
            recommendations.append("More than 10% of items failed validation. Review data quality.")
        
        if len(errors) > len(warnings) * 2:
            recommendations.append("High number of errors compared to warnings. Check data format.")
        
        # Specific error type recommendations
        error_types = {}
        for error in errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            if count > total_items * 0.05:  # More than 5% of items
                recommendations.append(f"High frequency of {error_type} errors. Review validation rules.")
        
        return recommendations 