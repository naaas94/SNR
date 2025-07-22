"""Structured logging for ML operations."""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import datetime


@dataclass
class MLLogEvent:
    """Structured log event for ML operations."""
    timestamp: str
    event_type: str
    component: str
    message: str
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[Dict[str, Any]] = None


class MLLogger:
    """Structured logger for ML operations."""
    
    def __init__(self, log_file: Optional[str] = None, 
                 log_level: str = "INFO",
                 enable_console: bool = True):
        """
        Initialize ML logger.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level
            enable_console: Whether to log to console
        """
        self.logger = logging.getLogger("snr_ml")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
    
    def log_event(self, event_type: str, component: str, message: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  performance_metrics: Optional[Dict[str, float]] = None,
                  error_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured ML event.
        
        Args:
            event_type: Type of event (e.g., "model_prediction", "training", "error")
            component: Component name (e.g., "embedder", "router", "evaluator")
            message: Human-readable message
            metadata: Additional event metadata
            performance_metrics: Performance metrics for this event
            error_details: Error details if applicable
        """
        event = MLLogEvent(
            timestamp=datetime.datetime.now().isoformat(),
            event_type=event_type,
            component=component,
            message=message,
            metadata=metadata or {},
            performance_metrics=performance_metrics,
            error_details=error_details
        )
        
        # Log as JSON for structured logging
        log_entry = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "component": event.component,
            "message": event.message,
            "metadata": event.metadata
        }
        
        if event.performance_metrics:
            log_entry["performance_metrics"] = event.performance_metrics
        
        if event.error_details:
            log_entry["error_details"] = event.error_details
        
        # Choose log level based on event type
        if event_type == "error":
            self.logger.error(json.dumps(log_entry))
        elif event_type == "warning":
            self.logger.warning(json.dumps(log_entry))
        elif event_type == "performance":
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))
        
        # Track performance metrics
        if performance_metrics:
            for metric_name, value in performance_metrics.items():
                if metric_name not in self.performance_metrics:
                    self.performance_metrics[metric_name] = []
                self.performance_metrics[metric_name].append(value)
        
        # Track error counts
        if error_details:
            error_type = error_details.get("error_type", "unknown")
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def log_prediction(self, note_id: str, predicted_tag: str, 
                      confidence: float, processing_time_ms: float,
                      similarity_score: float, model_name: str) -> None:
        """Log a prediction event."""
        self.log_event(
            event_type="prediction",
            component="router",
            message=f"Predicted tag '{predicted_tag}' for note {note_id}",
            metadata={
                "note_id": note_id,
                "predicted_tag": predicted_tag,
                "model_name": model_name
            },
            performance_metrics={
                "confidence": confidence,
                "processing_time_ms": processing_time_ms,
                "similarity_score": similarity_score
            }
        )
    
    def log_training(self, model_name: str, training_time_seconds: float,
                    num_samples: int, metrics: Dict[str, float]) -> None:
        """Log a training event."""
        self.log_event(
            event_type="training",
            component="evaluator",
            message=f"Trained model {model_name} on {num_samples} samples",
            metadata={
                "model_name": model_name,
                "num_samples": num_samples
            },
            performance_metrics={
                "training_time_seconds": training_time_seconds,
                **metrics
            }
        )
    
    def log_error(self, component: str, error_message: str, 
                  error_type: str, stack_trace: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error event."""
        self.log_event(
            event_type="error",
            component=component,
            message=error_message,
            metadata=context or {},
            error_details={
                "error_type": error_type,
                "stack_trace": stack_trace
            }
        )
    
    def log_performance(self, component: str, operation: str,
                       duration_ms: float, memory_usage_mb: Optional[float] = None,
                       additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log a performance event."""
        metrics = {"duration_ms": duration_ms}
        if memory_usage_mb:
            metrics["memory_usage_mb"] = memory_usage_mb
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_event(
            event_type="performance",
            component=component,
            message=f"{operation} completed in {duration_ms:.2f}ms",
            metadata={"operation": operation},
            performance_metrics=metrics
        )
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of performance metrics."""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts by type."""
        return self.error_counts.copy()
    
    def export_logs(self, output_file: str, 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None) -> None:
        """Export logs to JSON file with optional time filtering."""
        # This would require reading from the log file and filtering
        # Implementation depends on how logs are stored
        pass


def setup_logging(log_file: Optional[str] = None,
                 log_level: str = "INFO",
                 enable_console: bool = True) -> MLLogger:
    """
    Setup and return a configured ML logger.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        enable_console: Whether to log to console
        
    Returns:
        Configured MLLogger instance
    """
    return MLLogger(
        log_file=log_file,
        log_level=log_level,
        enable_console=enable_console
    )


# Global logger instance
_global_logger: Optional[MLLogger] = None


def get_logger() -> MLLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger


def set_logger(logger: MLLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger 


def setup_api_logging(api_name: str, log_file: Optional[str] = None, log_level: str = "INFO") -> MLLogger:
    """
    Set up logging for API interfaces.
    
    Args:
        api_name: Name of the API
        log_file: Path to log file (optional)
        log_level: Logging level
        
    Returns:
        Configured MLLogger instance
    """
    logger = MLLogger(log_file=log_file, log_level=log_level)
    logger.log_event(
        event_type="startup",
        component=api_name,
        message=f"API {api_name} started",
        metadata={"log_level": log_level}
    )
    return logger


def setup_cli_logging(cli_name: str, log_file: Optional[str] = None, log_level: str = "INFO") -> MLLogger:
    """
    Set up logging for CLI interfaces.
    
    Args:
        cli_name: Name of the CLI
        log_file: Path to log file (optional)
        log_level: Logging level
        
    Returns:
        Configured MLLogger instance
    """
    logger = MLLogger(log_file=log_file, log_level=log_level)
    logger.log_event(
        event_type="startup",
        component=cli_name,
        message=f"CLI {cli_name} started",
        metadata={"log_level": log_level}
    )
    return logger 


class CircuitBreaker:
    """Circuit breaker pattern to manage failures and prevent system overload."""
    
    def __init__(self, failure_threshold: int, recovery_timeout: int, logger: Optional[MLLogger] = None):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger the circuit breaker
            recovery_timeout: Time in seconds to wait before attempting recovery
            logger: ML logger instance
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.logger = logger or MLLogger()
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                self.logger.log_event(
                    event_type="circuit_breaker",
                    component="circuit_breaker",
                    message="Circuit breaker is open, request blocked",
                    metadata={"state": self.state}
                )
                raise Exception("Circuit breaker is open")
            else:
                self.state = "HALF_OPEN"
                self.logger.log_event(
                    event_type="circuit_breaker",
                    component="circuit_breaker",
                    message="Circuit breaker transitioning to half-open",
                    metadata={"state": self.state}
                )
        
        try:
            result = func(*args, **kwargs)
            self._reset()
            return result
        except Exception as e:
            self._record_failure()
            self.logger.log_error(
                component="circuit_breaker",
                error_message=str(e),
                error_type="execution_failure"
            )
            raise
    
    def _record_failure(self) -> None:
        """Record a failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.log_event(
                event_type="circuit_breaker",
                component="circuit_breaker",
                message="Circuit breaker opened due to failures",
                metadata={"state": self.state, "failure_count": self.failure_count}
            )
    
    def _reset(self) -> None:
        """Reset the circuit breaker after a successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.logger.log_event(
            event_type="circuit_breaker",
            component="circuit_breaker",
            message="Circuit breaker closed after successful call",
            metadata={"state": self.state}
        ) 