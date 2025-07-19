"""Performance monitoring and error tracking for SNR system."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

from .logger import MLLogger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    processing_time_ms: float
    throughput_notes_per_second: float
    error_rate: float
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "processing_time_ms": self.processing_time_ms,
            "throughput_notes_per_second": self.throughput_notes_per_second,
            "error_rate": self.error_rate,
            "active_connections": self.active_connections
        }


@dataclass
class ErrorEvent:
    """Container for error event information."""
    timestamp: float
    error_type: str
    component: str
    message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "component": self.component,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "context": self.context
        }


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, logger: Optional[MLLogger] = None,
                 metrics_file: Optional[str] = None,
                 max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            logger: ML logger instance
            metrics_file: File to save metrics (optional)
            max_history: Maximum number of metrics to keep in memory
        """
        self.logger = logger or MLLogger()
        self.metrics_file = metrics_file
        self.max_history = max_history
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.current_metrics: Dict[str, float] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.total_notes_processed = 0
        self.total_errors = 0
        self.processing_times: List[float] = []
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
    
    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.log_event(
            event_type="monitoring",
            component="monitor",
            message="Started performance monitoring",
            metadata={"interval_seconds": interval_seconds}
        )
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.log_event(
            event_type="monitoring",
            component="monitor",
            message="Stopped performance monitoring"
        )
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Background monitoring loop."""
        while not self.stop_monitoring:
            try:
                self.collect_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.log_error(
                    component="monitor",
                    error_message=f"Error in monitoring loop: {e}",
                    error_type="monitoring_error"
                )
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # Calculate processing metrics
        avg_processing_time = (
            sum(self.processing_times[-100:]) / len(self.processing_times[-100:])
            if self.processing_times else 0.0
        )
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        throughput = self.total_notes_processed / elapsed_time if elapsed_time > 0 else 0.0
        
        # Calculate error rate
        error_rate = (
            self.total_errors / self.total_notes_processed
            if self.total_notes_processed > 0 else 0.0
        )
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            processing_time_ms=avg_processing_time,
            throughput_notes_per_second=throughput,
            error_rate=error_rate
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.current_metrics = metrics.to_dict()
        
        # Log metrics
        self.logger.log_performance(
            component="system",
            operation="monitoring",
            duration_ms=avg_processing_time,
            memory_usage_mb=memory_mb,
            additional_metrics={
                "cpu_percent": cpu_percent,
                "throughput_notes_per_second": throughput,
                "error_rate": error_rate
            }
        )
        
        # Save to file if specified
        if self.metrics_file:
            self._save_metrics_to_file()
        
        return metrics
    
    def record_processing_time(self, processing_time_ms: float) -> None:
        """Record processing time for a note."""
        self.processing_times.append(processing_time_ms)
        self.total_notes_processed += 1
        
        # Keep only recent processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.total_errors += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_mb for m in recent_metrics]
        processing_times = [m.processing_time_ms for m in recent_metrics]
        throughput_values = [m.throughput_notes_per_second for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        
        return {
            "cpu_percent": {
                "mean": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_mb": {
                "mean": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "processing_time_ms": {
                "mean": sum(processing_times) / len(processing_times),
                "max": max(processing_times),
                "min": min(processing_times)
            },
            "throughput_notes_per_second": {
                "mean": sum(throughput_values) / len(throughput_values),
                "max": max(throughput_values),
                "min": min(throughput_values)
            },
            "error_rate": {
                "mean": sum(error_rates) / len(error_rates),
                "max": max(error_rates),
                "min": min(error_rates)
            },
            "total_notes_processed": self.total_notes_processed,
            "total_errors": self.total_errors
        }
    
    def _save_metrics_to_file(self) -> None:
        """Save current metrics to file."""
        if not self.metrics_file:
            return
        
        try:
            metrics_data = {
                "timestamp": time.time(),
                "metrics": self.current_metrics,
                "summary": self.get_performance_summary()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            self.logger.log_error(
                component="monitor",
                error_message=f"Failed to save metrics: {e}",
                error_type="file_error"
            )
    
    def export_metrics(self, output_file: str) -> None:
        """Export all collected metrics to file."""
        try:
            metrics_data = {
                "summary": self.get_performance_summary(),
                "history": [m.to_dict() for m in self.metrics_history]
            }
            
            with open(output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            self.logger.log_error(
                component="monitor",
                error_message=f"Failed to export metrics: {e}",
                error_type="file_error"
            )


class ErrorTracker:
    """Tracks and analyzes errors."""
    
    def __init__(self, logger: Optional[MLLogger] = None,
                 max_errors: int = 1000):
        """
        Initialize error tracker.
        
        Args:
            logger: ML logger instance
            max_errors: Maximum number of errors to keep in memory
        """
        self.logger = logger or MLLogger()
        self.max_errors = max_errors
        
        # Error storage
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.component_error_counts: Dict[str, int] = defaultdict(int)
        
        # Error rate tracking
        self.error_rate_window = 3600  # 1 hour window
        self.error_timestamps: List[float] = []
    
    def record_error(self, error_type: str, component: str, message: str,
                    stack_trace: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error event.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
            message: Error message
            stack_trace: Stack trace (optional)
            context: Additional context (optional)
        """
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type=error_type,
            component=component,
            message=message,
            stack_trace=stack_trace,
            context=context or {}
        )
        
        # Store error
        self.errors.append(error_event)
        
        # Update counts
        self.error_counts[error_type] += 1
        self.component_error_counts[component] += 1
        
        # Update error rate tracking
        self.error_timestamps.append(error_event.timestamp)
        
        # Remove old timestamps
        current_time = time.time()
        self.error_timestamps = [
            ts for ts in self.error_timestamps
            if current_time - ts <= self.error_rate_window
        ]
        
        # Log error
        self.logger.log_error(
            component=component,
            error_message=message,
            error_type=error_type,
            stack_trace=stack_trace,
            context=context
        )
    
    def get_error_rate(self, window_seconds: Optional[int] = None) -> float:
        """
        Calculate error rate over specified window.
        
        Args:
            window_seconds: Time window in seconds (default: 1 hour)
            
        Returns:
            Error rate (errors per second)
        """
        if window_seconds is None:
            window_seconds = self.error_rate_window
        
        current_time = time.time()
        recent_errors = [
            ts for ts in self.error_timestamps
            if current_time - ts <= window_seconds
        ]
        
        return len(recent_errors) / window_seconds if window_seconds > 0 else 0.0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics."""
        return {
            "total_errors": len(self.errors),
            "error_counts_by_type": dict(self.error_counts),
            "error_counts_by_component": dict(self.component_error_counts),
            "current_error_rate": self.get_error_rate(),
            "recent_errors": [
                error.to_dict() for error in list(self.errors)[-10:]  # Last 10 errors
            ]
        }
    
    def get_most_common_errors(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most common error types."""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:top_n]
        ]
    
    def get_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for each component."""
        health_status = {}
        
        for component, error_count in self.component_error_counts.items():
            # Calculate error rate for this component
            component_errors = [
                error for error in self.errors
                if error.component == component
            ]
            
            recent_errors = [
                error for error in component_errors
                if time.time() - error.timestamp <= 3600  # Last hour
            ]
            
            health_status[component] = {
                "total_errors": error_count,
                "recent_errors": len(recent_errors),
                "error_rate_per_hour": len(recent_errors),
                "status": "healthy" if len(recent_errors) == 0 else "warning" if len(recent_errors) < 5 else "critical"
            }
        
        return health_status
    
    def export_errors(self, output_file: str) -> None:
        """Export all error data to file."""
        try:
            error_data = {
                "summary": self.get_error_summary(),
                "component_health": self.get_component_health(),
                "most_common_errors": self.get_most_common_errors(),
                "all_errors": [error.to_dict() for error in self.errors]
            }
            
            with open(output_file, 'w') as f:
                json.dump(error_data, f, indent=2)
        except Exception as e:
            self.logger.log_error(
                component="error_tracker",
                error_message=f"Failed to export errors: {e}",
                error_type="file_error"
            ) 