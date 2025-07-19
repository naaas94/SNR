#!/usr/bin/env python3
"""
Comprehensive evaluation script for SNR system.

This script implements the critical ML Engineer recommendations:
1. Evaluation Framework with precision/recall metrics
2. Model validation and comparison
3. Confidence calibration
4. Performance monitoring
5. Data validation
6. Test suite execution
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import calculate_precision_recall, generate_evaluation_report
from evaluation.model_validator import ModelValidator, benchmark_embedding_models
from evaluation.confidence_calibrator import ConfidenceCalibrator, evaluate_calibration_quality
from utils.logger import setup_logging, MLLogger
from utils.monitoring import PerformanceMonitor, ErrorTracker
from utils.data_validation import DataValidator
from embedding.embedder import TextEmbedder, TagEmbedder
from routing.route_with_faiss import FAISSRouter
from config.tag_loader import TagLoader
from ingestion.load_notes import NoteLoader


def setup_evaluation_environment(output_dir: str) -> Dict[str, Any]:
    """Setup evaluation environment and components."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_file=str(output_path / "evaluation.log"),
        log_level="INFO",
        enable_console=True
    )
    
    # Setup monitoring
    performance_monitor = PerformanceMonitor(
        logger=logger,
        metrics_file=str(output_path / "performance_metrics.json")
    )
    
    error_tracker = ErrorTracker(logger=logger)
    
    # Setup data validator
    data_validator = DataValidator(logger=logger)
    
    logger.log_event(
        event_type="evaluation",
        component="evaluator",
        message="Evaluation environment setup complete",
        metadata={"output_dir": str(output_path)}
    )
    
    return {
        "logger": logger,
        "performance_monitor": performance_monitor,
        "error_tracker": error_tracker,
        "data_validator": data_validator,
        "output_path": output_path
    }


def validate_input_data(notes_file: str, tags_file: str, 
                       data_validator: DataValidator,
                       logger: MLLogger) -> Dict[str, Any]:
    """Validate input data quality."""
    logger.log_event(
        event_type="validation",
        component="evaluator",
        message="Starting input data validation"
    )
    
    # Load and validate notes
    note_loader = NoteLoader()
    notes = note_loader.load_from_csv(notes_file)
    
    # Convert to validation format
    notes_data = []
    for note in notes:
        notes_data.append({
            "note_id": note.note_id,
            "text": note.text,
            "timestamp": note.timestamp
        })
    
    # Validate notes
    note_validation_results = data_validator.validate_batch_notes(notes_data)
    
    # Load and validate tags
    tag_loader = TagLoader(tags_file)
    tags = tag_loader.load_tags()
    
    tag_validation_results = {}
    for tag in tags:
        tag_data = {
            "tag": tag.tag,
            "description": tag.description,
            "examples": tag.examples,
            "confidence_threshold": tag.confidence_threshold
        }
        tag_validation_results[tag.tag] = data_validator.validate_tag_definition(tag_data)
    
    # Generate validation reports
    note_report = data_validator.generate_validation_report(
        note_validation_results,
        output_file=str(Path("results") / "note_validation_report.json")
    )
    
    tag_report = data_validator.generate_validation_report(
        tag_validation_results,
        output_file=str(Path("results") / "tag_validation_report.json")
    )
    
    # Detect outliers
    outliers = data_validator.detect_outliers(notes_data, field="text")
    
    validation_summary = {
        "notes": {
            "total": len(notes),
            "valid": sum(1 for r in note_validation_results.values() if r.is_valid),
            "invalid": sum(1 for r in note_validation_results.values() if not r.is_valid),
            "avg_quality_score": sum(r.data_quality_score for r in note_validation_results.values()) / len(note_validation_results),
            "outliers": len(outliers)
        },
        "tags": {
            "total": len(tags),
            "valid": sum(1 for r in tag_validation_results.values() if r.is_valid),
            "invalid": sum(1 for r in tag_validation_results.values() if not r.is_valid),
            "avg_quality_score": sum(r.data_quality_score for r in tag_validation_results.values()) / len(tag_validation_results)
        }
    }
    
    logger.log_event(
        event_type="validation",
        component="evaluator",
        message="Input data validation complete",
        metadata=validation_summary
    )
    
    return {
        "notes": notes,
        "tags": tags,
        "note_validation_results": note_validation_results,
        "tag_validation_results": tag_validation_results,
        "validation_summary": validation_summary,
        "outliers": outliers
    }


def benchmark_models(notes: List, tags: List, 
                    model_names: List[str],
                    output_path: Path,
                    logger: MLLogger) -> Dict[str, Any]:
    """Benchmark multiple embedding models."""
    logger.log_event(
        event_type="benchmarking",
        component="evaluator",
        message="Starting model benchmarking",
        metadata={"models": model_names}
    )
    
    # Create test data for benchmarking
    # This would require ground truth labels - for now we'll use a synthetic approach
    test_data = []
    for i, note in enumerate(notes[:100]):  # Use first 100 notes for benchmarking
        # For demonstration, assign tags based on content similarity
        # In practice, this would come from human annotations
        test_data.append((
            note.note_id,
            note.text,
            tags[i % len(tags)].tag if tags else "unknown"
        ))
    
    # Benchmark models
    results = []
    for model_name in model_names:
        try:
            logger.log_event(
                event_type="benchmarking",
                component="evaluator",
                message=f"Evaluating model: {model_name}"
            )
            
            # Initialize model and router
            embedder = TextEmbedder(model_name)
            tag_embedder = TagEmbedder(embedder)
            router = FAISSRouter(embedder, tag_embedder)
            
            # Build index
            router.build_index(tags)
            
            # Make predictions
            predicted_tags = []
            confidence_scores = []
            processing_times = []
            
            for note_id, text, _ in test_data:
                try:
                    result = router.route_note(note_id, text, "2025-01-27T00:00:00")
                    predicted_tags.append(result.routed_tag)
                    confidence_scores.append(result.confidence)
                    processing_times.append(result.metadata.get("processing_time_ms", 0))
                except Exception as e:
                    logger.log_error(
                        component="benchmarking",
                        error_message=f"Error routing note {note_id}: {e}",
                        error_type="routing_error"
                    )
                    predicted_tags.append("unknown")
                    confidence_scores.append(0.0)
                    processing_times.append(0)
            
            # Calculate metrics
            true_tags = [note[2] for note in test_data]
            available_tags = [tag.tag for tag in tags]
            
            metrics = calculate_precision_recall(
                true_tags=true_tags,
                predicted_tags=predicted_tags,
                confidence_scores=confidence_scores,
                processing_times=processing_times,
                available_tags=available_tags
            )
            
            # Estimate memory usage
            memory_usage = embedder.get_embedding_dimension() * 4 * 1000 / (1024 * 1024)  # Rough estimate
            
            model_result = {
                "model_name": model_name,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "processing_time_ms": metrics.processing_time_ms,
                "memory_usage_mb": memory_usage,
                "embedding_dimension": embedder.get_embedding_dimension(),
                "confidence_correlation": metrics.confidence_correlation
            }
            
            results.append(model_result)
            
            logger.log_event(
                event_type="benchmarking",
                component="evaluator",
                message=f"Completed evaluation of {model_name}",
                metadata=model_result
            )
            
        except Exception as e:
            logger.log_error(
                component="benchmarking",
                error_message=f"Failed to evaluate model {model_name}: {e}",
                error_type="model_evaluation_error"
            )
    
    # Save benchmark results
    benchmark_report = {
        "summary": {
            "total_models_evaluated": len(results),
            "best_model": max(results, key=lambda x: x["f1_score"])["model_name"] if results else None,
            "best_f1_score": max(results, key=lambda x: x["f1_score"])["f1_score"] if results else 0.0
        },
        "model_comparisons": results
    }
    
    with open(output_path / "model_benchmark_results.json", 'w') as f:
        json.dump(benchmark_report, f, indent=2)
    
    logger.log_event(
        event_type="benchmarking",
        component="evaluator",
        message="Model benchmarking complete",
        metadata={"total_models": len(results)}
    )
    
    return benchmark_report


def evaluate_confidence_calibration(notes: List, tags: List,
                                  best_model_name: str,
                                  output_path: Path,
                                  logger: MLLogger) -> Dict[str, Any]:
    """Evaluate and calibrate confidence scores."""
    logger.log_event(
        event_type="calibration",
        component="evaluator",
        message="Starting confidence calibration evaluation"
    )
    
    # Initialize best model
    embedder = TextEmbedder(best_model_name)
    tag_embedder = TagEmbedder(embedder)
    router = FAISSRouter(embedder, tag_embedder)
    router.build_index(tags)
    
    # Create calibration dataset
    calibration_data = []
    for note in notes[:200]:  # Use first 200 notes for calibration
        try:
            result = router.route_note(note.note_id, note.text, note.timestamp)
            calibration_data.append({
                "note_id": note.note_id,
                "similarity_score": result.similarity_score,
                "confidence": result.confidence,
                "predicted_tag": result.routed_tag
            })
        except Exception as e:
            logger.log_error(
                component="calibration",
                error_message=f"Error processing note {note.note_id}: {e}",
                error_type="calibration_error"
            )
    
    # Extract similarity scores and create synthetic labels for calibration
    # In practice, these would come from human annotations
    similarity_scores = [d["similarity_score"] for d in calibration_data]
    
    # Create synthetic labels based on similarity threshold
    # This is a simplified approach - real implementation would use human labels
    true_labels = [score > 0.7 for score in similarity_scores]
    
    # Initialize calibrator
    calibrator = ConfidenceCalibrator(method="platt")
    
    # Fit calibrator
    calibration_result = calibrator.fit(similarity_scores, true_labels)
    
    # Evaluate calibration quality
    calibrated_scores = calibrator.predict_proba(similarity_scores)
    calibration_quality = evaluate_calibration_quality(true_labels, calibrated_scores)
    
    # Save calibration results
    calibration_report = {
        "calibration_result": calibration_result.to_dict(),
        "calibration_quality": calibration_quality,
        "original_scores": similarity_scores[:10],  # First 10 for reference
        "calibrated_scores": calibrated_scores[:10]
    }
    
    with open(output_path / "confidence_calibration_results.json", 'w') as f:
        json.dump(calibration_report, f, indent=2)
    
    # Save calibrated model
    calibrator.save_calibrator(str(output_path / "confidence_calibrator.json"))
    
    logger.log_event(
        event_type="calibration",
        component="evaluator",
        message="Confidence calibration complete",
        metadata={
            "brier_score_before": calibration_result.brier_score_before,
            "brier_score_after": calibration_result.brier_score_after,
            "improvement": calibration_result.brier_score_before - calibration_result.brier_score_after
        }
    )
    
    return calibration_report


def run_performance_tests(notes: List, tags: List,
                         best_model_name: str,
                         performance_monitor: PerformanceMonitor,
                         logger: MLLogger) -> Dict[str, Any]:
    """Run comprehensive performance tests."""
    logger.log_event(
        event_type="performance",
        component="evaluator",
        message="Starting performance tests"
    )
    
    # Start monitoring
    performance_monitor.start_monitoring(interval_seconds=10)
    
    # Initialize router
    embedder = TextEmbedder(best_model_name)
    tag_embedder = TagEmbedder(embedder)
    router = FAISSRouter(embedder, tag_embedder)
    router.build_index(tags)
    
    # Performance test scenarios
    test_scenarios = [
        {"name": "single_note", "notes": notes[:1]},
        {"name": "small_batch", "notes": notes[:10]},
        {"name": "medium_batch", "notes": notes[:100]},
        {"name": "large_batch", "notes": notes[:500]}
    ]
    
    performance_results = {}
    
    for scenario in test_scenarios:
        logger.log_event(
            event_type="performance",
            component="evaluator",
            message=f"Running performance test: {scenario['name']}"
        )
        
        start_time = time.time()
        processed_notes = 0
        errors = 0
        
        for note in scenario["notes"]:
            try:
                start_note_time = time.time()
                result = router.route_note(note.note_id, note.text, note.timestamp)
                note_time = (time.time() - start_note_time) * 1000
                
                performance_monitor.record_processing_time(note_time)
                processed_notes += 1
                
            except Exception as e:
                errors += 1
                performance_monitor.record_error()
                logger.log_error(
                    component="performance_test",
                    error_message=f"Error processing note {note.note_id}: {e}",
                    error_type="performance_error"
                )
        
        total_time = time.time() - start_time
        throughput = processed_notes / total_time if total_time > 0 else 0
        
        performance_results[scenario["name"]] = {
            "total_notes": len(scenario["notes"]),
            "processed_notes": processed_notes,
            "errors": errors,
            "total_time_seconds": total_time,
            "throughput_notes_per_second": throughput,
            "avg_processing_time_ms": total_time * 1000 / processed_notes if processed_notes > 0 else 0
        }
        
        logger.log_performance(
            component="performance_test",
            operation=scenario["name"],
            duration_ms=total_time * 1000,
            additional_metrics={
                "throughput_notes_per_second": throughput,
                "error_rate": errors / len(scenario["notes"]) if scenario["notes"] else 0
            }
        )
    
    # Stop monitoring
    performance_monitor.stop_monitoring()
    
    # Get performance summary
    performance_summary = performance_monitor.get_performance_summary()
    
    logger.log_event(
        event_type="performance",
        component="evaluator",
        message="Performance tests complete",
        metadata=performance_summary
    )
    
    return {
        "scenario_results": performance_results,
        "system_summary": performance_summary
    }


def generate_comprehensive_report(validation_data: Dict[str, Any],
                                benchmark_results: Dict[str, Any],
                                calibration_results: Dict[str, Any],
                                performance_results: Dict[str, Any],
                                output_path: Path,
                                logger: MLLogger) -> None:
    """Generate comprehensive evaluation report."""
    logger.log_event(
        event_type="reporting",
        component="evaluator",
        message="Generating comprehensive evaluation report"
    )
    
    # Compile comprehensive report
    comprehensive_report = {
        "evaluation_summary": {
            "timestamp": time.time(),
            "total_notes_evaluated": len(validation_data["notes"]),
            "total_tags_evaluated": len(validation_data["tags"]),
            "best_model": benchmark_results["summary"]["best_model"],
            "best_f1_score": benchmark_results["summary"]["best_f1_score"],
            "calibration_improvement": calibration_results["calibration_result"]["brier_score_before"] - 
                                     calibration_results["calibration_result"]["brier_score_after"]
        },
        "data_quality": validation_data["validation_summary"],
        "model_performance": benchmark_results,
        "confidence_calibration": calibration_results,
        "system_performance": performance_results,
        "recommendations": _generate_recommendations(
            validation_data, benchmark_results, calibration_results, performance_results
        )
    }
    
    # Save comprehensive report
    with open(output_path / "comprehensive_evaluation_report.json", 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    # Generate human-readable summary
    summary_text = _generate_human_readable_summary(comprehensive_report)
    with open(output_path / "evaluation_summary.txt", 'w') as f:
        f.write(summary_text)
    
    logger.log_event(
        event_type="reporting",
        component="evaluator",
        message="Comprehensive evaluation report generated",
        metadata={"report_files": ["comprehensive_evaluation_report.json", "evaluation_summary.txt"]}
    )


def _generate_recommendations(validation_data: Dict[str, Any],
                            benchmark_results: Dict[str, Any],
                            calibration_results: Dict[str, Any],
                            performance_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on evaluation results."""
    recommendations = []
    
    # Data quality recommendations
    if validation_data["validation_summary"]["notes"]["invalid"] > 0:
        recommendations.append("Data quality issues detected. Review and clean input data.")
    
    if validation_data["validation_summary"]["notes"]["outliers"] > 0:
        recommendations.append("Outliers detected in note data. Consider data preprocessing.")
    
    # Model performance recommendations
    best_f1 = benchmark_results["summary"]["best_f1_score"]
    if best_f1 < 0.7:
        recommendations.append("Model performance below target (F1 < 0.7). Consider model selection or data quality.")
    
    # Calibration recommendations
    calibration_improvement = (calibration_results["calibration_result"]["brier_score_before"] - 
                             calibration_results["calibration_result"]["brier_score_after"])
    if calibration_improvement < 0.01:
        recommendations.append("Minimal calibration improvement. Consider different calibration method.")
    
    # Performance recommendations
    avg_processing_time = performance_results["system_summary"].get("processing_time_ms", {}).get("mean", 0)
    if avg_processing_time > 100:
        recommendations.append("Processing time above target (100ms). Consider optimization.")
    
    return recommendations


def _generate_human_readable_summary(report: Dict[str, Any]) -> str:
    """Generate human-readable summary of evaluation results."""
    summary = []
    summary.append("=" * 80)
    summary.append("SNR SYSTEM EVALUATION REPORT")
    summary.append("=" * 80)
    summary.append("")
    
    # Summary
    summary.append("EVALUATION SUMMARY:")
    summary.append(f"  Total Notes Evaluated: {report['evaluation_summary']['total_notes_evaluated']}")
    summary.append(f"  Total Tags Evaluated: {report['evaluation_summary']['total_tags_evaluated']}")
    summary.append(f"  Best Model: {report['evaluation_summary']['best_model']}")
    summary.append(f"  Best F1 Score: {report['evaluation_summary']['best_f1_score']:.3f}")
    summary.append("")
    
    # Data Quality
    summary.append("DATA QUALITY:")
    notes_quality = report['data_quality']['notes']
    summary.append(f"  Note Validation Rate: {notes_quality['valid']}/{notes_quality['total']} ({notes_quality['valid']/notes_quality['total']*100:.1f}%)")
    summary.append(f"  Average Note Quality Score: {notes_quality['avg_quality_score']:.3f}")
    summary.append(f"  Outliers Detected: {notes_quality['outliers']}")
    summary.append("")
    
    # Model Performance
    summary.append("MODEL PERFORMANCE:")
    for model in report['model_performance']['model_comparisons']:
        summary.append(f"  {model['model_name']}:")
        summary.append(f"    F1 Score: {model['f1_score']:.3f}")
        summary.append(f"    Precision: {model['precision']:.3f}")
        summary.append(f"    Recall: {model['recall']:.3f}")
        summary.append(f"    Processing Time: {model['processing_time_ms']:.1f}ms")
        summary.append("")
    
    # Performance
    summary.append("SYSTEM PERFORMANCE:")
    perf_summary = report['system_performance']['system_summary']
    if 'processing_time_ms' in perf_summary:
        summary.append(f"  Average Processing Time: {perf_summary['processing_time_ms']['mean']:.1f}ms")
    if 'throughput_notes_per_second' in perf_summary:
        summary.append(f"  Throughput: {perf_summary['throughput_notes_per_second']['mean']:.1f} notes/sec")
    summary.append("")
    
    # Recommendations
    summary.append("RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        summary.append(f"  {i}. {rec}")
    summary.append("")
    
    summary.append("=" * 80)
    
    return "\n".join(summary)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive SNR system evaluation")
    parser.add_argument("--notes", required=True, help="Path to notes CSV file")
    parser.add_argument("--tags", default="tags/default_tags.yaml", help="Path to tags YAML file")
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument("--models", nargs="+", 
                       default=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                       help="List of models to benchmark")
    
    args = parser.parse_args()
    
    print("Starting comprehensive SNR system evaluation...")
    print(f"Notes file: {args.notes}")
    print(f"Tags file: {args.tags}")
    print(f"Output directory: {args.output}")
    print(f"Models to benchmark: {args.models}")
    print()
    
    try:
        # Setup evaluation environment
        env = setup_evaluation_environment(args.output)
        
        # Validate input data
        validation_data = validate_input_data(
            args.notes, args.tags, env["data_validator"], env["logger"]
        )
        
        # Benchmark models
        benchmark_results = benchmark_models(
            validation_data["notes"], validation_data["tags"], 
            args.models, env["output_path"], env["logger"]
        )
        
        # Get best model
        best_model = benchmark_results["summary"]["best_model"]
        print(f"Best model: {best_model}")
        
        # Evaluate confidence calibration
        calibration_results = evaluate_confidence_calibration(
            validation_data["notes"], validation_data["tags"],
            best_model, env["output_path"], env["logger"]
        )
        
        # Run performance tests
        performance_results = run_performance_tests(
            validation_data["notes"], validation_data["tags"],
            best_model, env["performance_monitor"], env["logger"]
        )
        
        # Generate comprehensive report
        generate_comprehensive_report(
            validation_data, benchmark_results, calibration_results, 
            performance_results, env["output_path"], env["logger"]
        )
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Best model: {best_model}")
        print(f"Best F1 score: {benchmark_results['summary']['best_f1_score']:.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 