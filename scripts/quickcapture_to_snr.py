#!/usr/bin/env python3
"""
Integration script demonstrating QuickCapture to SNR pipeline.

This script shows how QuickCapture (symbolic ingestion) feeds into 
SNR (semantic routing) for a complete thought capture and organization system.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.quickcapture_parser import QuickCaptureProcessor, ParsedNote
from utils.logger import setup_logging, MLLogger
from utils.monitoring import PerformanceMonitor, ErrorTracker
from embedding.embedder import TextEmbedder, TagEmbedder
from routing.route_with_faiss import FAISSRouter
from config.tag_loader import TagLoader
from evaluation.metrics import calculate_precision_recall


def setup_integration_environment(output_dir: str) -> Dict[str, Any]:
    """Setup integration environment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_file=str(output_path / "integration.log"),
        log_level="INFO",
        enable_console=True
    )
    
    # Setup monitoring
    performance_monitor = PerformanceMonitor(
        logger=logger,
        metrics_file=str(output_path / "performance_metrics.json")
    )
    
    error_tracker = ErrorTracker(logger=logger)
    
    logger.log_event(
        event_type="integration",
        component="pipeline",
        message="Integration environment setup complete",
        metadata={"output_dir": str(output_path)}
    )
    
    return {
        "logger": logger,
        "performance_monitor": performance_monitor,
        "error_tracker": error_tracker,
        "output_path": output_path
    }


def generate_sample_quickcapture_notes() -> List[str]:
    """Generate sample QuickCapture notes for demonstration."""
    sample_notes = [
        "#learning_insight #ml learned that vector embeddings work better than tf-idf for semantic search //important discovery @2025-01-27T10:30:00",
        "#todo_item #backend need to fix the authentication bug in the login flow //high priority @2025-01-27T11:15:00",
        "#research_question #nlp how does attention mechanism work in transformers? //need to investigate @2025-01-27T12:00:00",
        "#code_snippet #python def quick_sort(arr): return sorted(arr) if len(arr) <= 1 else ... //useful algorithm @2025-01-27T13:45:00",
        "#resource_link #ml https://arxiv.org/abs/1706.03762 - Attention is All You Need //seminal paper @2025-01-27T14:20:00",
        "#meeting_note #team team decided to use React for the frontend, backend in Python //architecture decision @2025-01-27T15:30:00",
        "#idea_brainstorm #product idea: build a semantic search engine for personal notes //potential project @2025-01-27T16:15:00",
        "#reuse_candidate #code could use the PCC as platform for all projects //architecture insight @2025-01-27T17:00:00",
        "#learning_insight #debugging realized that the bottleneck was in the data preprocessing step //performance insight @2025-01-27T18:30:00",
        "#todo_item #frontend todo: implement caching for the API responses //optimization needed @2025-01-27T19:45:00"
    ]
    return sample_notes


def process_quickcapture_notes(notes: List[str], output_dir: str, 
                              logger: MLLogger) -> List[ParsedNote]:
    """Process notes through QuickCapture pipeline."""
    logger.log_event(
        event_type="processing",
        component="quickcapture",
        message="Starting QuickCapture processing",
        metadata={"num_notes": len(notes)}
    )
    
    # Process notes through QuickCapture
    with QuickCaptureProcessor(output_dir, logger) as processor:
        parsed_notes = processor.process_batch(notes, auto_timestamp=True)
    
    # Log processing results
    valid_notes = [note for note in parsed_notes if note.valid]
    invalid_notes = [note for note in parsed_notes if not note.valid]
    
    logger.log_event(
        event_type="processing",
        component="quickcapture",
        message="QuickCapture processing complete",
        metadata={
            "total_notes": len(parsed_notes),
            "valid_notes": len(valid_notes),
            "invalid_notes": len(invalid_notes),
            "validation_rate": len(valid_notes) / len(parsed_notes) if parsed_notes else 0
        }
    )
    
    return parsed_notes


def setup_snr_router(model_name: str, tags_file: str, logger: MLLogger) -> FAISSRouter:
    """Setup SNR router for semantic routing."""
    logger.log_event(
        event_type="setup",
        component="snr",
        message="Setting up SNR router",
        metadata={"model_name": model_name}
    )
    
    # Load tags
    tag_loader = TagLoader(tags_file)
    tags = tag_loader.load_tags()
    
    # Initialize embedders and router
    embedder = TextEmbedder(model_name)
    tag_embedder = TagEmbedder(embedder)
    router = FAISSRouter(embedder, tag_embedder)
    
    # Build index
    router.build_index(tags)
    
    logger.log_event(
        event_type="setup",
        component="snr",
        message="SNR router setup complete",
        metadata={"num_tags": len(tags), "model_name": model_name}
    )
    
    return router


def route_quickcapture_notes(parsed_notes: List[ParsedNote], router: FAISSRouter,
                           logger: MLLogger, performance_monitor: PerformanceMonitor,
                           error_tracker: ErrorTracker) -> List[Dict[str, Any]]:
    """Route QuickCapture notes through SNR."""
    logger.log_event(
        event_type="routing",
        component="snr",
        message="Starting semantic routing of QuickCapture notes",
        metadata={"num_notes": len(parsed_notes)}
    )
    
    routing_results = []
    
    for parsed_note in parsed_notes:
        try:
            start_time = time.time()
            
            # Route the note body through SNR
            result = router.route_note(
                note_id=parsed_note.timestamp,  # Use timestamp as ID
                text=parsed_note.body,
                timestamp=parsed_note.timestamp
            )
            
            processing_time = (time.time() - start_time) * 1000
            performance_monitor.record_processing_time(processing_time)
            
            # Create integrated result
            integrated_result = {
                "quickcapture_data": {
                    "tags": parsed_note.tags,
                    "body": parsed_note.body,
                    "comment": parsed_note.comment,
                    "timestamp": parsed_note.timestamp,
                    "valid": parsed_note.valid,
                    "issues": parsed_note.issues
                },
                "snr_routing": {
                    "routed_tag": result.routed_tag,
                    "similarity_score": result.similarity_score,
                    "confidence": result.confidence,
                    "tag_description": result.tag_description,
                    "tag_examples": result.tag_examples,
                    "processing_time_ms": result.metadata.get("processing_time_ms", 0)
                },
                "integration_metadata": {
                    "processing_time_ms": processing_time,
                    "model_name": result.metadata.get("embedding_model", "unknown"),
                    "integration_timestamp": time.time()
                }
            }
            
            routing_results.append(integrated_result)
            
            # Log routing result
            logger.log_prediction(
                note_id=parsed_note.timestamp,
                predicted_tag=result.routed_tag,
                confidence=result.confidence,
                processing_time_ms=processing_time,
                similarity_score=result.similarity_score,
                model_name=result.metadata.get("embedding_model", "unknown")
            )
            
        except Exception as e:
            error_tracker.record_error(
                error_type="routing_error",
                component="snr",
                message=f"Failed to route note: {e}",
                context={"note_body": parsed_note.body[:100]}
            )
            
            # Create error result
            error_result = {
                "quickcapture_data": {
                    "tags": parsed_note.tags,
                    "body": parsed_note.body,
                    "comment": parsed_note.comment,
                    "timestamp": parsed_note.timestamp,
                    "valid": parsed_note.valid,
                    "issues": parsed_note.issues
                },
                "snr_routing": {
                    "routed_tag": "error",
                    "similarity_score": 0.0,
                    "confidence": 0.0,
                    "tag_description": "Routing failed",
                    "tag_examples": [],
                    "processing_time_ms": 0
                },
                "integration_metadata": {
                    "processing_time_ms": 0,
                    "model_name": "unknown",
                    "integration_timestamp": time.time(),
                    "error": str(e)
                }
            }
            
            routing_results.append(error_result)
    
    logger.log_event(
        event_type="routing",
        component="snr",
        message="Semantic routing complete",
        metadata={"routed_notes": len(routing_results)}
    )
    
    return routing_results


def analyze_integration_results(parsed_notes: List[ParsedNote], 
                              routing_results: List[Dict[str, Any]],
                              logger: MLLogger) -> Dict[str, Any]:
    """Analyze integration results and generate insights."""
    logger.log_event(
        event_type="analysis",
        component="integration",
        message="Starting integration analysis"
    )
    
    # QuickCapture analysis
    quickcapture_stats = {
        "total_notes": len(parsed_notes),
        "valid_notes": sum(1 for note in parsed_notes if note.valid),
        "invalid_notes": sum(1 for note in parsed_notes if not note.valid),
        "notes_with_tags": sum(1 for note in parsed_notes if note.tags),
        "notes_with_comments": sum(1 for note in parsed_notes if note.comment),
        "avg_tags_per_note": sum(len(note.tags) for note in parsed_notes) / len(parsed_notes) if parsed_notes else 0
    }
    
    # SNR routing analysis
    successful_routes = [r for r in routing_results if r["snr_routing"]["routed_tag"] != "error"]
    failed_routes = [r for r in routing_results if r["snr_routing"]["routed_tag"] == "error"]
    
    snr_stats = {
        "total_routes": len(routing_results),
        "successful_routes": len(successful_routes),
        "failed_routes": len(failed_routes),
        "success_rate": len(successful_routes) / len(routing_results) if routing_results else 0,
        "avg_confidence": sum(r["snr_routing"]["confidence"] for r in successful_routes) / len(successful_routes) if successful_routes else 0,
        "avg_similarity": sum(r["snr_routing"]["similarity_score"] for r in successful_routes) / len(successful_routes) if successful_routes else 0
    }
    
    # Tag correlation analysis
    tag_correlations = {}
    for parsed_note, routing_result in zip(parsed_notes, routing_results):
        quickcapture_tags = set(parsed_note.tags)
        snr_tag = routing_result["snr_routing"]["routed_tag"]
        
        for qc_tag in quickcapture_tags:
            if qc_tag not in tag_correlations:
                tag_correlations[qc_tag] = {"count": 0, "snr_matches": 0}
            
            tag_correlations[qc_tag]["count"] += 1
            if qc_tag == snr_tag:
                tag_correlations[qc_tag]["snr_matches"] += 1
    
    # Calculate correlation rates
    for tag, stats in tag_correlations.items():
        stats["correlation_rate"] = stats["snr_matches"] / stats["count"] if stats["count"] > 0 else 0
    
    # Integration insights
    insights = {
        "quickcapture_stats": quickcapture_stats,
        "snr_stats": snr_stats,
        "tag_correlations": tag_correlations,
        "recommendations": _generate_integration_recommendations(quickcapture_stats, snr_stats, tag_correlations)
    }
    
    logger.log_event(
        event_type="analysis",
        component="integration",
        message="Integration analysis complete",
        metadata={
            "quickcapture_validation_rate": quickcapture_stats["valid_notes"] / quickcapture_stats["total_notes"],
            "snr_success_rate": snr_stats["success_rate"],
            "avg_confidence": snr_stats["avg_confidence"]
        }
    )
    
    return insights


def _generate_integration_recommendations(quickcapture_stats: Dict[str, Any],
                                        snr_stats: Dict[str, Any],
                                        tag_correlations: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on integration analysis."""
    recommendations = []
    
    # QuickCapture recommendations
    if quickcapture_stats["invalid_notes"] > 0:
        recommendations.append("Some QuickCapture notes failed validation. Review grammar rules.")
    
    if quickcapture_stats["avg_tags_per_note"] < 1:
        recommendations.append("Low tag usage in QuickCapture. Consider encouraging tag usage.")
    
    # SNR recommendations
    if snr_stats["success_rate"] < 0.9:
        recommendations.append("SNR routing success rate below 90%. Review model and tag definitions.")
    
    if snr_stats["avg_confidence"] < 0.7:
        recommendations.append("Low average confidence in SNR routing. Consider confidence calibration.")
    
    # Tag correlation recommendations
    low_correlation_tags = [
        tag for tag, stats in tag_correlations.items()
        if stats["correlation_rate"] < 0.5 and stats["count"] > 2
    ]
    
    if low_correlation_tags:
        recommendations.append(f"Low correlation between QuickCapture and SNR tags: {low_correlation_tags}")
    
    return recommendations


def save_integration_results(routing_results: List[Dict[str, Any]], 
                           insights: Dict[str, Any],
                           output_path: Path) -> None:
    """Save integration results to files."""
    # Save routing results
    with open(output_path / "integration_results.json", 'w') as f:
        json.dump(routing_results, f, indent=2)
    
    # Save insights
    with open(output_path / "integration_insights.json", 'w') as f:
        json.dump(insights, f, indent=2)
    
    # Generate human-readable summary
    summary = _generate_integration_summary(routing_results, insights)
    with open(output_path / "integration_summary.txt", 'w') as f:
        f.write(summary)


def _generate_integration_summary(routing_results: List[Dict[str, Any]], 
                                insights: Dict[str, Any]) -> str:
    """Generate human-readable integration summary."""
    summary = []
    summary.append("=" * 80)
    summary.append("QUICKCAPTURE TO SNR INTEGRATION SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # QuickCapture summary
    qc_stats = insights["quickcapture_stats"]
    summary.append("QUICKCAPTURE PROCESSING:")
    summary.append(f"  Total Notes: {qc_stats['total_notes']}")
    summary.append(f"  Valid Notes: {qc_stats['valid_notes']} ({qc_stats['valid_notes']/qc_stats['total_notes']*100:.1f}%)")
    summary.append(f"  Notes with Tags: {qc_stats['notes_with_tags']}")
    summary.append(f"  Average Tags per Note: {qc_stats['avg_tags_per_note']:.1f}")
    summary.append("")
    
    # SNR summary
    snr_stats = insights["snr_stats"]
    summary.append("SNR SEMANTIC ROUTING:")
    summary.append(f"  Total Routes: {snr_stats['total_routes']}")
    summary.append(f"  Successful Routes: {snr_stats['successful_routes']} ({snr_stats['success_rate']*100:.1f}%)")
    summary.append(f"  Average Confidence: {snr_stats['avg_confidence']:.3f}")
    summary.append(f"  Average Similarity: {snr_stats['avg_similarity']:.3f}")
    summary.append("")
    
    # Tag correlations
    summary.append("TAG CORRELATIONS (QuickCapture → SNR):")
    for tag, stats in insights["tag_correlations"].items():
        if stats["count"] > 1:  # Only show tags used multiple times
            correlation_rate = stats["correlation_rate"] * 100
            summary.append(f"  {tag}: {stats['snr_matches']}/{stats['count']} ({correlation_rate:.1f}%)")
    summary.append("")
    
    # Sample results
    summary.append("SAMPLE INTEGRATION RESULTS:")
    for i, result in enumerate(routing_results[:5]):
        qc_data = result["quickcapture_data"]
        snr_data = result["snr_routing"]
        summary.append(f"  {i+1}. [{', '.join(qc_data['tags'])}] {qc_data['body'][:50]}...")
        summary.append(f"     → SNR: {snr_data['routed_tag']} (conf: {snr_data['confidence']:.2f})")
    summary.append("")
    
    # Recommendations
    summary.append("RECOMMENDATIONS:")
    for i, rec in enumerate(insights["recommendations"], 1):
        summary.append(f"  {i}. {rec}")
    summary.append("")
    
    summary.append("=" * 80)
    
    return "\n".join(summary)


def main():
    """Main integration function."""
    parser = argparse.ArgumentParser(description="QuickCapture to SNR integration demo")
    parser.add_argument("--output", default="integration_results", help="Output directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--tags", default="tags/default_tags.yaml", help="Tags file")
    parser.add_argument("--notes", help="Custom notes file (optional)")
    
    args = parser.parse_args()
    
    print("Starting QuickCapture to SNR integration demo...")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Tags file: {args.tags}")
    print()
    
    try:
        # Setup environment
        env = setup_integration_environment(args.output)
        
        # Generate or load sample notes
        if args.notes:
            with open(args.notes, 'r') as f:
                notes = [line.strip() for line in f if line.strip()]
        else:
            notes = generate_sample_quickcapture_notes()
        
        print(f"Processing {len(notes)} QuickCapture notes...")
        
        # Process through QuickCapture
        parsed_notes = process_quickcapture_notes(notes, args.output, env["logger"])
        
        # Setup SNR router
        router = setup_snr_router(args.model, args.tags, env["logger"])
        
        # Route notes through SNR
        routing_results = route_quickcapture_notes(
            parsed_notes, router, env["logger"], 
            env["performance_monitor"], env["error_tracker"]
        )
        
        # Analyze results
        insights = analyze_integration_results(parsed_notes, routing_results, env["logger"])
        
        # Save results
        save_integration_results(routing_results, insights, env["output_path"])
        
        print("\nIntegration completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"QuickCapture validation rate: {insights['quickcapture_stats']['valid_notes']/insights['quickcapture_stats']['total_notes']*100:.1f}%")
        print(f"SNR success rate: {insights['snr_stats']['success_rate']*100:.1f}%")
        print(f"Average confidence: {insights['snr_stats']['avg_confidence']:.3f}")
        
        # Print sample results
        print("\nSample results:")
        for i, result in enumerate(routing_results[:3]):
            qc_data = result["quickcapture_data"]
            snr_data = result["snr_routing"]
            print(f"  {i+1}. [{', '.join(qc_data['tags'])}] {qc_data['body'][:40]}...")
            print(f"     → SNR: {snr_data['routed_tag']} (conf: {snr_data['confidence']:.2f})")
        
    except Exception as e:
        print(f"Integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 