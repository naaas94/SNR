#!/usr/bin/env python3
"""
Generate test data for SNR system evaluation.
"""

import csv
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_notes(num_notes: int = 100) -> list:
    """Generate sample notes for testing."""
    
    # Sample note templates by category
    note_templates = {
        "reuse_candidate": [
            "could use the {component} as platform for all projects",
            "this {tool} might generalize to other cases",
            "reuse my {project} as base for new projects",
            "the {system} could be adapted for {use_case}",
            "consider using {framework} for similar problems"
        ],
        "learning_insight": [
            "learned that {concept} works better than {alternative}",
            "realized that the bottleneck was in {component}",
            "insight: most issues come from {cause}",
            "discovered that {technique} improves {metric}",
            "found that {approach} is more effective than expected"
        ],
        "todo_item": [
            "need to fix the {issue} in {component}",
            "todo: implement {feature} for {system}",
            "should refactor {code} to support {requirement}",
            "must update {documentation} with {information}",
            "plan to optimize {process} for better {performance}"
        ],
        "research_question": [
            "how does {concept} work in {context}?",
            "what's the best way to handle {problem}?",
            "research: compare different {approaches}",
            "investigate whether {hypothesis} is true",
            "explore alternatives to {current_solution}"
        ],
        "code_snippet": [
            "def {function}({params}): return {result}",
            "{command} -{flag} {value}",
            "SELECT * FROM {table} WHERE {condition}",
            "import {module}; {operation}",
            "class {Class}: def __init__(self): {initialization}"
        ],
        "resource_link": [
            "https://{domain}/paper - {title}",
            "check out this tool: https://{domain}/{tool}",
            "useful resource: {url} for {topic}",
            "reference: {paper} on {subject}",
            "documentation: {docs} for {technology}"
        ],
        "meeting_note": [
            "team decided to use {technology} for {purpose}",
            "meeting with {stakeholder}: they want {requirement}",
            "discussed {topic}: {decision}",
            "agreed on {approach} for {project}",
            "reviewed {deliverable} and provided {feedback}"
        ],
        "idea_brainstorm": [
            "idea: build a {system} for {purpose}",
            "what if we could {capability}?",
            "brainstorm: ways to improve {process}",
            "concept: {innovation} that could {benefit}",
            "explore: {possibility} for {application}"
        ]
    }
    
    # Fill-in words for templates
    fill_words = {
        "component": ["API", "database", "frontend", "backend", "cache", "queue"],
        "tool": ["script", "library", "framework", "algorithm", "method"],
        "project": ["classifier", "analyzer", "dashboard", "service", "app"],
        "system": ["platform", "infrastructure", "architecture", "pipeline"],
        "use_case": ["monitoring", "analysis", "automation", "reporting"],
        "framework": ["React", "Django", "TensorFlow", "FastAPI", "Spark"],
        "concept": ["vector embeddings", "caching", "async processing", "microservices"],
        "alternative": ["TF-IDF", "synchronous calls", "monolith", "batch processing"],
        "cause": ["edge cases", "data quality", "timing issues", "resource limits"],
        "technique": ["optimization", "parallelization", "caching", "indexing"],
        "metric": ["performance", "accuracy", "reliability", "scalability"],
        "approach": ["iterative", "agile", "waterfall", "test-driven"],
        "issue": ["bug", "performance problem", "security vulnerability"],
        "feature": ["authentication", "caching", "logging", "monitoring"],
        "code": ["module", "function", "class", "service"],
        "requirement": ["scalability", "security", "performance", "maintainability"],
        "documentation": ["README", "API docs", "user guide", "architecture doc"],
        "information": ["new features", "best practices", "examples", "troubleshooting"],
        "process": ["deployment", "testing", "monitoring", "backup"],
        "performance": ["speed", "efficiency", "throughput", "response time"],
        "context": ["distributed systems", "machine learning", "web applications"],
        "problem": ["data imbalance", "scaling", "security", "compatibility"],
        "approaches": ["algorithms", "architectures", "frameworks", "methodologies"],
        "hypothesis": ["this approach is better", "the bottleneck is here", "this will scale"],
        "current_solution": ["manual process", "legacy system", "basic approach"],
        "function": ["process_data", "validate_input", "transform_result", "calculate_metric"],
        "params": ["data, config", "input, options", "args, kwargs", "x, y"],
        "result": ["processed_data", "validation_result", "transformed_data", "calculated_value"],
        "command": ["docker run", "python script.py", "npm install", "git commit"],
        "flag": ["p", "v", "d", "f", "r"],
        "value": ["8080:8080", "requirements.txt", "package.json", "main"],
        "table": ["users", "orders", "products", "logs"],
        "condition": ["created_at > NOW() - INTERVAL '7 days'", "status = 'active'"],
        "module": ["pandas", "numpy", "requests", "json"],
        "operation": ["df.head()", "np.array()", "response.json()", "json.dumps()"],
        "Class": ["DataProcessor", "Validator", "Transformer", "Calculator"],
        "initialization": ["self.data = data", "self.config = config", "self.logger = logger"],
        "domain": ["arxiv.org", "github.com", "docs.python.org", "stackoverflow.com"],
        "title": ["Attention is All You Need", "BERT: Pre-training", "ResNet Architecture"],
        "tool": ["streamlit/streamlit", "pytorch/pytorch", "tensorflow/tensorflow"],
        "url": ["https://example.com/resource", "https://docs.example.com"],
        "topic": ["machine learning", "web development", "data science"],
        "paper": ["BERT paper", "Transformer paper", "ResNet paper"],
        "subject": ["natural language processing", "computer vision", "reinforcement learning"],
        "docs": ["official documentation", "tutorial", "guide"],
        "technology": ["Python", "React", "Docker", "Kubernetes"],
        "purpose": ["frontend", "backend", "data processing", "deployment"],
        "stakeholder": ["client", "team lead", "product manager", "stakeholder"],
        "requirement": ["MVP by end of month", "additional features", "performance improvements"],
        "decision": ["microservices vs monolith", "React vs Vue", "Python vs Node.js"],
        "deliverable": ["prototype", "documentation", "presentation"],
        "feedback": ["positive comments", "suggestions for improvement", "approval"],
        "capability": ["automatically tag emails", "predict user behavior", "optimize performance"],
        "innovation": ["AI-powered tool", "automated system", "smart algorithm"],
        "benefit": ["save time", "improve accuracy", "reduce costs"],
        "possibility": ["real-time processing", "automated deployment", "intelligent routing"],
        "application": ["customer service", "content management", "data analysis"]
    }
    
    notes = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_notes):
        # Randomly select a tag category
        tag_category = random.choice(list(note_templates.keys()))
        template = random.choice(note_templates[tag_category])
        
        # Fill in the template
        note_text = template
        for placeholder, options in fill_words.items():
            if placeholder in note_text:
                note_text = note_text.replace(f"{{{placeholder}}}", random.choice(options))
        
        # Generate timestamp (within last 30 days)
        timestamp = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        note = {
            "note_id": str(uuid.uuid4()),
            "text": note_text,
            "timestamp": timestamp.isoformat(),
            "true_tag": tag_category  # For evaluation purposes
        }
        
        notes.append(note)
    
    return notes


def save_notes_to_csv(notes: list, output_file: str):
    """Save notes to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['note_id', 'text', 'timestamp', 'true_tag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for note in notes:
            writer.writerow(note)


def main():
    """Generate test data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data for SNR evaluation")
    parser.add_argument("--num-notes", type=int, default=100, help="Number of notes to generate")
    parser.add_argument("--output", default="test_data/sample_notes.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_notes} sample notes...")
    
    # Generate notes
    notes = generate_sample_notes(args.num_notes)
    
    # Save to CSV
    save_notes_to_csv(notes, args.output)
    
    print(f"Generated {len(notes)} notes and saved to {args.output}")
    
    # Print sample
    print("\nSample notes:")
    for i, note in enumerate(notes[:5]):
        print(f"{i+1}. [{note['true_tag']}] {note['text']}")
    
    # Print distribution
    tag_counts = {}
    for note in notes:
        tag = note['true_tag']
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    print(f"\nTag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count} notes")


if __name__ == "__main__":
    main() 