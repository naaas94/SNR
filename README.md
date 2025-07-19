# SNR — SEMANTIC NOTE ROUTER

> **A lightweight semantic routing engine that classifies short-form personal notes into symbolic categories**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: MVP 2.0](https://img.shields.io/badge/Status-MVP%202.0-orange.svg)]()

## 🧭 DESIGN PHILOSOPHY

This is not a tagging engine. It's a **semantic prosthesis for symbolic cognition**.

SNR aims to:
- **Treat symbolic categories as conceptual operators**, not keywords
- **Avoid shallow embeddings** by grounding tags in descriptions + examples  
- **Embrace traceability**: every routed note stored with versioning and metadata
- **Enable deliberate disambiguation**, not naïve nearest-neighbor tagging
- **Design for modularity** only after core flow is usable end-to-end
- **Close the loop** with corrections, learning and symbolic drift detection

## 🚀 QUICK START

```bash
# Clone and setup
git clone <repo>
cd SNR
pip install -r requirements.txt

# Run router on sample data
python scripts/run_router.py --input examples/sample_notes.csv --mode faiss

# View results
cat outputs/tagged_notes.json
```

## 📐 ARCHITECTURE OVERVIEW

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INGESTION     │    │   EMBEDDING     │    │    ROUTING      │
│                 │    │                 │    │                 │
│ • CSV/TXT/MD    │───▶│ • SentenceTrans │───▶│ • FAISS KNN     │
│ • Normalization │    │ • Tag encoding  │    │ • Disambiguation│
│ • Deduplication │    │ • Vector store  │    │ • Confidence    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    OUTPUT       │
                       │                 │
                       │ • JSON metadata │
                       │ • MD summaries  │
                       │ • Obsidian YAML │
                       └─────────────────┘
```

## 🗂️ DATA MODEL

### NoteRecord Output
```json
{
  "note_id": "9a12f7ac-05e1-4c89-b1e3-f32a1bd5c9a2",
  "text": "could use the PCC as platform for all projs which is cool",
  "timestamp": "2025-07-15T13:42:10",
  "routed_tag": "reuse_candidate",
  "similarity_score": 0.84,
  "confidence": 0.72,
  "tag_description": "A note that suggests reusing or repurposing an existing structure for future work",
  "tag_examples": ["reuse my spotify classifier as base for new projects"],
  "routed_by": "faiss-v1",
  "version": 1,
  "metadata": {
    "embedding_model": "all-mpnet-base-v2",
    "processing_time_ms": 45,
    "disambiguation_used": false
  }
}
```

### Tag Ontology Format
```yaml
# tags/default_tags.yaml
- tag: reuse_candidate
  description: >
    A note that suggests that an idea, tool, or structure could be reused 
    in other contexts or projects.
  examples:
    - "could use the pcc as platform for all projs"
    - "this script might generalize to other cases"
    - "reuse my spotify classifier as base for new projects"
  confidence_threshold: 0.7
  parent_tag: "development_ideas"
```

## 🧰 PROJECT STRUCTURE

```
SNR/
├── src/
│   ├── config/              # YAML tag ontologies & settings
│   ├── ingestion/           # load_notes.py, normalizers
│   ├── embedding/           # embedder.py, tag_encoder.py
│   ├── routing/             # route_with_faiss.py, disambiguator.py
│   ├── evaluation/          # metrics.py, confidence_calibrator.py
│   ├── postprocessing/      # format_output.py, exporters
│   ├── utils/               # logger.py, file_utils.py
├── tags/
│   └── default_tags.yaml    # Semantic tag catalog
├── tests/                   # Unit & integration tests
├── examples/                # Sample notes and tags
├── outputs/                 # JSON/MD routed results
├── scripts/
│   ├── run_router.py        # CLI entry point
│   ├── benchmark_models.py  # Model comparison
│   └── generate_sample_data.py
├── requirements.txt
├── .env.example
├── README.md
└── Dockerfile
```

## 🚀 USAGE

### Basic Routing
```bash
python scripts/run_router.py \
  --input examples/sample_notes.csv \
  --mode faiss \
  --top_k 3 \
  --output outputs/tagged_notes.json
```

### Advanced Options
```bash
python scripts/run_router.py \
  --input my_notes.csv \
  --mode faiss \
  --disambiguate \
  --confidence_threshold 0.7 \
  --format markdown \
  --export_obsidian \
  --evaluate
```

### Configuration
```bash
# Use custom tag ontology
python scripts/run_router.py \
  --input notes.csv \
  --tags tags/my_custom_tags.yaml

# Benchmark different embedding models
python scripts/benchmark_models.py \
  --test_set examples/evaluation_notes.csv \
  --models all-mpnet-base-v2 all-MiniLM-L6-v2
```

## ✅ MVP STATUS

### Core Capabilities ✅
- [x] Embeds notes + semantic tag descriptions
- [x] KNN tag matching with FAISS
- [x] Configurable tag ontology in YAML
- [x] JSON output with full trace
- [x] Optional markdown export
- [x] Basic CLI runner
- [x] Ready for Obsidian use

### Key Additions in v2.0 ✅
- [x] Grounded tags (desc + examples)
- [x] Note versioning + metadata
- [x] Disambiguation hook
- [x] Evaluation hooks (low-sim reporting, confusion)

### Critical Improvements Needed 🚨
- [ ] **Evaluation Framework**: Precision/recall metrics, cross-validation
- [ ] **Model Benchmarking**: Test multiple embedding models
- [ ] **Confidence Calibration**: Convert similarity scores to calibrated probabilities
- [ ] **Active Learning**: Uncertainty sampling for manual review
- [ ] **Production Monitoring**: Logging, health checks, alerting

## 🧪 TESTING & EVALUATION

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Model Benchmarking
```bash
python scripts/benchmark_models.py \
  --test_set examples/evaluation_notes.csv \
  --output results/benchmark_results.json
```

### Human Evaluation
```bash
python scripts/human_evaluation.py \
  --notes examples/human_eval_notes.csv \
  --output results/human_agreement.json
```

## 🔮 ROADMAP

### Short-Term (2-4 weeks) 🎯
- [ ] **Evaluation Framework**: Implement precision/recall, semantic similarity metrics
- [ ] **Model Selection**: Benchmark 3-5 embedding models, select optimal
- [ ] **Confidence Calibration**: Add Platt scaling for calibrated probabilities
- [ ] **Active Learning**: Uncertainty sampling for manual review

### Mid-Term (1-3 months) 🚀
- [ ] **Production Readiness**: Logging, monitoring, health checks
- [ ] **Advanced Routing**: Ensemble methods, contextual disambiguation
- [ ] **GUI Dashboard**: Streamlit interface for interactive exploration
- [ ] **Obsidian Integration**: YAML frontmatter writer

### Long-Term (3-6 months) 🔮
- [ ] **Self-Improving System**: Few-shot learning, semantic drift detection
- [ ] **Multi-Modal**: Support for images, audio, structured data
- [ ] **Temporal Dynamics**: Note evolution tracking, concept drift
- [ ] **Personalization**: User-specific fine-tuning, pattern learning

## 🧠 DESIGN PRINCIPLES

| Principle | Implementation |
|-----------|----------------|
| **Symbolic grounding** | Tags must have natural-language grounding, not just labels |
| **Traceability** | Every note routed has UUID, version, similarity score, routing method |
| **Minimal deployability** | Default mode runs E2E with no config required |
| **Modular, not fragmented** | Modules exist only to serve coherent default flow |
| **Learnable over time** | Manual corrections form foundation for fine-tuning |

## 📊 SUCCESS METRICS

### Routing Quality
- **Precision@k**: Accuracy of top-k tag predictions
- **Semantic Similarity**: Cosine similarity scores distribution
- **Cross-validation**: Performance across different note types

### User Experience
- **Manual Correction Rate**: % of notes requiring human review
- **Time to Find**: How quickly users locate notes via tags
- **Tag Adoption**: Rate of new tag creation and usage

### System Performance
- **Inference Latency**: Average processing time per note
- **Memory Usage**: RAM consumption for different dataset sizes
- **Scalability**: Performance with 10k+ notes

## 🚨 CRITICAL LIMITATIONS

### Current Model Choice
- **all-MiniLM-L6-v2**: 2019 model, 384 dimensions, limited expressiveness
- **Recommendation**: Benchmark against all-mpnet-base-v2 (768d, better performance)

### Missing Evaluation
- **No quantitative metrics**: Can't measure if system is improving
- **No validation pipeline**: No way to test on unseen data
- **No confidence calibration**: Raw similarity scores aren't probabilities

### Scalability Concerns
- **FAISS Index**: No persistence, incremental updates, or memory management
- **Batch Processing**: No efficient batch embedding generation
- **Error Handling**: Missing graceful degradation for edge cases

## 🤝 CONTRIBUTING

### Development Setup
```bash
git clone <repo>
cd SNR
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### Code Style
- Follow PEP 8 with 88-character line limit
- Use type hints for all functions
- Add docstrings for all public APIs
- Write tests for new features

### Testing Strategy
- Unit tests for all modules
- Integration tests for end-to-end flows
- Performance benchmarks for routing
- Human evaluation protocols

## 📄 LICENSE

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 AUTHOR

**Alejandro Garay** - ML Engineer & Cognitive Systems Designer

---

> **"The best semantic router is the one that becomes invisible to your thinking process."**

*This project is in active development. Expect breaking changes until v1.0.* 
noteId: "a164285061a111f0947bd1be0d0d8af7"
tags: []

---

 