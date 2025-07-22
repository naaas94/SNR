---
noteId: "7ec5c941659111f08ed5b77ec1f2184b"
tags: []

---

# Core Data Structures

## Overview

QuickCapture employs a robust set of data structures to manage notes, metadata, and system state. These structures are designed to maintain consistency, ensure type safety, and facilitate efficient data processing across the system.

## Primary Data Models

### Note Structure

The `Note` class is the fundamental data structure representing a note within the system. It encapsulates all necessary information about a note, including its content, metadata, and classification details.

```python
class Note:
    note_id: str              # Unique identifier for the note
    content: str              # Main content of the note
    title: Optional[str]      # Optional title of the note
    tags: List[str]           # List of tags associated with the note
    metadata: Dict[str, Any]  # Additional metadata as key-value pairs
    created_at: datetime      # Timestamp of note creation
    updated_at: datetime      # Timestamp of the last update
    embedding: Optional[List[float]]  # Semantic embedding vector for the note
    category: Optional[str]   # Category assigned to the note
    confidence_score: float   # Confidence score of the note's classification
    source: Optional[str]     # Source from which the note was derived
    quality_score: float      # Assessment of the note's content quality
```

**Purpose**: The `Note` class serves as the primary container for note data, supporting operations such as storage, retrieval, and processing.

**Key Attributes**:
- `note_id`: Ensures each note is uniquely identifiable.
- `content`: Holds the main textual content of the note.
- `embedding`: Used for semantic similarity and search operations.

**Usage**: Instances of `Note` are created and manipulated throughout the system to represent and manage individual notes.

### Metadata Structure

The `NoteMetadata` class provides extended information about a note, focusing on its origin, processing, and analysis results.

```python
class NoteMetadata:
    author: Optional[str]     # Author of the note
    source_type: str          # Type of source (e.g., file, API, manual)
    file_path: Optional[str]  # Path to the original file, if applicable
    processing_time: float    # Time taken to process the note
    validation_status: str    # Result of the note's validation
    error_messages: List[str] # List of errors encountered during processing
    word_count: int           # Number of words in the note
    language: Optional[str]   # Detected language of the note
    sentiment: Optional[str]  # Sentiment analysis result
    entities: List[str]       # Named entities identified in the note
```

**Purpose**: This class captures metadata that provides context and insights into the note's lifecycle and content analysis.

**Key Attributes**:
- `processing_time`: Useful for performance monitoring and optimization.
- `validation_status`: Indicates the success or failure of validation checks.

**Usage**: `NoteMetadata` is typically associated with a `Note` to provide additional context and processing details.

## Configuration Data Structures

### Grammar Rules Configuration

The grammar rules configuration defines validation and content rules for notes, ensuring they meet specific criteria before processing.

```yaml
grammar_rules:
  validation_rules:
    - rule_name: "content_length"
      min_length: 10
      max_length: 10000
    - rule_name: "required_fields"
      fields: ["content", "note_id"]
  
  content_rules:
    - rule_name: "no_empty_content"
      pattern: "^\\s*$"
      action: "reject"
    - rule_name: "html_stripping"
      pattern: "<[^>]*>"
      action: "clean"
```

**Purpose**: To enforce content standards and prevent invalid or harmful data from entering the system.

**Key Rules**:
- `content_length`: Ensures notes have a reasonable length.
- `html_stripping`: Cleans HTML tags from note content.

**Usage**: Applied during the note validation phase to maintain data integrity.

### Semantic Validation Configuration

This configuration manages the parameters for semantic validation, including embedding and classification settings.

```yaml
semantic_validation:
  embedding_config:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    similarity_threshold: 0.8
  
  classification_config:
    categories: ["research", "personal", "work", "learning"]
    confidence_threshold: 0.7
    fallback_category: "general"
```

**Purpose**: To configure the semantic analysis and classification of notes, enhancing their categorization and retrieval.

**Key Parameters**:
- `similarity_threshold`: Determines the cutoff for semantic similarity.
- `confidence_threshold`: Sets the minimum confidence for classification acceptance.

**Usage**: Utilized during the semantic processing of notes to ensure accurate classification and similarity assessments.

### Storage Configuration

Defines the storage mechanisms for notes, including vector storage and file storage configurations.

```yaml
storage_config:
  vector_store:
    type: "chroma"
    path: "./storage/vector_store"
    collection_name: "notes"
  
  file_storage:
    base_path: "./storage/notes"
    backup_enabled: true
    compression: true
  
  metadata_store:
    type: "sqlite"
    path: "./storage/metadata.db"
```

**Purpose**: To specify how and where notes and their metadata are stored, ensuring data persistence and accessibility.

**Key Components**:
- `vector_store`: Manages semantic vectors for efficient retrieval.
- `metadata_store`: Uses SQLite for structured metadata storage.

**Usage**: Configurations are applied during system initialization to set up storage paths and options.

## Processing Data Structures

### Processing Pipeline State

The `ProcessingState` class tracks the state of a note as it moves through the processing pipeline.

```python
class ProcessingState:
    note_id: str
    stage: str                # Current processing stage
    status: str               # Success, error, pending
    start_time: datetime
    end_time: Optional[datetime]
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
```

**Purpose**: To monitor and report the progress and status of note processing, aiding in debugging and performance analysis.

**Key Attributes**:
- `stage`: Indicates the current phase of processing.
- `metrics`: Collects performance data for analysis.

**Usage**: Instances are updated as notes progress through different processing stages.

### Validation Results

The `ValidationResult` class encapsulates the outcome of note validation, including errors and warnings.

```python
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    score: float              # Overall validation score
    details: Dict[str, Any]   # Detailed validation info

class ValidationError:
    field: str
    message: str
    severity: str             # error, warning, info
    code: str                 # Error code for programmatic handling
```

**Purpose**: To provide a comprehensive report on the validation process, highlighting any issues that need attention.

**Key Attributes**:
- `is_valid`: A boolean indicating the overall success of validation.
- `errors`: Detailed information on validation failures.

**Usage**: Used to assess the validity of notes before further processing.

### Classification Results

The `ClassificationResult` class records the results of note classification, including primary and alternative categories.

```python
class ClassificationResult:
    primary_category: str
    confidence_score: float
    alternative_categories: List[Tuple[str, float]]
    tags: List[str]
    reasoning: str            # Explanation for classification
    model_version: str        # Model used for classification
```

**Purpose**: To document the classification outcome, providing insights into the decision-making process.

**Key Attributes**:
- `primary_category`: The main category assigned to the note.
- `reasoning`: Explains the rationale behind the classification.

**Usage**: Results are used to categorize notes and inform subsequent actions.

## Storage Data Structures

### Vector Store Entry

The `VectorEntry` class represents an entry in the vector store, linking semantic vectors to their source notes.

```python
class VectorEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    source_note_id: str
```

**Purpose**: To facilitate efficient semantic search and retrieval by storing vector representations of notes.

**Key Attributes**:
- `vector`: The semantic representation of the note.
- `source_note_id`: Links the vector to its originating note.

**Usage**: Entries are created and queried during semantic operations.

### File Storage Entry

The `FileEntry` class manages file-related metadata, ensuring integrity and traceability of stored files.

```python
class FileEntry:
    file_path: str
    file_name: str
    file_size: int
    content_hash: str
    created_at: datetime
    modified_at: datetime
    metadata: Dict[str, Any]
```

**Purpose**: To track file attributes and ensure data integrity through hashing and metadata.

**Key Attributes**:
- `content_hash`: Verifies the integrity of the file content.
- `metadata`: Stores additional file-related information.

**Usage**: Used to manage and verify files within the storage system.

## Observability Data Structures

### Metrics Data

The `MetricsData` class captures performance and usage metrics, supporting system monitoring and optimization.

```python
class MetricsData:
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]
    source: str
```

**Purpose**: To provide a structured format for recording and analyzing system metrics.

**Key Attributes**:
- `metric_name`: Identifies the specific metric being recorded.
- `value`: The measured value of the metric.

**Usage**: Metrics are collected and analyzed to monitor system health and performance.

### Health Status

The `HealthStatus` class provides a snapshot of the system's health, indicating the status of various components.

```python
class HealthStatus:
    component: str
    status: str               # healthy, degraded, unhealthy
    last_check: datetime
    response_time: float
    error_count: int
    details: Dict[str, Any]
```

**Purpose**: To assess and report the operational status of system components, aiding in maintenance and troubleshooting.

**Key Attributes**:
- `status`: Indicates the health of the component.
- `error_count`: Tracks the number of errors encountered.

**Usage**: Health statuses are monitored to ensure system reliability and prompt issue resolution.

### Performance Metrics

The `PerformanceMetrics` class records key performance indicators, supporting analysis and optimization efforts.

```python
class PerformanceMetrics:
    processing_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: datetime
```

**Purpose**: To provide detailed performance data for system analysis and improvement.

**Key Attributes**:
- `throughput`: Measures the system's processing capacity.
- `error_rate`: Indicates the frequency of errors.

**Usage**: Metrics are used to evaluate and enhance system performance.

## API Data Structures

### Request/Response Models

The `NoteRequest` and `NoteResponse` classes define the structure of API requests and responses, ensuring consistency and validation.

```python
class NoteRequest:
    content: str
    title: Optional[str]
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

class NoteResponse:
    note_id: str
    status: str
    message: str
    note: Optional[Note]
    processing_time: float
    errors: List[str]
```

**Purpose**: To standardize the format of data exchanged via the API, facilitating communication and integration.

**Key Attributes**:
- `NoteRequest`: Validates incoming data for note creation.
- `NoteResponse`: Provides feedback on the processing outcome.

**Usage**: Used in API endpoints to handle note-related operations.

### Batch Processing

The `BatchRequest` and `BatchResponse` classes support batch operations, enabling efficient processing of multiple notes.

```python
class BatchRequest:
    notes: List[NoteRequest]
    batch_id: str
    priority: str             # high, normal, low
    callback_url: Optional[str]

class BatchResponse:
    batch_id: str
    total_notes: int
    successful: int
    failed: int
    results: List[NoteResponse]
    processing_time: float
```

**Purpose**: To manage and report on batch processing tasks, optimizing throughput and resource utilization.

**Key Attributes**:
- `priority`: Allows prioritization of batch tasks.
- `results`: Summarizes the outcome of batch processing.

**Usage**: Employed in scenarios requiring the processing of multiple notes in a single operation.

## Data Validation Rules

### Content Validation

Content validation rules ensure that notes meet predefined standards, preventing invalid data from entering the system.

- **Minimum content length**: 10 characters
- **Maximum content length**: 10,000 characters
- **Required fields**: `content`, `note_id`
- **Prohibited content**: HTML tags, script tags
- **Language detection**: Automatic language identification

**Purpose**: To maintain data quality and integrity by enforcing content standards.

**Usage**: Applied during the note validation phase to filter out non-compliant data.

### Metadata Validation

Metadata validation rules ensure that metadata adheres to expected formats and constraints.

- **Timestamp format**: ISO 8601
- **UUID format for note_id**
- **Tag format**: Alphanumeric with hyphens
- **Category validation**: Must be from a predefined list
- **Confidence score**: 0.0 to 1.0 range

**Purpose**: To ensure metadata consistency and reliability.

**Usage**: Enforced during metadata processing to validate and correct metadata entries.

### Storage Validation

Storage validation rules protect data integrity and security within the storage system.

- **File size limits**: Configurable per storage type
- **Path validation**: Secure path construction
- **Hash verification**: Content integrity checks
- **Backup validation**: Backup integrity verification

**Purpose**: To safeguard stored data against corruption and unauthorized access.

**Usage**: Implemented during storage operations to verify data integrity and security.

## Data Serialization

### JSON Format

The JSON format provides a standardized way to serialize note data for storage and transmission.

```json
{
  "note_id": "uuid-string",
  "content": "Note content here",
  "title": "Optional title",
  "tags": ["tag1", "tag2"],
  "metadata": {
    "author": "user",
    "source": "manual",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "embedding": [0.1, 0.2, 0.3, ...],
  "category": "research",
  "confidence_score": 0.85
}
```

**Purpose**: To facilitate data exchange and storage in a widely supported format.

**Usage**: Used for API communication and data persistence.

### YAML Format

The YAML format offers a human-readable alternative for data serialization, often used in configuration files.

```yaml
note_id: uuid-string
content: Note content here
title: Optional title
tags:
  - tag1
  - tag2
metadata:
  author: user
  source: manual
  created_at: 2024-01-01T00:00:00Z
embedding: [0.1, 0.2, 0.3]
category: research
confidence_score: 0.85
```

**Purpose**: To provide a clear and concise format for configuration and data representation.

**Usage**: Commonly used in configuration files and documentation.

## Data Migration and Versioning

### Version Control

Version control mechanisms track schema changes and ensure backward compatibility.

- **Schema version tracking**
- **Backward compatibility support**
- **Migration scripts for data updates**
- **Version validation on startup**

**Purpose**: To manage changes to data structures and ensure smooth transitions between versions.

**Usage**: Applied during system updates and migrations to maintain data integrity.

### Data Migration

Data migration processes handle the transition of data between different schema versions.

- **Automatic migration detection**
- **Incremental migration support**
- **Rollback capabilities**
- **Migration logging and monitoring**

**Purpose**: To facilitate seamless data transitions and minimize disruption during updates.

**Usage**: Executed during system upgrades to migrate data to new formats.

---

 