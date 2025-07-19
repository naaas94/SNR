"""QuickCapture parser for symbolic ingestion layer."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..utils.logger import MLLogger
from ..utils.data_validation import DataValidator, ValidationResult


@dataclass
class ParsedNote:
    """Parsed note from QuickCapture grammar."""
    tags: List[str]
    body: str
    comment: Optional[str]
    timestamp: str
    raw_text: str
    valid: bool
    issues: List[str]
    metadata: Dict[str, Any]


class QuickCaptureParser:
    """Parser for QuickCapture controlled grammar."""
    
    def __init__(self, logger: Optional[MLLogger] = None):
        """
        Initialize QuickCapture parser.
        
        Args:
            logger: ML logger instance
        """
        self.logger = logger or MLLogger()
        self.validator = DataValidator(logger=logger)
        
        # Grammar patterns
        self.tag_pattern = r'#([a-zA-Z_][a-zA-Z0-9_]*)'
        self.comment_pattern = r'//(.+)'
        self.timestamp_pattern = r'@(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)'
        
        # Validation rules
        self.grammar_rules = {
            "max_tags": 10,
            "max_body_length": 1000,
            "max_comment_length": 200,
            "required_body": True,
            "allowed_tag_chars": r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        }
    
    def parse_note(self, raw_text: str, auto_timestamp: bool = True) -> ParsedNote:
        """
        Parse a note using QuickCapture grammar.
        
        Grammar format:
        #tag1 #tag2 body text //optional comment @timestamp
        
        Args:
            raw_text: Raw note text
            auto_timestamp: Whether to add timestamp if missing
            
        Returns:
            ParsedNote object with structured data
        """
        issues = []
        metadata = {}
        
        # Extract components
        tags = self._extract_tags(raw_text)
        comment = self._extract_comment(raw_text)
        timestamp = self._extract_timestamp(raw_text)
        body = self._extract_body(raw_text, tags, comment, timestamp)
        
        # Add timestamp if missing and auto_timestamp is enabled
        if not timestamp and auto_timestamp:
            timestamp = datetime.now().isoformat()
            metadata["auto_timestamped"] = True
        
        # Validate parsed components
        validation_result = self._validate_parsed_components(tags, body, comment, timestamp)
        valid = validation_result.is_valid
        issues.extend([error.message for error in validation_result.errors])
        
        # Create parsed note
        parsed_note = ParsedNote(
            tags=tags,
            body=body,
            comment=comment,
            timestamp=timestamp or datetime.now().isoformat(),
            raw_text=raw_text,
            valid=valid,
            issues=issues,
            metadata=metadata
        )
        
        # Log parsing result
        self.logger.log_event(
            event_type="parsing",
            component="quickcapture_parser",
            message=f"Parsed note with {len(tags)} tags",
            metadata={
                "num_tags": len(tags),
                "body_length": len(body),
                "has_comment": comment is not None,
                "valid": valid,
                "issues": issues
            }
        )
        
        return parsed_note
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text."""
        tags = re.findall(self.tag_pattern, text)
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)
        return unique_tags
    
    def _extract_comment(self, text: str) -> Optional[str]:
        """Extract comment from text."""
        comment_match = re.search(self.comment_pattern, text)
        if comment_match:
            return comment_match.group(1).strip()
        return None
    
    def _extract_timestamp(self, text: str) -> Optional[str]:
        """Extract timestamp from text."""
        timestamp_match = re.search(self.timestamp_pattern, text)
        if timestamp_match:
            return timestamp_match.group(1)
        return None
    
    def _extract_body(self, text: str, tags: List[str], comment: Optional[str], 
                     timestamp: Optional[str]) -> str:
        """Extract body text by removing tags, comment, and timestamp."""
        # Remove tags
        for tag in tags:
            text = re.sub(rf'#{re.escape(tag)}\s*', '', text)
        
        # Remove comment
        if comment:
            text = re.sub(rf'//{re.escape(comment)}\s*', '', text)
        
        # Remove timestamp
        if timestamp:
            text = re.sub(rf'@{re.escape(timestamp)}\s*', '', text)
        
        # Clean up whitespace
        body = re.sub(r'\s+', ' ', text).strip()
        return body
    
    def _validate_parsed_components(self, tags: List[str], body: str, 
                                  comment: Optional[str], timestamp: Optional[str]) -> ValidationResult:
        """Validate parsed components against grammar rules."""
        errors = []
        warnings = []
        
        # Validate tags
        if len(tags) > self.grammar_rules["max_tags"]:
            errors.append(f"Too many tags: {len(tags)} > {self.grammar_rules['max_tags']}")
        
        for tag in tags:
            if not re.match(self.grammar_rules["allowed_tag_chars"], tag):
                errors.append(f"Invalid tag format: {tag}")
        
        # Validate body
        if self.grammar_rules["required_body"] and not body:
            errors.append("Body is required but empty")
        
        if len(body) > self.grammar_rules["max_body_length"]:
            warnings.append(f"Body too long: {len(body)} > {self.grammar_rules['max_body_length']}")
        
        # Validate comment
        if comment and len(comment) > self.grammar_rules["max_comment_length"]:
            warnings.append(f"Comment too long: {len(comment)} > {self.grammar_rules['max_comment_length']}")
        
        # Validate timestamp
        if timestamp:
            try:
                datetime.fromisoformat(timestamp)
            except ValueError:
                errors.append(f"Invalid timestamp format: {timestamp}")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data_quality_score=quality_score
        )
    
    def _calculate_quality_score(self, errors: List[str], warnings: List[str]) -> float:
        """Calculate quality score for parsed note."""
        if not errors and not warnings:
            return 1.0
        
        # Weight errors more heavily than warnings
        weighted_issues = len(errors) * 2 + len(warnings)
        score = max(0.0, 1.0 - (weighted_issues / 10.0))
        return round(score, 3)
    
    def parse_batch(self, notes: List[str], auto_timestamp: bool = True) -> List[ParsedNote]:
        """Parse a batch of notes."""
        parsed_notes = []
        
        for i, note_text in enumerate(notes):
            try:
                parsed_note = self.parse_note(note_text, auto_timestamp)
                parsed_notes.append(parsed_note)
            except Exception as e:
                self.logger.log_error(
                    component="quickcapture_parser",
                    error_message=f"Failed to parse note {i}: {e}",
                    error_type="parsing_error",
                    context={"note_index": i, "note_text": note_text[:100]}
                )
                # Create invalid note for failed parsing
                failed_note = ParsedNote(
                    tags=[],
                    body=note_text,
                    comment=None,
                    timestamp=datetime.now().isoformat(),
                    raw_text=note_text,
                    valid=False,
                    issues=[f"Parsing failed: {str(e)}"],
                    metadata={"parsing_failed": True}
                )
                parsed_notes.append(failed_note)
        
        return parsed_notes
    
    def to_dict(self, parsed_note: ParsedNote) -> Dict[str, Any]:
        """Convert ParsedNote to dictionary for JSON serialization."""
        return {
            "tags": parsed_note.tags,
            "body": parsed_note.body,
            "comment": parsed_note.comment,
            "timestamp": parsed_note.timestamp,
            "raw_text": parsed_note.raw_text,
            "valid": parsed_note.valid,
            "issues": parsed_note.issues,
            "metadata": parsed_note.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> ParsedNote:
        """Create ParsedNote from dictionary."""
        return ParsedNote(
            tags=data.get("tags", []),
            body=data.get("body", ""),
            comment=data.get("comment"),
            timestamp=data.get("timestamp", ""),
            raw_text=data.get("raw_text", ""),
            valid=data.get("valid", False),
            issues=data.get("issues", []),
            metadata=data.get("metadata", {})
        )


class JSONLWriter:
    """Writer for distributed JSONL files."""
    
    def __init__(self, output_dir: str, logger: Optional[MLLogger] = None):
        """
        Initialize JSONL writer.
        
        Args:
            output_dir: Directory for JSONL files
            logger: ML logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or MLLogger()
        
        # Track file handles for efficient writing
        self.file_handles: Dict[str, Any] = {}
    
    def write_note(self, parsed_note: ParsedNote) -> None:
        """
        Write note to tag-specific JSONL files.
        
        Args:
            parsed_note: Parsed note to write
        """
        # Write to each tag's file
        for tag in parsed_note.tags:
            self._write_to_tag_file(tag, parsed_note)
        
        # Write to general file if no tags
        if not parsed_note.tags:
            self._write_to_tag_file("untagged", parsed_note)
    
    def _write_to_tag_file(self, tag: str, parsed_note: ParsedNote) -> None:
        """Write note to specific tag's JSONL file."""
        file_path = self.output_dir / f"{tag}.jsonl"
        
        # Get or create file handle
        if tag not in self.file_handles:
            self.file_handles[tag] = open(file_path, 'a', encoding='utf-8')
        
        # Prepare note data for this tag
        note_data = {
            "tag": tag,
            "body": parsed_note.body,
            "comment": parsed_note.comment,
            "timestamp": parsed_note.timestamp,
            "valid": parsed_note.valid,
            "issues": parsed_note.issues,
            "metadata": parsed_note.metadata,
            "all_tags": parsed_note.tags  # Include all tags for context
        }
        
        # Write to file
        self.file_handles[tag].write(json.dumps(note_data) + '\n')
        self.file_handles[tag].flush()  # Ensure immediate write
        
        self.logger.log_event(
            event_type="writing",
            component="jsonl_writer",
            message=f"Wrote note to {tag}.jsonl",
            metadata={
                "tag": tag,
                "file_path": str(file_path),
                "valid": parsed_note.valid
            }
        )
    
    def close(self) -> None:
        """Close all file handles."""
        for handle in self.file_handles.values():
            handle.close()
        self.file_handles.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class QuickCaptureProcessor:
    """Complete QuickCapture processing pipeline."""
    
    def __init__(self, output_dir: str, logger: Optional[MLLogger] = None):
        """
        Initialize QuickCapture processor.
        
        Args:
            output_dir: Directory for JSONL files
            logger: ML logger instance
        """
        self.parser = QuickCaptureParser(logger)
        self.writer = JSONLWriter(output_dir, logger)
        self.logger = logger or MLLogger()
    
    def process_note(self, raw_text: str, auto_timestamp: bool = True) -> ParsedNote:
        """
        Process a single note through the complete pipeline.
        
        Args:
            raw_text: Raw note text
            auto_timestamp: Whether to add timestamp if missing
            
        Returns:
            ParsedNote object
        """
        # Parse note
        parsed_note = self.parser.parse_note(raw_text, auto_timestamp)
        
        # Write to JSONL files
        self.writer.write_note(parsed_note)
        
        return parsed_note
    
    def process_batch(self, notes: List[str], auto_timestamp: bool = True) -> List[ParsedNote]:
        """
        Process a batch of notes.
        
        Args:
            notes: List of raw note texts
            auto_timestamp: Whether to add timestamps if missing
            
        Returns:
            List of ParsedNote objects
        """
        parsed_notes = self.parser.parse_batch(notes, auto_timestamp)
        
        for parsed_note in parsed_notes:
            self.writer.write_note(parsed_note)
        
        return parsed_notes
    
    def close(self) -> None:
        """Close the processor and all file handles."""
        self.writer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 