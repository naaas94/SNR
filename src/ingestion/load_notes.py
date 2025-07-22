"""Load and preprocess notes from various file formats."""

import pandas as pd
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Note:
    """Represents a single note with metadata."""
    note_id: str
    text: str
    timestamp: str
    source_file: str
    original_text: Optional[str] = None


class NoteLoader:
    """Loads notes from various file formats and normalizes them."""
    
    def __init__(self):
        """Initialize the note loader."""
        pass
        
    def load_from_csv(self, file_path: str, text_column: str = "text", 
                     id_column: Optional[str] = None, 
                     timestamp_column: Optional[str] = None) -> List[Note]:
        """Load notes from a CSV file with enhanced validation and error handling."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return []
        
        if text_column not in df.columns:
            print(f"Text column '{text_column}' not found in CSV")
            return []
        
        notes = []
        for idx, row in df.iterrows():
            # Generate note ID
            note_id = row.get(id_column) if id_column and id_column in df.columns else str(uuid.uuid4())
            
            # Get timestamp
            if timestamp_column and timestamp_column in df.columns:
                timestamp = str(row[timestamp_column])
            else:
                timestamp = datetime.now().isoformat()
                
            # Get text and normalize
            text = str(row[text_column]).strip()
            if not text:
                continue
                
            note = Note(
                note_id=str(note_id),
                text=self._normalize_text(text),
                timestamp=timestamp,
                source_file=file_path,
                original_text=text
            )
            notes.append(note)
            
        return notes
    
    def load_from_txt(self, file_path: str, delimiter: str = "\n") -> List[Note]:
        """Load notes from a text file, splitting by delimiter with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return []
            
        lines = [line.strip() for line in content.split(delimiter) if line.strip()]
        
        notes = []
        for line in lines:
            note = Note(
                note_id=str(uuid.uuid4()),
                text=self._normalize_text(line),
                timestamp=datetime.now().isoformat(),
                source_file=file_path,
                original_text=line
            )
            notes.append(note)
            
        return notes
    
    def load_from_markdown(self, file_path: str) -> List[Note]:
        """Load notes from a markdown file, treating each paragraph as a note with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading markdown file {file_path}: {e}")
            return []
            
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        notes = []
        for paragraph in paragraphs:
            # Skip headers and other markdown elements
            if paragraph.startswith('#'):
                continue
                
            note = Note(
                note_id=str(uuid.uuid4()),
                text=self._normalize_text(paragraph),
                timestamp=datetime.now().isoformat(),
                source_file=file_path,
                original_text=paragraph
            )
            notes.append(note)
            
        return notes
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove common punctuation that might interfere with semantic meaning
        # Keep periods, commas, and question marks as they carry semantic weight
        text = text.replace('"', '').replace("'", '').replace('!', '.')
        
        return text.strip()
    
    def deduplicate_notes(self, notes: List[Note]) -> List[Note]:
        """Remove duplicate notes based on normalized text."""
        seen_texts = set()
        unique_notes = []
        
        for note in notes:
            if note.text not in seen_texts:
                seen_texts.add(note.text)
                unique_notes.append(note)
                
        return unique_notes
    
    def filter_notes_by_length(self, notes: List[Note], 
                              min_length: int = 3, 
                              max_length: int = 500) -> List[Note]:
        """Filter notes by text length."""
        filtered_notes = []
        
        for note in notes:
            word_count = len(note.text.split())
            if min_length <= word_count <= max_length:
                filtered_notes.append(note)
                
        return filtered_notes 