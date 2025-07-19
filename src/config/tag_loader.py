"""Load and validate tag ontologies from YAML files."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TagDefinition:
    """Represents a semantic tag with its metadata."""
    tag: str
    description: str
    examples: List[str]
    confidence_threshold: float
    parent_tag: Optional[str] = None


class TagLoader:
    """Loads and validates tag ontologies from YAML files."""
    
    def __init__(self, tags_file: Optional[str] = None):
        """Initialize with optional custom tags file path."""
        self.tags_file = tags_file or "tags/default_tags.yaml"
        
    def load_tags(self) -> List[TagDefinition]:
        """Load tags from YAML file and return as TagDefinition objects."""
        tags_path = Path(self.tags_file)
        
        if not tags_path.exists():
            raise FileNotFoundError(f"Tags file not found: {tags_path}")
            
        with open(tags_path, 'r', encoding='utf-8') as f:
            tags_data = yaml.safe_load(f)
            
        tags = []
        for tag_data in tags_data:
            tag = TagDefinition(
                tag=tag_data['tag'],
                description=tag_data['description'].strip(),
                examples=tag_data['examples'],
                confidence_threshold=tag_data['confidence_threshold'],
                parent_tag=tag_data.get('parent_tag')
            )
            tags.append(tag)
            
        return tags
    
    def get_tag_by_name(self, tag_name: str) -> Optional[TagDefinition]:
        """Get a specific tag by name."""
        tags = self.load_tags()
        for tag in tags:
            if tag.tag == tag_name:
                return tag
        return None
    
    def get_all_tag_names(self) -> List[str]:
        """Get list of all tag names."""
        tags = self.load_tags()
        return [tag.tag for tag in tags]
    
    def validate_tags(self) -> List[str]:
        """Validate tag definitions and return list of errors."""
        errors = []
        tags = self.load_tags()
        
        for tag in tags:
            if not tag.tag:
                errors.append(f"Tag name is empty")
            if not tag.description:
                errors.append(f"Tag '{tag.tag}' has empty description")
            if not tag.examples:
                errors.append(f"Tag '{tag.tag}' has no examples")
            if tag.confidence_threshold < 0 or tag.confidence_threshold > 1:
                errors.append(f"Tag '{tag.tag}' has invalid confidence threshold: {tag.confidence_threshold}")
                
        return errors 