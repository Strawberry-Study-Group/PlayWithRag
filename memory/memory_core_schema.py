"""Memory core schema validation and standardization module."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


class MemoryCoreError(Exception):
    """Base exception for memory core operations."""
    pass


class ValidationError(MemoryCoreError):
    """Raised when memory core validation fails."""
    pass


@dataclass
class MemoryCoreSchema:
    """Standard schema for memory core directory structure."""
    
    # Required files
    CONFIG_FILE = "config.json"
    
    # Required directories
    ASSETS_DIR = "assets"
    IMG_DIR = "img"
    VOICE_DIR = "voice"
    DATA_DIR = "data"
    
    # Default file names (can be overridden in config)
    DEFAULT_GRAPH_FILE = "graph.json"
    DEFAULT_INDEX_FILE = "emb_index.json"
    DEFAULT_SCHEMA_FILE = "schema.json"
    
    # Standard memory core structure:
    # memory_core/
    # ├── config.json                 (required configuration)
    # ├── assets/                     (required asset directory)
    # │   ├── img/                   (images subdirectory)
    # │   ├── voice/                 (voice/audio subdirectory)
    # │   └── [other asset types]    (extensible)
    # ├── data/                      (optional data directory)
    # │   ├── graph.json            (concept graph data)
    # │   ├── emb_index.json        (embeddings index)
    # │   └── schema.json           (optional graph schema)
    # └── [custom directories]       (extensible)


class IMemoryCoreValidator(ABC):
    """Interface for memory core validators."""
    
    @abstractmethod
    def validate_structure(self, memory_core_path: str) -> List[str]:
        """Validate directory structure. Returns list of errors."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration. Returns list of errors."""
        pass


class MemoryCoreValidator(IMemoryCoreValidator):
    """Standard memory core validator."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.schema = MemoryCoreSchema()
    
    def validate_structure(self, memory_core_path: str) -> List[str]:
        """Validate memory core directory structure."""
        errors = []
        core_path = Path(memory_core_path)
        
        # Check if memory core path exists
        if not core_path.exists():
            errors.append(f"Memory core path does not exist: {memory_core_path}")
            return errors
        
        if not core_path.is_dir():
            errors.append(f"Memory core path is not a directory: {memory_core_path}")
            return errors
        
        # Check required config file
        config_file = core_path / self.schema.CONFIG_FILE
        if not config_file.exists():
            errors.append(f"Required config file missing: {self.schema.CONFIG_FILE}")
        elif not config_file.is_file():
            errors.append(f"Config path is not a file: {self.schema.CONFIG_FILE}")
        
        # Check required assets directory
        assets_dir = core_path / self.schema.ASSETS_DIR
        if not assets_dir.exists():
            errors.append(f"Required assets directory missing: {self.schema.ASSETS_DIR}")
        elif not assets_dir.is_dir():
            errors.append(f"Assets path is not a directory: {self.schema.ASSETS_DIR}")
        else:
            # Check required asset subdirectories
            img_dir = assets_dir / self.schema.IMG_DIR
            if not img_dir.exists():
                errors.append(f"Required img directory missing: {self.schema.ASSETS_DIR}/{self.schema.IMG_DIR}")
            elif not img_dir.is_dir():
                errors.append(f"Img path is not a directory: {self.schema.ASSETS_DIR}/{self.schema.IMG_DIR}")
            
            voice_dir = assets_dir / self.schema.VOICE_DIR
            if not voice_dir.exists():
                errors.append(f"Required voice directory missing: {self.schema.ASSETS_DIR}/{self.schema.VOICE_DIR}")
            elif not voice_dir.is_dir():
                errors.append(f"Voice path is not a directory: {self.schema.ASSETS_DIR}/{self.schema.VOICE_DIR}")
        
        # Check optional data directory (create if it doesn't exist)
        data_dir = core_path / self.schema.DATA_DIR
        if not data_dir.exists():
            self.logger.info(f"Creating optional data directory: {data_dir}")
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to create data directory: {e}")
        elif not data_dir.is_dir():
            errors.append(f"Data path is not a directory: {self.schema.DATA_DIR}")
        
        return errors
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate memory core configuration structure."""
        errors = []
        
        # Check required top-level sections
        required_sections = ["embedding", "files"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Required config section missing: {section}")
        
        # Validate embedding configuration
        if "embedding" in config:
            embedding_config = config["embedding"]
            required_embedding_fields = ["provider", "api_key", "model", "dim"]
            
            for field in required_embedding_fields:
                if field not in embedding_config:
                    errors.append(f"Required embedding config field missing: {field}")
            
            # Provider-specific validation
            provider = embedding_config.get("provider")
            if provider == "remote":
                remote_fields = ["pinecone_api_key", "pinecone_index_name"]
                for field in remote_fields:
                    if field not in embedding_config:
                        errors.append(f"Required remote embedding config field missing: {field}")
            elif provider == "local":
                # Local provider is valid with just the basic fields
                pass
            elif provider is not None:
                errors.append(f"Invalid embedding provider: {provider}. Must be 'local' or 'remote'")
        
        # Validate files configuration
        if "files" in config:
            files_config = config["files"]
            # Files config is mostly optional with defaults, just check types
            if not isinstance(files_config, dict):
                errors.append("Files config must be a dictionary")
        
        return errors
    
    def validate_memory_core(self, memory_core_path: str) -> List[str]:
        """Validate complete memory core (structure + config)."""
        all_errors = []
        
        # Validate structure first
        structure_errors = self.validate_structure(memory_core_path)
        all_errors.extend(structure_errors)
        
        # If structure is valid, validate config
        if not structure_errors:
            try:
                config_path = Path(memory_core_path) / self.schema.CONFIG_FILE
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                config_errors = self.validate_config(config)
                all_errors.extend(config_errors)
            except json.JSONDecodeError as e:
                all_errors.append(f"Invalid JSON in config file: {e}")
            except Exception as e:
                all_errors.append(f"Failed to read config file: {e}")
        
        return all_errors


class MemoryCoreInitializer:
    """Utility for initializing new memory cores with proper structure."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.schema = MemoryCoreSchema()
        self.validator = MemoryCoreValidator(logger)
    
    def create_memory_core_structure(self, memory_core_path: str) -> None:
        """Create the standard memory core directory structure."""
        core_path = Path(memory_core_path)
        
        # Check if path exists as file first
        if core_path.exists() and not core_path.is_dir():
            raise ValueError(f"Path exists as file, not directory: {memory_core_path}")
        
        # Create main directory
        core_path.mkdir(parents=True, exist_ok=True)
        
        # Create assets directory and subdirectories
        assets_dir = core_path / self.schema.ASSETS_DIR
        assets_dir.mkdir(exist_ok=True)
        
        (assets_dir / self.schema.IMG_DIR).mkdir(exist_ok=True)
        (assets_dir / self.schema.VOICE_DIR).mkdir(exist_ok=True)
        
        # Create optional data directory
        data_dir = core_path / self.schema.DATA_DIR
        data_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Created memory core structure at: {memory_core_path}")
    
    def create_default_config(self, 
                            memory_core_path: str,
                            api_key: str,
                            provider: str = "local",
                            model: str = "text-embedding-3-small",
                            dim: int = 1536,
                            pinecone_api_key: Optional[str] = None,
                            pinecone_index_name: Optional[str] = None) -> None:
        """Create a default config.json file."""
        
        config = {
            "embedding": {
                "provider": provider,
                "api_key": api_key,
                "model": model,
                "dim": dim
            },
            "files": {
                "graph_file": self.schema.DEFAULT_GRAPH_FILE,
                "index_file": self.schema.DEFAULT_INDEX_FILE
            }
        }
        
        # Add remote-specific fields if needed
        if provider == "remote":
            if not pinecone_api_key or not pinecone_index_name:
                raise ValueError("Remote provider requires pinecone_api_key and pinecone_index_name")
            
            config["embedding"]["pinecone_api_key"] = pinecone_api_key
            config["embedding"]["pinecone_index_name"] = pinecone_index_name
            config["embedding"]["metric"] = "cosine"
        
        # Write config file
        config_path = Path(memory_core_path) / self.schema.CONFIG_FILE
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created default config at: {config_path}")
    
    def initialize_memory_core(self,
                             memory_core_path: str,
                             api_key: str,
                             provider: str = "local",
                             model: str = "text-embedding-3-small",
                             dim: int = 1536,
                             pinecone_api_key: Optional[str] = None,
                             pinecone_index_name: Optional[str] = None,
                             validate: bool = True) -> None:
        """Initialize a complete memory core with structure and config."""
        
        # Create directory structure
        self.create_memory_core_structure(memory_core_path)
        
        # Create default config
        self.create_default_config(
            memory_core_path, api_key, provider, model, dim,
            pinecone_api_key, pinecone_index_name
        )
        
        # Validate if requested
        if validate:
            errors = self.validator.validate_memory_core(memory_core_path)
            if errors:
                raise ValidationError(f"Memory core validation failed: {'; '.join(errors)}")
        
        self.logger.info(f"Successfully initialized memory core at: {memory_core_path}")


def load_memory_core_config(memory_core_path: str) -> Dict[str, Any]:
    """Load and validate memory core configuration."""
    validator = MemoryCoreValidator()
    
    # Validate structure first
    structure_errors = validator.validate_structure(memory_core_path)
    if structure_errors:
        raise ValidationError(f"Memory core structure validation failed: {'; '.join(structure_errors)}")
    
    # Load and validate config
    config_path = Path(memory_core_path) / MemoryCoreSchema.CONFIG_FILE
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to read config file: {e}")
    
    config_errors = validator.validate_config(config)
    if config_errors:
        raise ValidationError(f"Memory core config validation failed: {'; '.join(config_errors)}")
    
    return config