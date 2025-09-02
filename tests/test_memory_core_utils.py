"""Utility functions for creating and managing test memory cores."""

import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from memory.memory_core_schema import MemoryCoreInitializer, MemoryCoreSchema


def create_test_memory_core(
    temp_dir: Optional[str] = None,
    api_key: str = "test_api_key",
    provider: str = "local",
    model: str = "text-embedding-3-small",
    dim: int = 1536,
    pinecone_api_key: Optional[str] = None,
    pinecone_index_name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> str:
    """Create a test memory core with proper structure and configuration.
    
    Args:
        temp_dir: Parent directory for memory core (if None, creates temporary directory)
        api_key: OpenAI API key for embeddings
        provider: Embedding provider ("local" or "remote")
        model: Embedding model name
        dim: Embedding dimensions
        pinecone_api_key: Pinecone API key (required for remote provider)
        pinecone_index_name: Pinecone index name (required for remote provider)
        custom_config: Optional custom config to override defaults
        
    Returns:
        Path to the created memory core directory
    """
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="test_memory_core_")
    else:
        memory_core_path = str(Path(temp_dir) / "test_memory_core")
        Path(memory_core_path).mkdir(parents=True, exist_ok=True)
        temp_dir = memory_core_path
    
    # Initialize memory core with proper structure
    initializer = MemoryCoreInitializer()
    
    if custom_config:
        # Create structure manually and write custom config
        initializer.create_memory_core_structure(temp_dir)
        config_path = Path(temp_dir) / MemoryCoreSchema.CONFIG_FILE
        with open(config_path, 'w') as f:
            json.dump(custom_config, f, indent=2)
    else:
        # Use standard initialization
        initializer.initialize_memory_core(
            temp_dir, api_key, provider, model, dim,
            pinecone_api_key, pinecone_index_name,
            validate=True
        )
    
    return temp_dir


def cleanup_test_memory_core(memory_core_path: str) -> None:
    """Clean up a test memory core directory."""
    if Path(memory_core_path).exists():
        shutil.rmtree(memory_core_path)


def create_memory_core_config(
    provider: str = "local",
    api_key: str = "test_api_key",
    model: str = "text-embedding-3-small",
    dim: int = 1536,
    pinecone_api_key: Optional[str] = None,
    pinecone_index_name: Optional[str] = None,
    graph_file: str = "graph.json",
    index_file: str = "emb_index.json",
    schema_file: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standard memory core configuration dictionary.
    
    Args:
        provider: Embedding provider ("local" or "remote")
        api_key: OpenAI API key
        model: Embedding model name
        dim: Embedding dimensions
        pinecone_api_key: Pinecone API key (required for remote)
        pinecone_index_name: Pinecone index name (required for remote)
        graph_file: Graph file name
        index_file: Embedding index file name
        schema_file: Optional schema file name
        
    Returns:
        Dictionary with memory core configuration
    """
    config = {
        "embedding": {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "dim": dim
        },
        "files": {
            "graph_file": graph_file,
            "index_file": index_file
        }
    }
    
    if schema_file:
        config["files"]["schema_file"] = schema_file
    
    if provider == "remote":
        if not pinecone_api_key or not pinecone_index_name:
            raise ValueError("Remote provider requires pinecone_api_key and pinecone_index_name")
        config["embedding"]["pinecone_api_key"] = pinecone_api_key
        config["embedding"]["pinecone_index_name"] = pinecone_index_name
        config["embedding"]["metric"] = "cosine"
    
    return config


class MemoryCoreTestContext:
    """Context manager for test memory cores."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.memory_core_path = None
    
    def __enter__(self) -> str:
        self.memory_core_path = create_test_memory_core(**self.kwargs)
        return self.memory_core_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.memory_core_path:
            cleanup_test_memory_core(self.memory_core_path)


def verify_memory_core_structure(memory_core_path: str) -> bool:
    """Verify that a memory core has the correct directory structure.
    
    Args:
        memory_core_path: Path to memory core directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    from memory.memory_core_schema import MemoryCoreValidator
    
    validator = MemoryCoreValidator()
    errors = validator.validate_structure(memory_core_path)
    return len(errors) == 0


def add_test_files_to_memory_core(memory_core_path: str) -> None:
    """Add some test files to a memory core for testing."""
    schema = MemoryCoreSchema()
    core_path = Path(memory_core_path)
    
    # Add test image to assets/img
    img_dir = core_path / schema.ASSETS_DIR / schema.IMG_DIR
    test_img_path = img_dir / "test_image.jpg"
    test_img_path.write_text("fake image content")
    
    # Add test audio to assets/voice
    voice_dir = core_path / schema.ASSETS_DIR / schema.VOICE_DIR
    test_audio_path = voice_dir / "test_audio.mp3"
    test_audio_path.write_text("fake audio content")
    
    # Add test data files
    data_dir = core_path / schema.DATA_DIR
    test_graph_path = data_dir / "graph.json"
    test_graph_path.write_text('{"node_dict": {}, "edge_dict": {}}')
    
    test_index_path = data_dir / "emb_index.json"
    test_index_path.write_text('{"embeddings": {}}')


def create_invalid_memory_core(temp_dir: Optional[str] = None) -> str:
    """Create an invalid memory core for testing validation failures.
    
    Args:
        temp_dir: Parent directory for memory core
        
    Returns:
        Path to the invalid memory core directory
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="invalid_memory_core_")
    
    # Create directory without proper structure (missing required directories)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Create invalid config.json
    invalid_config = {
        "invalid_section": {
            "missing_required_fields": True
        }
    }
    
    config_path = Path(temp_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(invalid_config, f)
    
    return temp_dir