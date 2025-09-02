"""Tests for memory core schema validation and initialization."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock
import logging

from memory.memory_core_schema import (
    MemoryCoreSchema,
    MemoryCoreValidator,
    MemoryCoreInitializer,
    ValidationError,
    load_memory_core_config
)


class TestMemoryCoreSchema:
    """Test the memory core schema constants."""
    
    def test_schema_constants(self):
        """Test that schema defines required constants."""
        schema = MemoryCoreSchema()
        
        assert schema.CONFIG_FILE == "config.json"
        assert schema.ASSETS_DIR == "assets"
        assert schema.IMG_DIR == "img"
        assert schema.VOICE_DIR == "voice"
        assert schema.DATA_DIR == "data"
        assert schema.DEFAULT_GRAPH_FILE == "graph.json"
        assert schema.DEFAULT_INDEX_FILE == "emb_index.json"
        assert schema.DEFAULT_SCHEMA_FILE == "schema.json"


class TestMemoryCoreValidator:
    """Test memory core validation."""
    
    @pytest.fixture
    def validator(self):
        return MemoryCoreValidator()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    def test_validate_structure_success(self, validator):
        """Test successful structure validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create proper structure
            core_path = Path(temp_dir)
            (core_path / "config.json").write_text("{}")
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            errors = validator.validate_structure(str(core_path))
            assert len(errors) == 0
    
    def test_validate_structure_missing_config(self, validator):
        """Test validation with missing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            errors = validator.validate_structure(temp_dir)
            assert any("config file missing" in error.lower() for error in errors)
    
    def test_validate_structure_missing_assets(self, validator):
        """Test validation with missing assets directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "config.json").write_text("{}")
            
            errors = validator.validate_structure(temp_dir)
            assert any("assets directory missing" in error.lower() for error in errors)
    
    def test_validate_structure_missing_img_dir(self, validator):
        """Test validation with missing img directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            core_path = Path(temp_dir)
            (core_path / "config.json").write_text("{}")
            (core_path / "assets").mkdir()
            (core_path / "assets" / "voice").mkdir()
            
            errors = validator.validate_structure(temp_dir)
            assert any("img directory missing" in error.lower() for error in errors)
    
    def test_validate_structure_creates_data_dir(self, validator):
        """Test that validation creates missing data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            core_path = Path(temp_dir)
            (core_path / "config.json").write_text("{}")
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            errors = validator.validate_structure(temp_dir)
            
            # Should have no errors and data directory should be created
            assert len(errors) == 0
            assert (core_path / "data").exists()
            assert (core_path / "data").is_dir()
    
    def test_validate_structure_nonexistent_path(self, validator):
        """Test validation with nonexistent path."""
        errors = validator.validate_structure("/nonexistent/path")
        assert any("does not exist" in error for error in errors)
    
    def test_validate_config_success(self, validator):
        """Test successful config validation."""
        config = {
            "embedding": {
                "provider": "local",
                "api_key": "test_key",
                "model": "text-embedding-3-small",
                "dim": 1536
            },
            "files": {
                "graph_file": "graph.json",
                "index_file": "emb_index.json"
            }
        }
        
        errors = validator.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_missing_sections(self, validator):
        """Test config validation with missing sections."""
        config = {}
        
        errors = validator.validate_config(config)
        assert any("embedding" in error for error in errors)
        assert any("files" in error for error in errors)
    
    def test_validate_config_missing_embedding_fields(self, validator):
        """Test config validation with missing embedding fields."""
        config = {
            "embedding": {
                "provider": "local"
                # Missing api_key, model, dim
            },
            "files": {}
        }
        
        errors = validator.validate_config(config)
        assert any("api_key" in error for error in errors)
        assert any("model" in error for error in errors)
        assert any("dim" in error for error in errors)
    
    def test_validate_config_remote_provider_missing_fields(self, validator):
        """Test remote provider validation with missing fields."""
        config = {
            "embedding": {
                "provider": "remote",
                "api_key": "test_key",
                "model": "text-embedding-3-small",
                "dim": 1536
                # Missing pinecone_api_key, pinecone_index_name
            },
            "files": {}
        }
        
        errors = validator.validate_config(config)
        assert any("pinecone_api_key" in error for error in errors)
        assert any("pinecone_index_name" in error for error in errors)
    
    def test_validate_config_invalid_provider(self, validator):
        """Test config validation with invalid provider."""
        config = {
            "embedding": {
                "provider": "invalid_provider",
                "api_key": "test_key",
                "model": "text-embedding-3-small",
                "dim": 1536
            },
            "files": {}
        }
        
        errors = validator.validate_config(config)
        assert any("Invalid embedding provider" in error for error in errors)
    
    def test_validate_memory_core_full_success(self, validator):
        """Test full memory core validation (structure + config)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create proper structure
            core_path = Path(temp_dir)
            
            config = {
                "embedding": {
                    "provider": "local",
                    "api_key": "test_key",
                    "model": "text-embedding-3-small",
                    "dim": 1536
                },
                "files": {
                    "graph_file": "graph.json",
                    "index_file": "emb_index.json"
                }
            }
            
            with open(core_path / "config.json", 'w') as f:
                json.dump(config, f)
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            errors = validator.validate_memory_core(str(core_path))
            assert len(errors) == 0
    
    def test_validate_memory_core_invalid_json(self, validator):
        """Test validation with invalid JSON in config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            core_path = Path(temp_dir)
            
            # Write invalid JSON
            (core_path / "config.json").write_text("invalid json content")
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            errors = validator.validate_memory_core(str(core_path))
            assert any("Invalid JSON" in error for error in errors)


class TestMemoryCoreInitializer:
    """Test memory core initialization."""
    
    @pytest.fixture
    def initializer(self):
        return MemoryCoreInitializer()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    def test_create_memory_core_structure(self, initializer):
        """Test creating memory core directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_core_path = str(Path(temp_dir) / "test_core")
            
            initializer.create_memory_core_structure(memory_core_path)
            
            core_path = Path(memory_core_path)
            assert core_path.exists()
            assert (core_path / "assets").exists()
            assert (core_path / "assets" / "img").exists()
            assert (core_path / "assets" / "voice").exists()
            assert (core_path / "data").exists()
    
    def test_create_default_config_local(self, initializer):
        """Test creating default config for local provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initializer.create_default_config(
                temp_dir, "test_key", "local"
            )
            
            config_path = Path(temp_dir) / "config.json"
            assert config_path.exists()
            
            with open(config_path) as f:
                config = json.load(f)
            
            assert config["embedding"]["provider"] == "local"
            assert config["embedding"]["api_key"] == "test_key"
            assert "pinecone_api_key" not in config["embedding"]
    
    def test_create_default_config_remote(self, initializer):
        """Test creating default config for remote provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initializer.create_default_config(
                temp_dir, "test_key", "remote",
                pinecone_api_key="pinecone_key",
                pinecone_index_name="test_index"
            )
            
            config_path = Path(temp_dir) / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            
            assert config["embedding"]["provider"] == "remote"
            assert config["embedding"]["pinecone_api_key"] == "pinecone_key"
            assert config["embedding"]["pinecone_index_name"] == "test_index"
    
    def test_create_default_config_remote_missing_fields(self, initializer):
        """Test error when remote provider is missing required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Remote provider requires"):
                initializer.create_default_config(
                    temp_dir, "test_key", "remote"
                    # Missing pinecone_api_key and pinecone_index_name
                )
    
    def test_initialize_memory_core_success(self, initializer):
        """Test full memory core initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_core_path = str(Path(temp_dir) / "test_core")
            
            initializer.initialize_memory_core(
                memory_core_path, "test_key", "local"
            )
            
            # Verify structure was created
            core_path = Path(memory_core_path)
            assert core_path.exists()
            assert (core_path / "config.json").exists()
            assert (core_path / "assets" / "img").exists()
            assert (core_path / "assets" / "voice").exists()
            assert (core_path / "data").exists()
            
            # Verify config is valid
            with open(core_path / "config.json") as f:
                config = json.load(f)
            assert config["embedding"]["provider"] == "local"
    
    def test_initialize_memory_core_validation_failure(self, initializer):
        """Test initialization with validation enabled and failure."""
        # Create a temporary file path that already exists as a file (not directory)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Try to initialize memory core at file location (should fail with ValueError)
            with pytest.raises((ValidationError, ValueError)):
                initializer.initialize_memory_core(
                    temp_file_path, "test_key", validate=True
                )
        finally:
            # Clean up the temporary file
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestLoadMemoryCoreConfig:
    """Test the load_memory_core_config utility function."""
    
    def test_load_config_success(self):
        """Test successfully loading a valid config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initializer = MemoryCoreInitializer()
            initializer.initialize_memory_core(temp_dir, "test_key")
            
            config = load_memory_core_config(temp_dir)
            
            assert config["embedding"]["provider"] == "local"
            assert config["embedding"]["api_key"] == "test_key"
    
    def test_load_config_structure_error(self):
        """Test loading config with structure errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create proper structure
            with pytest.raises(ValidationError, match="structure validation failed"):
                load_memory_core_config(temp_dir)
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create proper structure
            core_path = Path(temp_dir)
            (core_path / "config.json").write_text("invalid json")
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            with pytest.raises(ValidationError, match="Invalid JSON"):
                load_memory_core_config(temp_dir)
    
    def test_load_config_validation_error(self):
        """Test loading config with validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create proper structure
            core_path = Path(temp_dir)
            
            # Create invalid config
            invalid_config = {"invalid": "config"}
            with open(core_path / "config.json", 'w') as f:
                json.dump(invalid_config, f)
            
            assets_dir = core_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "img").mkdir()
            (assets_dir / "voice").mkdir()
            
            with pytest.raises(ValidationError, match="config validation failed"):
                load_memory_core_config(temp_dir)