import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch
import logging
from datetime import datetime
from typing import Dict, Any, List

from concept_graph.graph_store import (
    GraphStore,
    GraphStoreFactory,
    SchemaValidator,
    GraphStoreError,
    ValidationError,
    NodeNotFoundError,
    EdgeNotFoundError,
    DuplicateError,
    SchemaError,
    IGraphValidator,
    IGraphStore
)
from concept_graph.file_store import IFileStore


class MockFileStore(IFileStore):
    """Mock file store for testing."""
    
    def __init__(self):
        self.files = {}
        self.prefix_path = Path("/mock/path")
    
    def add_file(self, local_file_path: str, remote_file_name: str) -> None:
        with open(local_file_path, 'r') as f:
            self.files[remote_file_name] = f.read()
    
    def get_file(self, remote_file_name: str, local_file_path: str) -> None:
        if remote_file_name in self.files:
            with open(local_file_path, 'w') as f:
                f.write(self.files[remote_file_name])
        else:
            raise FileNotFoundError(f"File {remote_file_name} not found")
    
    def delete_file(self, remote_file_name: str) -> None:
        if remote_file_name in self.files:
            del self.files[remote_file_name]
        else:
            raise FileNotFoundError(f"File {remote_file_name} not found")
    
    def update_file(self, local_file_path: str, remote_file_name: str) -> None:
        self.add_file(local_file_path, remote_file_name)
    
    def file_exists(self, remote_file_name: str) -> bool:
        return remote_file_name in self.files
    
    def save_img_from_url(self, url: str, file_name: str) -> str:
        pass


class MockValidator(IGraphValidator):
    """Mock validator for testing."""
    
    def __init__(self, should_validate: bool = True):
        self.should_validate = should_validate
        self.validate_node_calls = []
        self.validate_edge_calls = []
    
    def validate_node(self, node: Dict[str, Any]) -> bool:
        self.validate_node_calls.append(node)
        if not self.should_validate:
            raise ValidationError("Mock validation failed")
        return True
    
    def validate_edge(self, edge: Dict[str, Any]) -> bool:
        self.validate_edge_calls.append(edge)
        if not self.should_validate:
            raise ValidationError("Mock validation failed")
        return True
    
    def validate_graph(self, graph: Dict[str, Any]) -> bool:
        return self.should_validate
    
    def normalize_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        result = node.copy()
        result['refs'] = result.get('refs', {
            'ref_img': [],
            'ref_audio': [],
            'ref_video': [],
            'ref_docs': []
        })
        result['created_at'] = result.get('created_at', '2023-01-01T00:00:00')
        result['updated_at'] = '2023-01-01T00:00:00'
        return result
    
    def normalize_edge(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        result = edge.copy()
        result['created_at'] = result.get('created_at', '2023-01-01T00:00:00')
        result['updated_at'] = '2023-01-01T00:00:00'
        result['is_directed'] = result.get('is_directed', False)
        return result


class TestSchemaValidator:
    
    @pytest.fixture
    def temp_schema_file(self):
        schema = {
            "node": {
                "required_fields": {
                    "node_id": {"type": "string", "constraints": {"min_length": 1}},
                    "node_name": {"type": "string", "constraints": {"min_length": 1}},
                    "node_type": {"type": "string", "allowed_values": ["character", "location"]},
                    "is_editable": {"type": "boolean", "default": True},
                    "refs": {"type": "object", "required": False, "default": {}}
                }
            },
            "edge": {
                "required_fields": {
                    "source_node_id": {"type": "string"},
                    "target_node_id": {"type": "string"},
                    "edge_type": {"type": "string", "allowed_values": ["has", "uses"]},
                    "is_editable": {"type": "boolean", "default": True}
                }
            },
            "graph": {
                "required_fields": {
                    "node_dict": {"type": "object"},
                    "edge_dict": {"type": "object"}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = f.name
        
        yield schema_file
        Path(schema_file).unlink()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    def test_init_success(self, temp_schema_file, mock_logger):
        validator = SchemaValidator(temp_schema_file, mock_logger)
        assert validator.schema_file == temp_schema_file
        assert validator.schema is not None
        mock_logger.info.assert_called()
    
    def test_init_schema_not_found(self, mock_logger):
        with pytest.raises(SchemaError, match="Failed to load schema"):
            SchemaValidator("nonexistent.json", mock_logger)
    
    def test_validate_node_success(self, temp_schema_file, mock_logger):
        validator = SchemaValidator(temp_schema_file, mock_logger)
        
        valid_node = {
            "node_id": "test1",
            "node_name": "Test Node",
            "node_type": "character",
            "is_editable": True
        }
        
        assert validator.validate_node(valid_node) is True
    
    def test_validate_node_missing_required_field(self, temp_schema_file, mock_logger):
        validator = SchemaValidator(temp_schema_file, mock_logger)
        
        invalid_node = {
            "node_id": "test1",
            "node_name": "Test Node"
            # Missing node_type
        }
        
        with pytest.raises(ValidationError, match="Missing required field"):
            validator.validate_node(invalid_node)
    
    def test_validate_node_invalid_type(self, temp_schema_file, mock_logger):
        validator = SchemaValidator(temp_schema_file, mock_logger)
        
        invalid_node = {
            "node_id": "test1",
            "node_name": "Test Node",
            "node_type": "invalid_type",  # Not in allowed_values
            "is_editable": True
        }
        
        with pytest.raises(ValidationError):
            validator.validate_node(invalid_node)
    
    def test_normalize_node_applies_defaults(self, temp_schema_file, mock_logger):
        validator = SchemaValidator(temp_schema_file, mock_logger)
        
        node = {
            "node_id": "test1",
            "node_name": "Test Node",
            "node_type": "character"
        }
        
        normalized = validator.normalize_node(node)
        
        assert normalized["is_editable"] is True  # Default applied
        assert "refs" in normalized  # Default refs added
        assert "created_at" in normalized
        assert "updated_at" in normalized


class TestGraphStore:
    
    @pytest.fixture
    def mock_file_store(self):
        return MockFileStore()
    
    @pytest.fixture
    def mock_validator(self):
        return MockValidator()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def graph_store(self, mock_file_store, mock_validator, mock_logger):
        return GraphStore(mock_file_store, mock_validator, "test_graph.json", mock_logger)
    
    @pytest.fixture
    def sample_node(self):
        return {
            "node_id": "node1",
            "node_name": "Test Node",
            "node_type": "character",
            "node_attributes": {"level": 1},
            "is_editable": True,
            "refs": {
                "ref_img": ["image1.jpg"],
                "ref_audio": ["audio1.mp3"],
                "ref_video": [],
                "ref_docs": []
            }
        }
    
    @pytest.fixture
    def sample_edge(self):
        return {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "has",
            "is_editable": True,
            "is_directed": False
        }
    
    def test_init_creates_empty_graph(self, mock_file_store, mock_validator, mock_logger):
        store = GraphStore(mock_file_store, mock_validator, "test_graph.json", mock_logger)
        
        assert "node_dict" in store.graph
        assert "edge_dict" in store.graph
        assert "neighbor_dict" in store.graph
        assert "node_name_to_id" in store.graph
        assert "metadata" in store.graph
    
    def test_load_existing_graph(self, mock_file_store, mock_validator, mock_logger):
        # Setup existing graph data
        existing_graph = {
            "node_dict": {"node1": {"node_id": "node1", "node_name": "Test"}},
            "edge_dict": {},
            "neighbor_dict": {"node1": []},
            "node_name_to_id": {"Test": "node1"},
            "metadata": {"version": "1.0"}
        }
        
        # Mock file store to return existing data
        mock_file_store.files["test_graph.json"] = json.dumps(existing_graph)
        
        store = GraphStore(mock_file_store, mock_validator, "test_graph.json", mock_logger)
        
        assert len(store.graph["node_dict"]) == 1
        assert "node1" in store.graph["node_dict"]
    
    def test_add_node_success(self, graph_store, sample_node, mock_logger):
        graph_store.add_node(sample_node)
        
        assert sample_node["node_id"] in graph_store.graph["node_dict"]
        assert sample_node["node_name"] in graph_store.graph["node_name_to_id"]
        assert sample_node["node_id"] in graph_store.graph["neighbor_dict"]
        # Check that the node was logged (might be overridden by save_graph)
        mock_logger.debug.assert_any_call(f"Added node: {sample_node['node_id']}")
    
    def test_add_node_duplicate_id(self, graph_store, sample_node):
        graph_store.add_node(sample_node)
        
        with pytest.raises(DuplicateError, match="Node with ID .* already exists"):
            graph_store.add_node(sample_node)
    
    def test_add_node_duplicate_name(self, graph_store, sample_node):
        graph_store.add_node(sample_node)
        
        duplicate_name_node = sample_node.copy()
        duplicate_name_node["node_id"] = "node2"
        
        with pytest.raises(DuplicateError, match="Node with name .* already exists"):
            graph_store.add_node(duplicate_name_node)
    
    def test_add_node_invalid_format(self, graph_store):
        invalid_node = {"node_id": "test"}  # Missing required fields
        
        mock_validator = MockValidator(should_validate=False)
        graph_store.validator = mock_validator
        
        with pytest.raises(ValidationError):
            graph_store.add_node(invalid_node)
    
    def test_update_node_success(self, graph_store, sample_node, mock_logger):
        graph_store.add_node(sample_node)
        
        updated_node = sample_node.copy()
        updated_node["node_attributes"] = {"level": 2}
        
        graph_store.update_node(updated_node)
        
        stored_node = graph_store.get_node(sample_node["node_id"])
        assert stored_node["node_attributes"]["level"] == 2
        mock_logger.debug.assert_any_call(f"Updated node: {sample_node['node_id']}")
    
    def test_update_node_not_found(self, graph_store):
        non_existent_node = {
            "node_id": "nonexistent",
            "node_name": "Non-existent",
            "node_type": "character",
            "node_attributes": {},
            "is_editable": True
        }
        
        with pytest.raises(NodeNotFoundError):
            graph_store.update_node(non_existent_node)
    
    def test_update_node_name_change(self, graph_store, sample_node):
        graph_store.add_node(sample_node)
        
        updated_node = sample_node.copy()
        updated_node["node_name"] = "Updated Name"
        
        graph_store.update_node(updated_node)
        
        assert "Updated Name" in graph_store.graph["node_name_to_id"]
        assert sample_node["node_name"] not in graph_store.graph["node_name_to_id"]
    
    def test_delete_node_success(self, graph_store, sample_node, mock_logger):
        graph_store.add_node(sample_node)
        node_id = sample_node["node_id"]
        
        graph_store.delete_node(node_id)
        
        assert node_id not in graph_store.graph["node_dict"]
        assert sample_node["node_name"] not in graph_store.graph["node_name_to_id"]
        assert node_id not in graph_store.graph["neighbor_dict"]
        mock_logger.debug.assert_any_call(f"Deleted node: {node_id}")
    
    def test_delete_node_not_found(self, graph_store):
        with pytest.raises(NodeNotFoundError):
            graph_store.delete_node("nonexistent")
    
    def test_get_node_success(self, graph_store, sample_node):
        graph_store.add_node(sample_node)
        
        retrieved_node = graph_store.get_node(sample_node["node_id"])
        assert retrieved_node is not None
        assert retrieved_node["node_id"] == sample_node["node_id"]
    
    def test_get_node_not_found(self, graph_store):
        retrieved_node = graph_store.get_node("nonexistent")
        assert retrieved_node is None
    
    def test_add_edge_success(self, graph_store, sample_node, sample_edge, mock_logger):
        # Add nodes first
        node2 = sample_node.copy()
        node2["node_id"] = "node2"
        node2["node_name"] = "Test Node 2"
        
        graph_store.add_node(sample_node)
        graph_store.add_node(node2)
        
        # Add edge
        graph_store.add_edge(sample_edge)
        
        edge_key = graph_store.parse_edge_key("node1", "node2", False)
        assert edge_key in graph_store.graph["edge_dict"]
        assert "node2" in graph_store.graph["neighbor_dict"]["node1"]
        assert "node1" in graph_store.graph["neighbor_dict"]["node2"]
    
    def test_add_directed_edge(self, graph_store, sample_node, sample_edge):
        # Add nodes first
        node2 = sample_node.copy()
        node2["node_id"] = "node2"
        node2["node_name"] = "Test Node 2"
        
        graph_store.add_node(sample_node)
        graph_store.add_node(node2)
        
        # Add directed edge
        directed_edge = sample_edge.copy()
        directed_edge["is_directed"] = True
        
        graph_store.add_edge(directed_edge)
        
        # Check neighbor lists for directed edge
        assert "node2" in graph_store.graph["neighbor_dict"]["node1"]
        assert "node1" not in graph_store.graph["neighbor_dict"]["node2"]
    
    def test_add_edge_source_not_found(self, graph_store, sample_edge):
        with pytest.raises(NodeNotFoundError, match="Source node .* not found"):
            graph_store.add_edge(sample_edge)
    
    def test_add_edge_self_loop(self, graph_store, sample_node):
        graph_store.add_node(sample_node)
        
        self_loop_edge = {
            "source_node_id": "node1",
            "target_node_id": "node1",
            "edge_type": "has",
            "is_editable": True
        }
        
        with pytest.raises(ValidationError, match="Self-loops are not allowed"):
            graph_store.add_edge(self_loop_edge)
    
    def test_delete_edge_success(self, graph_store, sample_node, sample_edge, mock_logger):
        # Setup nodes and edge
        node2 = sample_node.copy()
        node2["node_id"] = "node2"
        node2["node_name"] = "Test Node 2"
        
        graph_store.add_node(sample_node)
        graph_store.add_node(node2)
        graph_store.add_edge(sample_edge)
        
        # Delete edge
        graph_store.delete_edge("node1", "node2")
        
        edge_key = graph_store.parse_edge_key("node1", "node2", False)
        assert edge_key not in graph_store.graph["edge_dict"]
        assert "node2" not in graph_store.graph["neighbor_dict"]["node1"]
    
    def test_delete_edge_not_found(self, graph_store):
        with pytest.raises(EdgeNotFoundError):
            graph_store.delete_edge("node1", "node2")
    
    def test_get_neighbor_info_single_hop(self, graph_store, sample_node, sample_edge):
        # Setup nodes and edge
        node2 = sample_node.copy()
        node2["node_id"] = "node2"
        node2["node_name"] = "Test Node 2"
        
        graph_store.add_node(sample_node)
        graph_store.add_node(node2)
        graph_store.add_edge(sample_edge)
        
        # Get neighbor info
        nodes, edges = graph_store.get_neighbor_info("node1", hop=1)
        
        assert len(nodes) == 1
        assert len(edges) == 1
        assert nodes[0]["node_id"] == "node2"
    
    def test_get_neighbor_info_multi_hop(self, graph_store, sample_node):
        # Create a chain of nodes: node1 -> node2 -> node3
        node2 = sample_node.copy()
        node2["node_id"] = "node2"
        node2["node_name"] = "Node 2"
        
        node3 = sample_node.copy()
        node3["node_id"] = "node3"
        node3["node_name"] = "Node 3"
        
        edge1_2 = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "has",
            "is_editable": True
        }
        
        edge2_3 = {
            "source_node_id": "node2",
            "target_node_id": "node3",
            "edge_type": "has",
            "is_editable": True
        }
        
        graph_store.add_node(sample_node)
        graph_store.add_node(node2)
        graph_store.add_node(node3)
        graph_store.add_edge(edge1_2)
        graph_store.add_edge(edge2_3)
        
        # Get 2-hop neighbors
        nodes, edges = graph_store.get_neighbor_info("node1", hop=2)
        
        assert len(nodes) == 2  # node2 and node3
        assert len(edges) == 2  # edge1_2 and edge2_3
    
    def test_get_neighbor_info_node_not_found(self, graph_store):
        with pytest.raises(NodeNotFoundError):
            graph_store.get_neighbor_info("nonexistent")
    
    def test_delete_graph(self, graph_store, sample_node, mock_logger):
        graph_store.add_node(sample_node)
        
        graph_store.delete_graph()
        
        assert len(graph_store.graph["node_dict"]) == 0
        assert len(graph_store.graph["edge_dict"]) == 0
        mock_logger.info.assert_called_with("Graph deleted successfully")
    
    def test_parse_edge_key_undirected(self, graph_store):
        key1 = graph_store.parse_edge_key("node1", "node2", False)
        key2 = graph_store.parse_edge_key("node2", "node1", False)
        
        assert key1 == key2  # Should be the same for undirected edges
        assert "<->" in key1
    
    def test_parse_edge_key_directed(self, graph_store):
        key1 = graph_store.parse_edge_key("node1", "node2", True)
        key2 = graph_store.parse_edge_key("node2", "node1", True)
        
        assert key1 != key2  # Should be different for directed edges
        assert "->" in key1
    
    def test_refs_field_in_normalized_node(self, graph_store, mock_validator):
        node = {
            "node_id": "test1",
            "node_name": "Test Node",
            "node_type": "character",
            "is_editable": True
        }
        
        graph_store.add_node(node)
        stored_node = graph_store.get_node("test1")
        
        assert "refs" in stored_node
        assert "ref_img" in stored_node["refs"]
        assert "ref_audio" in stored_node["refs"]
        assert "ref_video" in stored_node["refs"]
        assert "ref_docs" in stored_node["refs"]


class TestGraphStoreFactory:
    
    @pytest.fixture
    def mock_file_store(self):
        return MockFileStore()
    
    @pytest.fixture
    def temp_schema_file(self):
        schema = {"node": {"required_fields": {}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = f.name
        
        yield schema_file
        Path(schema_file).unlink()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    def test_create_local_store_without_schema(self, mock_file_store, mock_logger):
        store = GraphStoreFactory.create_local_store(mock_file_store, logger=mock_logger)
        
        assert isinstance(store, GraphStore)
        assert store.file_store == mock_file_store
        assert store.validator is None
    
    def test_create_local_store_with_schema(self, mock_file_store, temp_schema_file, mock_logger):
        store = GraphStoreFactory.create_local_store(
            mock_file_store, schema_file=temp_schema_file, logger=mock_logger
        )
        
        assert isinstance(store, GraphStore)
        assert store.validator is not None
        assert isinstance(store.validator, SchemaValidator)
    
    def test_create_cloud_store_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Cloud graph store provider .* not yet implemented"):
            GraphStoreFactory.create_cloud_store("neo4j")


class TestInterfaces:
    
    def test_graph_validator_interface(self):
        """Test that IGraphValidator defines required methods."""
        required_methods = ['validate_node', 'validate_edge', 'validate_graph']
        
        for method_name in required_methods:
            assert hasattr(IGraphValidator, method_name)
            assert callable(getattr(IGraphValidator, method_name))
    
    def test_graph_store_interface(self):
        """Test that IGraphStore defines required methods."""
        required_methods = [
            'load_graph', 'save_graph', 'add_node', 'update_node', 'delete_node',
            'get_node', 'add_edge', 'update_edge', 'delete_edge', 'get_edge',
            'get_neighbor_info', 'delete_graph'
        ]
        
        for method_name in required_methods:
            assert hasattr(IGraphStore, method_name)
            assert callable(getattr(IGraphStore, method_name))


class TestErrorHandling:
    
    def test_custom_exceptions_inheritance(self):
        """Test that custom exceptions inherit from base exception."""
        assert issubclass(GraphStoreError, Exception)
        assert issubclass(ValidationError, GraphStoreError)
        assert issubclass(NodeNotFoundError, GraphStoreError)
        assert issubclass(EdgeNotFoundError, GraphStoreError)
        assert issubclass(DuplicateError, GraphStoreError)
        assert issubclass(SchemaError, GraphStoreError)
    
    def test_validation_error_message(self):
        """Test ValidationError with custom message."""
        error = ValidationError("Test validation error")
        assert str(error) == "Test validation error"
    
    def test_node_not_found_error_message(self):
        """Test NodeNotFoundError with custom message."""
        error = NodeNotFoundError("Node not found")
        assert str(error) == "Node not found"