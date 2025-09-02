import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from memory.memory import (
    ConceptGraphService,
    ConceptGraphFactory,
    ConceptGraphError,
    ConceptNotFoundError,
    RelationNotFoundError,
    IConceptGraph
)
from memory.file_store import IFileStore
from memory.graph_store import IGraphStore
from memory.emb_store import IEmbService


class MockFileStore(IFileStore):
    """Mock file store for testing."""
    
    def __init__(self):
        self.files = {}
        self.prefix_path = Path("/mock/path")
    
    def add_file(self, local_file_path: str, remote_file_name: str) -> None:
        self.files[remote_file_name] = f"content_from_{local_file_path}"
    
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
    
    def delete_prefix(self) -> None:
        self.files.clear()


class MockGraphStore(IGraphStore):
    """Mock graph store for testing."""
    
    def __init__(self):
        self.graph = {
            "node_dict": {},
            "edge_dict": {},
            "neighbor_dict": {},
            "node_name_to_id": {},
            "metadata": {}
        }
    
    def load_graph(self) -> Dict[str, Any]:
        return self.graph
    
    def save_graph(self) -> None:
        pass
    
    def add_node(self, node: Dict[str, Any]) -> None:
        node_id = node["node_id"]
        node_name = node["node_name"]
        
        if node_id in self.graph["node_dict"]:
            raise ValueError(f"Node {node_id} already exists")
        if node_name in self.graph["node_name_to_id"]:
            raise ValueError(f"Node name {node_name} already exists")
        
        self.graph["node_dict"][node_id] = node
        self.graph["neighbor_dict"][node_id] = []
        self.graph["node_name_to_id"][node_name] = node_id
    
    def update_node(self, node: Dict[str, Any]) -> None:
        node_id = node["node_id"]
        if node_id not in self.graph["node_dict"]:
            raise ValueError(f"Node {node_id} not found")
        
        old_name = self.graph["node_dict"][node_id]["node_name"]
        new_name = node["node_name"]
        
        if old_name != new_name:
            del self.graph["node_name_to_id"][old_name]
            self.graph["node_name_to_id"][new_name] = node_id
        
        self.graph["node_dict"][node_id] = node
    
    def delete_node(self, node_id: str) -> None:
        if node_id not in self.graph["node_dict"]:
            raise ValueError(f"Node {node_id} not found")
        
        node_name = self.graph["node_dict"][node_id]["node_name"]
        del self.graph["node_dict"][node_id]
        del self.graph["neighbor_dict"][node_id]
        del self.graph["node_name_to_id"][node_name]
        
        # Remove from neighbor lists
        for neighbor_list in self.graph["neighbor_dict"].values():
            if node_id in neighbor_list:
                neighbor_list.remove(node_id)
        
        # Remove edges
        edges_to_remove = []
        for edge_key in self.graph["edge_dict"].keys():
            if node_id in edge_key:
                edges_to_remove.append(edge_key)
        for edge_key in edges_to_remove:
            del self.graph["edge_dict"][edge_key]
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.graph["node_dict"].get(node_id)
    
    def add_edge(self, edge: Dict[str, Any]) -> None:
        source_id = edge["source_node_id"]
        target_id = edge["target_node_id"]
        edge_key = f"{source_id}<->{target_id}"
        
        if edge_key in self.graph["edge_dict"]:
            raise ValueError(f"Edge {edge_key} already exists")
        
        self.graph["edge_dict"][edge_key] = edge
        self.graph["neighbor_dict"][source_id].append(target_id)
        self.graph["neighbor_dict"][target_id].append(source_id)
    
    def update_edge(self, edge: Dict[str, Any]) -> None:
        source_id = edge["source_node_id"]
        target_id = edge["target_node_id"]
        edge_key = f"{source_id}<->{target_id}"
        
        if edge_key not in self.graph["edge_dict"]:
            raise ValueError(f"Edge {edge_key} not found")
        
        self.graph["edge_dict"][edge_key] = edge
    
    def delete_edge(self, source_node_id: str, target_node_id: str) -> None:
        edge_key = f"{source_node_id}<->{target_node_id}"
        if edge_key not in self.graph["edge_dict"]:
            raise ValueError(f"Edge {edge_key} not found")
        
        del self.graph["edge_dict"][edge_key]
        self.graph["neighbor_dict"][source_node_id].remove(target_node_id)
        self.graph["neighbor_dict"][target_node_id].remove(source_node_id)
    
    def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        edge_key = f"{source_node_id}<->{target_node_id}"
        return self.graph["edge_dict"].get(edge_key)
    
    def get_neighbor_info(self, node_id: str, hop: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if node_id not in self.graph["node_dict"]:
            raise ValueError(f"Node {node_id} not found")
        
        neighbor_ids = self.graph["neighbor_dict"].get(node_id, [])
        nodes = [self.graph["node_dict"][nid] for nid in neighbor_ids if nid in self.graph["node_dict"]]
        
        edges = []
        for neighbor_id in neighbor_ids:
            edge = self.get_edge(node_id, neighbor_id)
            if edge:
                edges.append(edge)
        
        return nodes, edges
    
    def delete_graph(self) -> None:
        self.graph = {
            "node_dict": {},
            "edge_dict": {},
            "neighbor_dict": {},
            "node_name_to_id": {},
            "metadata": {}
        }
    
    def parse_edge_key(self, source_node_id: str, target_node_id: str, is_directed: bool = False) -> str:
        if is_directed:
            return f"{source_node_id}->{target_node_id}"
        else:
            keys = [source_node_id, target_node_id]
            keys.sort()
            return f"{keys[0]}<->{keys[1]}"
    
    def get_node_by_name(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get a node by name."""
        node_id = self.graph["node_name_to_id"].get(node_name)
        if node_id:
            return self.graph["node_dict"].get(node_id)
        return None
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the graph."""
        return list(self.graph["node_dict"].values())


class MockEmbStore(IEmbService):
    """Mock embedding store for testing."""
    
    def __init__(self):
        self.embeddings = {}  # namespace -> {node_id: embedding}
    
    def insert_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        if ns not in self.embeddings:
            self.embeddings[ns] = {}
        # Create a simple embedding based on text hash
        embedding = np.random.random(384).astype(np.float32)
        self.embeddings[ns][node_id] = embedding
    
    def update_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        self.insert_node_emb(node_id, node_text, namespace)
    
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        if ns in self.embeddings and node_id in self.embeddings[ns]:
            del self.embeddings[ns][node_id]
    
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        ns = namespace or "default"
        return self.embeddings.get(ns, {}).get(node_id)
    
    def query_similar_nodes(self, query_text: str, top_k: int = 5, 
                           namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        ns = namespace or "default"
        if ns not in self.embeddings:
            return []
        
        # Return mock similarity results
        results = []
        for node_id in list(self.embeddings[ns].keys())[:top_k]:
            score = np.random.random()  # Mock similarity score
            results.append((node_id, float(score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self.embeddings.clear()
        elif namespace in self.embeddings:
            del self.embeddings[namespace]
    
    def save_index(self) -> None:
        """Mock save_index method - no-op for testing."""
        pass


class TestConceptGraphService:
    
    @pytest.fixture
    def mock_file_store(self):
        return MockFileStore()
    
    @pytest.fixture
    def mock_graph_store(self):
        return MockGraphStore()
    
    @pytest.fixture
    def mock_emb_store(self):
        return MockEmbStore()
    
    @pytest.fixture
    def mock_file_store_with_spy(self):
        mock = MockFileStore()
        mock.add_file = Mock(side_effect=mock.add_file)
        return mock
    
    @pytest.fixture
    def mock_graph_store_with_spy(self):
        mock = MockGraphStore()
        mock.save_graph = Mock(side_effect=mock.save_graph)
        mock.delete_graph = Mock(side_effect=mock.delete_graph)
        return mock
    
    @pytest.fixture
    def mock_emb_store_with_spy(self):
        mock = MockEmbStore()
        mock.query_similar_nodes = Mock(side_effect=mock.query_similar_nodes)
        mock.delete_index = Mock(side_effect=mock.delete_index)
        mock.save_index = Mock(side_effect=mock.save_index)
        return mock
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def concept_graph(self, mock_file_store, mock_graph_store, mock_emb_store, mock_logger):
        return ConceptGraphService(mock_file_store, mock_graph_store, mock_emb_store, mock_logger)
    
    @pytest.fixture
    def sample_concept_data(self):
        return {
            "concept_name": "Hero",
            "concept_type": "character",
            "concept_attributes": {"level": 1, "health": 100},
            "refs": {
                "ref_img": ["hero.jpg"],
                "ref_audio": ["hero_voice.mp3"],
                "ref_video": [],
                "ref_docs": ["hero_backstory.txt"]
            }
        }
    
    def test_init(self, mock_file_store, mock_graph_store, mock_emb_store, mock_logger):
        service = ConceptGraphService(mock_file_store, mock_graph_store, mock_emb_store, mock_logger)
        
        assert service.file_store == mock_file_store
        assert service.graph_store == mock_graph_store
        assert service.emb_store == mock_emb_store
        assert service.logger == mock_logger
    
    def test_add_concept_success(self, concept_graph, sample_concept_data, mock_logger):
        concept_id = concept_graph.add_concept(**sample_concept_data)
        
        assert isinstance(concept_id, str)
        assert len(concept_id) == 32  # UUID hex length
        
        # Verify concept was added to graph
        stored_concept = concept_graph.get_concept(concept_id)
        assert stored_concept is not None
        assert stored_concept["node_name"] == "Hero"
        assert stored_concept["node_type"] == "character"
        assert stored_concept["refs"] == sample_concept_data["refs"]
        
        mock_logger.debug.assert_called()
    
    def test_add_concept_with_default_refs(self, concept_graph):
        concept_id = concept_graph.add_concept(
            "Villain", "character", {"level": 5}
        )
        
        stored_concept = concept_graph.get_concept(concept_id)
        assert stored_concept["refs"] == {
            "ref_img": [],
            "ref_audio": [],
            "ref_video": [],
            "ref_docs": []
        }
    
    def test_add_concept_with_image(self, mock_file_store_with_spy, mock_graph_store, mock_emb_store, mock_logger):
        concept_graph = ConceptGraphService(mock_file_store_with_spy, mock_graph_store, mock_emb_store, mock_logger)
        
        concept_id = concept_graph.add_concept(
            "Castle", "location", {"rooms": 10}, image_path="/path/to/castle.jpg"
        )
        
        stored_concept = concept_graph.get_concept(concept_id)
        assert stored_concept["image_path"] == f"imgs/{concept_id}.jpg"
        
        # Verify file was stored
        mock_file_store_with_spy.add_file.assert_called_with("/path/to/castle.jpg", f"imgs/{concept_id}.jpg")
    
    def test_add_concept_invalid_attributes(self, concept_graph):
        with pytest.raises(ValueError, match="Attributes must be a dictionary"):
            concept_graph.add_concept("Test", "character", "invalid_attributes")
    
    def test_update_concept_success(self, concept_graph, sample_concept_data, mock_logger):
        concept_id = concept_graph.add_concept(**sample_concept_data)
        
        new_refs = {
            "ref_img": ["new_hero.jpg"],
            "ref_audio": [],
            "ref_video": ["hero_video.mp4"],
            "ref_docs": []
        }
        
        concept_graph.update_concept(
            concept_id,
            concept_name="Updated Hero",
            concept_attributes={"level": 2},
            refs=new_refs
        )
        
        stored_concept = concept_graph.get_concept(concept_id)
        assert stored_concept["node_name"] == "Updated Hero"
        assert stored_concept["node_attributes"]["level"] == 2
        assert stored_concept["refs"] == new_refs
        
        mock_logger.debug.assert_called()
    
    def test_update_concept_not_found(self, concept_graph):
        with pytest.raises(ConceptNotFoundError):
            concept_graph.update_concept("nonexistent", concept_name="Test")
    
    def test_delete_concept_success(self, concept_graph, sample_concept_data, mock_logger):
        concept_id = concept_graph.add_concept(**sample_concept_data)
        
        concept_graph.delete_concept(concept_id)
        
        assert concept_graph.get_concept(concept_id) is None
        mock_logger.debug.assert_called()
    
    def test_delete_concept_not_found(self, concept_graph):
        with pytest.raises(ConceptNotFoundError):
            concept_graph.delete_concept("nonexistent")
    
    def test_get_concept_success(self, concept_graph, sample_concept_data):
        concept_id = concept_graph.add_concept(**sample_concept_data)
        
        retrieved_concept = concept_graph.get_concept(concept_id)
        
        assert retrieved_concept is not None
        assert retrieved_concept["node_id"] == concept_id
        assert retrieved_concept["node_name"] == "Hero"
    
    def test_get_concept_not_found(self, concept_graph):
        result = concept_graph.get_concept("nonexistent")
        assert result is None
    
    def test_add_relation_success(self, concept_graph, mock_logger):
        concept1_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        concept2_id = concept_graph.add_concept("Sword", "item", {"damage": 10})
        
        concept_graph.add_relation(concept1_id, concept2_id, "has")
        
        relation = concept_graph.get_relation(concept1_id, concept2_id)
        assert relation is not None
        assert relation["edge_type"] == "has"
        
        mock_logger.debug.assert_called()
    
    def test_add_relation_source_not_found(self, concept_graph):
        concept_id = concept_graph.add_concept("Sword", "item", {"damage": 10})
        
        with pytest.raises(ConceptNotFoundError):
            concept_graph.add_relation("nonexistent", concept_id, "has")
    
    def test_delete_relation_success(self, concept_graph, mock_logger):
        concept1_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        concept2_id = concept_graph.add_concept("Sword", "item", {"damage": 10})
        
        concept_graph.add_relation(concept1_id, concept2_id, "has")
        concept_graph.delete_relation(concept1_id, concept2_id)
        
        relation = concept_graph.get_relation(concept1_id, concept2_id)
        assert relation is None
        
        mock_logger.debug.assert_called()
    
    def test_query_similar_concepts(self, concept_graph):
        concept1_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        concept2_id = concept_graph.add_concept("Villain", "character", {"level": 5})
        
        results = concept_graph.query_similar_concepts("character", top_k=2)
        
        # Verify we get results (exact ordering may vary due to mock randomness)
        assert len(results) <= 2
        for concept, score in results:
            assert concept["node_id"] in [concept1_id, concept2_id]
            assert isinstance(score, float)
            assert 0 <= score <= 1
    
    def test_get_related_concepts_success(self, concept_graph):
        hero_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        sword_id = concept_graph.add_concept("Sword", "item", {"damage": 10})
        
        concept_graph.add_relation(hero_id, sword_id, "has")
        
        related_concepts, related_relations = concept_graph.get_related_concepts(hero_id)
        
        assert len(related_concepts) == 1
        assert len(related_relations) == 1
        assert related_concepts[0]["node_name"] == "Sword"
    
    def test_get_related_concepts_not_found(self, concept_graph):
        with pytest.raises(ConceptNotFoundError):
            concept_graph.get_related_concepts("nonexistent")
    
    def test_save_graph(self, mock_file_store, mock_graph_store_with_spy, mock_emb_store_with_spy, mock_logger):
        concept_graph = ConceptGraphService(mock_file_store, mock_graph_store_with_spy, mock_emb_store_with_spy, mock_logger)
        concept_graph.save_graph()
        
        mock_graph_store_with_spy.save_graph.assert_called_once()
        mock_emb_store_with_spy.save_index.assert_called_once()
        mock_logger.debug.assert_called_with("Graph and embeddings saved")
    
    def test_empty_graph(self, mock_file_store, mock_graph_store_with_spy, mock_emb_store_with_spy, mock_logger):
        concept_graph = ConceptGraphService(mock_file_store, mock_graph_store_with_spy, mock_emb_store_with_spy, mock_logger)
        concept_graph.empty_graph()
        
        mock_graph_store_with_spy.delete_graph.assert_called_once()
        mock_emb_store_with_spy.delete_index.assert_called_once()
        mock_logger.info.assert_called_with("Graph emptied successfully (in-memory only)")
    
    def test_is_concept(self, concept_graph):
        concept_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        
        assert concept_graph.is_concept(concept_id) is True
        assert concept_graph.is_concept("nonexistent") is False
    
    def test_is_relation(self, concept_graph):
        concept1_id = concept_graph.add_concept("Hero", "character", {"level": 1})
        concept2_id = concept_graph.add_concept("Sword", "item", {"damage": 10})
        
        assert concept_graph.is_relation(concept1_id, concept2_id) is False
        
        concept_graph.add_relation(concept1_id, concept2_id, "has")
        assert concept_graph.is_relation(concept1_id, concept2_id) is True
    
    def test_node_to_text_excludes_refs(self, concept_graph):
        """Test that _node_to_text excludes refs from embedding text."""
        node = {
            "node_name": "Hero",
            "node_type": "character",
            "node_attributes": {"level": 1},
            "refs": {
                "ref_img": ["hero.jpg"],
                "ref_audio": ["hero_voice.mp3"],
                "ref_video": [],
                "ref_docs": []
            }
        }
        
        text = concept_graph._node_to_text(node)
        
        # Verify refs are not included in the text
        assert "refs" not in text
        assert "hero.jpg" not in text
        assert "hero_voice.mp3" not in text
        
        # Verify other fields are included
        assert "Hero" in text
        assert "character" in text
        assert "level" in text


class TestConceptGraphFactory:
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def sample_concept_config(self):
        return {
            "provider": "local",
            "embedding_api_key": "test_key",
            "emb_model": "text-embedding-3-small",
            "emb_dim": 384
        }
    
    @pytest.fixture
    def sample_file_config(self):
        return {
            "provider": "local",
            "save_path": "/tmp/test"
        }
    
    @patch('memory.memory.FileStoreFactory')
    @patch('memory.memory.GraphStoreFactory')
    @patch('memory.memory.EmbServiceFactory')
    def test_create_from_memory_core_local(self, mock_emb_factory, mock_graph_factory, 
                                          mock_file_factory, sample_concept_config, 
                                          sample_file_config, mock_logger):
        mock_file_store = Mock()
        mock_graph_store = Mock()
        mock_emb_store = Mock()
        
        mock_file_factory.create_local_store.return_value = mock_file_store
        mock_graph_factory.create_local_store.return_value = mock_graph_store
        mock_emb_factory.create_local_store.return_value = mock_emb_store
        
        # Create memory core config structure
        memory_core_config = {
            "embedding": {
                "provider": sample_concept_config["provider"],
                "api_key": sample_concept_config.get("embedding_api_key", ""),
                "model": sample_concept_config.get("emb_model", "text-embedding-3-small"),
                "dim": sample_concept_config.get("emb_dim", 1536)
            },
            "files": {
                "graph_file": "graph.json",
                "index_file": "emb_index.json"
            }
        }
        
        result = ConceptGraphFactory.create_from_memory_core(
            memory_core_path="/test/memory_core",
            memory_core_config=memory_core_config,
            logger=mock_logger
        )
        
        assert isinstance(result, ConceptGraphService)
        assert result.file_store == mock_file_store
        assert result.graph_store == mock_graph_store
        assert result.emb_store == mock_emb_store
    
    def test_create_custom(self, mock_logger):
        mock_file_store = Mock(spec=IFileStore)
        mock_graph_store = Mock(spec=IGraphStore)
        mock_emb_store = Mock(spec=IEmbService)
        
        result = ConceptGraphFactory.create_custom(
            mock_file_store, mock_graph_store, mock_emb_store, mock_logger
        )
        
        assert isinstance(result, ConceptGraphService)
        assert result.file_store == mock_file_store
        assert result.graph_store == mock_graph_store
        assert result.emb_store == mock_emb_store




class TestInterfaces:
    
    def test_concept_graph_interface(self):
        """Test that IConceptGraph defines required methods."""
        required_methods = [
            'add_concept', 'update_concept', 'delete_concept', 'get_concept',
            'add_relation', 'query_similar_concepts', 'get_related_concepts'
        ]
        
        for method_name in required_methods:
            assert hasattr(IConceptGraph, method_name)
            assert callable(getattr(IConceptGraph, method_name))


class TestErrorHandling:
    
    def test_custom_exceptions_inheritance(self):
        """Test that custom exceptions inherit from base exception."""
        assert issubclass(ConceptGraphError, Exception)
        assert issubclass(ConceptNotFoundError, ConceptGraphError)
        assert issubclass(RelationNotFoundError, ConceptGraphError)
    
    def test_concept_not_found_error_message(self):
        """Test ConceptNotFoundError with custom message."""
        error = ConceptNotFoundError("Concept not found")
        assert str(error) == "Concept not found"