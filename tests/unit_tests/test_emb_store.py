import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import List, Tuple, Optional

from concept_graph.emb_store import (
    EmbStoreService,
    EmbServiceFactory,
    OpenAIEmbeddingProvider,
    FAISSVectorStore,
    PineconeVectorStore,
    EmbStoreError,
    EmbeddingError,
    IndexError,
    NamespaceError,
    IEmbeddingProvider,
    IVectorStore,
    IEmbService
)


class MockEmbeddingProvider(IEmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def generate_embedding(self, text: str) -> np.ndarray:
        # Generate deterministic embedding based on text hash
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        return np.random.random(self.dimension).astype(np.float32)


class MockVectorStore(IVectorStore):
    """Mock vector store for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.data = {}  # namespace -> {node_id: embedding}
    
    def insert_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        if ns not in self.data:
            self.data[ns] = {}
        self.data[ns][node_id] = embedding.copy()
    
    def update_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        self.insert_node_emb(node_id, embedding, namespace)
    
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        ns = namespace or "default"
        if ns in self.data and node_id in self.data[ns]:
            del self.data[ns][node_id]
    
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        ns = namespace or "default"
        return self.data.get(ns, {}).get(node_id)
    
    def query_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        ns = namespace or "default"
        if ns not in self.data:
            return []
        
        similarities = []
        for node_id, embedding in self.data[ns].items():
            score = float(np.dot(query_embedding, embedding))
            similarities.append((node_id, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self.data.clear()
        elif namespace in self.data:
            del self.data[namespace]
    
    def save_index(self) -> None:
        """Mock save index method."""
        pass


class TestEmbStoreService:
    
    @pytest.fixture
    def mock_embedding_provider(self):
        return MockEmbeddingProvider()
    
    @pytest.fixture
    def mock_vector_store(self):
        return MockVectorStore()
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def emb_store(self, mock_embedding_provider, mock_vector_store, mock_logger):
        return EmbStoreService(mock_embedding_provider, mock_vector_store, mock_logger)
    
    def test_init(self, mock_embedding_provider, mock_vector_store, mock_logger):
        store = EmbStoreService(mock_embedding_provider, mock_vector_store, mock_logger)
        assert store.embedding_provider == mock_embedding_provider
        assert store.vector_store == mock_vector_store
        assert store.logger == mock_logger
    
    def test_insert_node_emb(self, emb_store, mock_logger):
        node_id = "test_node"
        node_text = "test text"
        
        emb_store.insert_node_emb(node_id, node_text)
        
        # Verify embedding was generated and stored
        stored_emb = emb_store.get_node_emb(node_id)
        assert stored_emb is not None
        assert isinstance(stored_emb, np.ndarray)
        mock_logger.debug.assert_called_with(f"Inserted node: {node_id}")
    
    def test_insert_node_emb_with_namespace(self, emb_store):
        node_id = "test_node"
        node_text = "test text"
        namespace = "test_namespace"
        
        emb_store.insert_node_emb(node_id, node_text, namespace)
        
        # Verify embedding was stored in correct namespace
        stored_emb = emb_store.get_node_emb(node_id, namespace)
        assert stored_emb is not None
        
        # Verify it's not in default namespace
        default_emb = emb_store.get_node_emb(node_id)
        assert default_emb is None
    
    def test_update_node_emb(self, emb_store, mock_logger):
        node_id = "test_node"
        original_text = "original text"
        updated_text = "updated text"
        
        # Insert original
        emb_store.insert_node_emb(node_id, original_text)
        original_emb = emb_store.get_node_emb(node_id)
        
        # Update
        emb_store.update_node_emb(node_id, updated_text)
        updated_emb = emb_store.get_node_emb(node_id)
        
        # Verify embedding changed
        assert not np.array_equal(original_emb, updated_emb)
        mock_logger.debug.assert_called_with(f"Updated node: {node_id}")
    
    def test_delete_node_emb(self, emb_store, mock_logger):
        node_id = "test_node"
        node_text = "test text"
        
        # Insert then delete
        emb_store.insert_node_emb(node_id, node_text)
        assert emb_store.get_node_emb(node_id) is not None
        
        emb_store.delete_node_emb(node_id)
        assert emb_store.get_node_emb(node_id) is None
        mock_logger.debug.assert_called_with(f"Deleted node: {node_id}")
    
    def test_query_similar_nodes(self, emb_store):
        # Insert test nodes
        nodes = [
            ("node1", "apple fruit red"),
            ("node2", "orange fruit orange"),
            ("node3", "car vehicle transport")
        ]
        
        for node_id, text in nodes:
            emb_store.insert_node_emb(node_id, text)
        
        # Query for fruit-related content
        results = emb_store.query_similar_nodes("fruit", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result[0], str) and isinstance(result[1], float) for result in results)
    
    def test_delete_index(self, emb_store):
        # Insert test data
        emb_store.insert_node_emb("node1", "text1")
        emb_store.insert_node_emb("node2", "text2", "namespace1")
        
        # Delete all
        emb_store.delete_index()
        
        assert emb_store.get_node_emb("node1") is None
        assert emb_store.get_node_emb("node2", "namespace1") is None


class TestOpenAIEmbeddingProvider:
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @patch('concept_graph.emb_store.OPENAI_AVAILABLE', True)
    @patch('concept_graph.emb_store.OpenAI')
    def test_init_success(self, mock_openai_class, mock_logger):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider("test_key", "test_model", mock_logger)
        
        assert provider.client == mock_client
        assert provider.model == "test_model"
        assert provider.logger == mock_logger
        mock_openai_class.assert_called_once_with(api_key="test_key")
    
    @patch('concept_graph.emb_store.OPENAI_AVAILABLE', False)
    def test_init_openai_not_available(self, mock_logger):
        with pytest.raises(ImportError, match="OpenAI package not available"):
            OpenAIEmbeddingProvider("test_key", "test_model", mock_logger)
    
    @patch('concept_graph.emb_store.OPENAI_AVAILABLE', True)
    @patch('concept_graph.emb_store.OpenAI')
    def test_generate_embedding_success(self, mock_openai_class, mock_logger):
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider("test_key", "test_model", mock_logger)
        result = provider.generate_embedding("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])
        mock_client.embeddings.create.assert_called_once_with(
            input="test text",
            model="test_model"
        )
    
    @patch('concept_graph.emb_store.OPENAI_AVAILABLE', True)
    @patch('concept_graph.emb_store.OpenAI')
    def test_generate_embedding_api_error(self, mock_openai_class, mock_logger):
        # Setup mock client to raise exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIEmbeddingProvider("test_key", "test_model", mock_logger)
        
        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            provider.generate_embedding("test text")


class TestFAISSVectorStore:
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_file_store(self, temp_dir):
        mock_store = Mock()
        mock_store.prefix_path = Path(temp_dir)
        mock_store.add_file = Mock()
        return mock_store
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @patch('concept_graph.emb_store.FAISS_AVAILABLE', True)
    @patch('concept_graph.emb_store.faiss')
    def test_init_success(self, mock_faiss, mock_file_store, mock_logger):
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        store = FAISSVectorStore(384, mock_file_store, "test_index.json", mock_logger)
        
        assert store.dimension == 384
        assert store.file_store == mock_file_store
        assert store.index_file_name == "test_index.json"
        assert "default" in store.namespaces
    
    @patch('concept_graph.emb_store.FAISS_AVAILABLE', False)
    def test_init_faiss_not_available(self, mock_file_store, mock_logger):
        with pytest.raises(ImportError, match="FAISS package not available"):
            FAISSVectorStore(384, mock_file_store, "test_index.json", mock_logger)
    
    @patch('concept_graph.emb_store.FAISS_AVAILABLE', True)
    @patch('concept_graph.emb_store.faiss')
    def test_insert_and_get_node_emb(self, mock_faiss, mock_file_store, mock_logger):
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.reconstruct_n.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Mock file store to avoid JSON serialization issues
        mock_file_store.add_file = Mock()
        
        store = FAISSVectorStore(384, mock_file_store, "test_index.json", mock_logger)
        
        # Mock the add operation
        def mock_add(vectors):
            mock_index.ntotal += vectors.shape[0]
        mock_index.add.side_effect = mock_add
        
        # Mock the reconstruct operation
        test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_index.reconstruct.return_value = test_embedding
        
        # Insert embedding
        store.insert_node_emb("test_node", test_embedding)
        
        # Verify insert was called (FAISS only saves to file when save_index is called)
        mock_index.add.assert_called()
        mock_file_store.add_file.assert_not_called()  # Should not be called on insert
        
        # Get embedding
        result = store.get_node_emb("test_node")
        assert result is not None
        
        # Test saving index
        store.save_index()
        mock_file_store.add_file.assert_called_once()  # Should be called when saving


class TestPineconeVectorStore:
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @patch('concept_graph.emb_store.PINECONE_AVAILABLE', True)
    @patch('concept_graph.emb_store.Pinecone')
    def test_init_success(self, mock_pinecone_class, mock_logger):
        # Setup mocks
        mock_pc = Mock()
        mock_pc.list_indexes.return_value.names.return_value = ["existing_index"]
        
        # Mock describe_index to return an object with status dict
        mock_describe_response = Mock()
        mock_describe_response.status = {'ready': True}
        mock_pc.describe_index.return_value = mock_describe_response
        
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc
        
        store = PineconeVectorStore(
            "test_key", "test-index", 384, logger=mock_logger
        )
        
        assert store.index_name == "test-index"
        assert store.dimension == 384
        mock_pc.create_index.assert_called_once()
    
    @patch('concept_graph.emb_store.PINECONE_AVAILABLE', False)
    def test_init_pinecone_not_available(self, mock_logger):
        with pytest.raises(ImportError, match="Pinecone package not available"):
            PineconeVectorStore("test_key", "test_index", 384, logger=mock_logger)
    
    def test_validate_index_name_empty(self):
        with pytest.raises(ValueError, match="Pinecone index name is required"):
            PineconeVectorStore._validate_index_name(None, "")
    
    def test_validate_index_name_invalid_format(self):
        with pytest.raises(ValueError, match="Index name must consist of lowercase"):
            PineconeVectorStore._validate_index_name(None, "Invalid_Name")


class TestEmbServiceFactory:
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @patch('concept_graph.emb_store.OpenAIEmbeddingProvider')
    @patch('concept_graph.emb_store.PineconeVectorStore')
    def test_create_pinecone_store(self, mock_pinecone_store, mock_openai_provider, mock_logger):
        mock_provider_instance = Mock()
        mock_store_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance
        mock_pinecone_store.return_value = mock_store_instance
        
        result = EmbServiceFactory.create_pinecone_store(
            "emb_key", "pc_key", "index", "model", 384, logger=mock_logger
        )
        
        assert isinstance(result, EmbStoreService)
        assert result.embedding_provider == mock_provider_instance
        assert result.vector_store == mock_store_instance
    
    @patch('concept_graph.emb_store.OpenAIEmbeddingProvider')
    @patch('concept_graph.emb_store.FAISSVectorStore')
    def test_create_local_store(self, mock_faiss_store, mock_openai_provider, mock_logger):
        mock_provider_instance = Mock()
        mock_store_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance
        mock_faiss_store.return_value = mock_store_instance
        mock_file_store = Mock()
        
        result = EmbServiceFactory.create_local_store(
            "emb_key", mock_file_store, "model", 384, logger=mock_logger
        )
        
        assert isinstance(result, EmbStoreService)
        assert result.embedding_provider == mock_provider_instance
        assert result.vector_store == mock_store_instance
    
    def test_create_custom_store(self, mock_logger):
        mock_provider = Mock(spec=IEmbeddingProvider)
        mock_store = Mock(spec=IVectorStore)
        
        result = EmbServiceFactory.create_custom_store(mock_provider, mock_store, mock_logger)
        
        assert isinstance(result, EmbStoreService)
        assert result.embedding_provider == mock_provider
        assert result.vector_store == mock_store


class TestInterfaces:
    
    def test_embedding_provider_interface(self):
        """Test that IEmbeddingProvider defines required methods."""
        required_methods = ['generate_embedding']
        
        for method_name in required_methods:
            assert hasattr(IEmbeddingProvider, method_name)
            assert callable(getattr(IEmbeddingProvider, method_name))
    
    def test_vector_store_interface(self):
        """Test that IVectorStore defines required methods."""
        required_methods = [
            'insert_node_emb', 'update_node_emb', 'delete_node_emb',
            'get_node_emb', 'query_similar_nodes', 'delete_index'
        ]
        
        for method_name in required_methods:
            assert hasattr(IVectorStore, method_name)
            assert callable(getattr(IVectorStore, method_name))
    
    def test_emb_store_interface(self):
        """Test that IEmbService defines required methods."""
        required_methods = [
            'insert_node_emb', 'update_node_emb', 'delete_node_emb',
            'get_node_emb', 'query_similar_nodes', 'delete_index'
        ]
        
        for method_name in required_methods:
            assert hasattr(IEmbService, method_name)
            assert callable(getattr(IEmbService, method_name))


class TestErrorHandling:
    
    def test_custom_exceptions_inheritance(self):
        """Test that custom exceptions inherit from base exception."""
        assert issubclass(EmbStoreError, Exception)
        assert issubclass(EmbeddingError, EmbStoreError)
        assert issubclass(IndexError, EmbStoreError)
        assert issubclass(NamespaceError, EmbStoreError)
    
    def test_embedding_error_message(self):
        """Test EmbeddingError with custom message."""
        error = EmbeddingError("Test embedding error")
        assert str(error) == "Test embedding error"
    
    def test_index_error_message(self):
        """Test IndexError with custom message."""
        error = IndexError("Test index error")
        assert str(error) == "Test index error"
    
    def test_namespace_error_message(self):
        """Test NamespaceError with custom message."""
        error = NamespaceError("Test namespace error")
        assert str(error) == "Test namespace error"