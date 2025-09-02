from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import re
import json
import numpy as np
import logging
from pathlib import Path
import tempfile
import os

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

if TYPE_CHECKING:
    from .file_store import IFileStore

class EmbStoreError(Exception):
    """Base exception for embedding store operations."""
    pass


class EmbeddingError(EmbStoreError):
    """Raised when embedding generation fails."""
    pass


class IndexError(EmbStoreError):
    """Raised when index operations fail."""
    pass


class NamespaceError(EmbStoreError):
    """Raised when namespace operations fail."""
    pass


class IEmbeddingProvider(ABC):
    """Interface for embedding generation providers."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage operations."""
    
    @abstractmethod
    def insert_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Insert node embedding."""
        pass
    
    @abstractmethod
    def update_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Update node embedding."""
        pass
    
    @abstractmethod
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding."""
        pass
    
    @abstractmethod
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding."""
        pass
    
    @abstractmethod
    def query_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes."""
        pass
    
    @abstractmethod
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete index or namespace."""
        pass
    
    @abstractmethod
    def save_index(self) -> None:
        """Save index to persistent storage."""
        pass


class IEmbService(ABC):
    """Interface for embedding service operations."""
    
    @abstractmethod
    def insert_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Insert node embedding from text."""
        pass
    
    @abstractmethod
    def update_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Update node embedding from text."""
        pass
    
    @abstractmethod
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding."""
        pass
    
    @abstractmethod
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding."""
        pass
    
    @abstractmethod
    def query_similar_nodes(self, query_text: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes by text."""
        pass
    
    @abstractmethod
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete index or namespace."""
        pass
    
    @abstractmethod
    def save_index(self) -> None:
        """Save index to persistent storage."""
        pass


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", logger: Optional[logging.Logger] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            self.logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")


class PineconeVectorStore(IVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: str, index_name: str, dimension: int, 
                 metric: str = 'cosine', cloud: str = 'aws', region: str = 'us-west-2',
                 logger: Optional[logging.Logger] = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone package not available. Install with: pip install pinecone-client")
        
        self._validate_index_name(index_name)
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.logger = logger or logging.getLogger(__name__)
        
        self._ensure_index_exists(metric, cloud, region)
        self.index = self.pc.Index(index_name)
    
    def _validate_index_name(self, index_name: str) -> None:
        """Validate Pinecone index name format."""
        if not index_name:
            raise ValueError("Pinecone index name is required")
        if not re.match(r'^[a-z0-9-]+$', index_name):
            raise ValueError("Index name must consist of lowercase alphanumeric characters or '-'")
    
    def _ensure_index_exists(self, metric: str, cloud: str, region: str) -> None:
        """Create index if it doesn't exist."""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                self.logger.info(f"Created Pinecone index: {self.index_name}")
                
                # Wait for index to be ready
                import time
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                    self.logger.debug(f"Waiting for index {self.index_name} to be ready...")
                self.logger.info(f"Index {self.index_name} is ready")
        except Exception as e:
            raise IndexError(f"Failed to create/access index {self.index_name}: {e}")
    
    def insert_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Insert node embedding into Pinecone."""
        try:
            vector_data = {"id": node_id, "values": embedding.tolist()}
            if namespace:
                self.index.upsert(vectors=[vector_data], namespace=namespace)
            else:
                self.index.upsert(vectors=[vector_data])
            self.logger.debug(f"Inserted embedding for node: {node_id}")
        except Exception as e:
            raise IndexError(f"Failed to insert node {node_id}: {e}")
    
    def update_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Update node embedding (same as insert for Pinecone)."""
        self.insert_node_emb(node_id, embedding, namespace)
    
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding from Pinecone."""
        try:
            if namespace:
                self.index.delete(ids=[node_id], namespace=namespace)
            else:
                self.index.delete(ids=[node_id])
            self.logger.debug(f"Deleted embedding for node: {node_id}")
        except Exception as e:
            raise IndexError(f"Failed to delete node {node_id}: {e}")
    
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding from Pinecone."""
        try:
            if namespace:
                response = self.index.fetch(ids=[node_id], namespace=namespace)
            else:
                response = self.index.fetch(ids=[node_id])
            
            if response.vectors.get(node_id):
                return np.array(response.vectors[node_id].values, dtype=np.float32)
            return None
        except Exception as e:
            raise IndexError(f"Failed to get node {node_id}: {e}")
    
    def query_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes from Pinecone."""
        try:
            if namespace:
                response = self.index.query(vector=query_embedding.tolist(), top_k=top_k, 
                                          include_values=True, namespace=namespace)
            else:
                response = self.index.query(vector=query_embedding.tolist(), top_k=top_k, include_values=True)
            
            return [(match.id, match.score) for match in response.matches]
        except Exception as e:
            raise IndexError(f"Failed to query similar nodes: {e}")
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete entire Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            self.logger.info(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            raise IndexError(f"Failed to delete index {self.index_name}: {e}")
    
    def save_index(self) -> None:
        """Pinecone automatically persists changes, no action needed."""
        pass


class FAISSVectorStore(IVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, dimension: int, file_store: 'IFileStore', index_file_name: str = "emb_index.json",
                 logger: Optional[logging.Logger] = None):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS package not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.file_store = file_store
        self.index_file_name = index_file_name
        self.logger = logger or logging.getLogger(__name__)
        self.namespaces: Dict[str, Dict[str, Any]] = {}
        self.load_index()
    
    def load_index(self) -> None:
        """Load FAISS index from file."""
        try:
            index_path = Path(self.file_store.prefix_path) / self.index_file_name
            if index_path.exists():
                with open(index_path, "r") as f:
                    data = json.load(f)
                
                for namespace, namespace_data in data.items():
                    self.namespaces[namespace] = {
                        "index": faiss.IndexFlatIP(self.dimension),
                        "id_to_index": namespace_data["id_to_index"]
                    }
                    vectors = np.array(namespace_data["vectors"], dtype=np.float32)
                    if vectors.shape[1] != self.dimension:
                        raise ValueError(f"Dimension mismatch in namespace {namespace}")
                    self.namespaces[namespace]["index"].add(vectors)
                
                self.logger.info(f"Loaded index with {len(self.namespaces)} namespaces")
            else:
                self._create_default_namespace()
        except Exception as e:
            self.logger.warning(f"Failed to load index: {e}, creating new index")
            self._create_default_namespace()
    
    def _create_default_namespace(self) -> None:
        """Create default namespace."""
        self.namespaces["default"] = {
            "index": faiss.IndexFlatIP(self.dimension),
            "id_to_index": {}
        }
    
    def save_index(self) -> None:
        """Save FAISS index to file."""
        try:
            data = {}
            for namespace, namespace_data in self.namespaces.items():
                vectors = namespace_data["index"].reconstruct_n(0, namespace_data["index"].ntotal)
                data[namespace] = {
                    "id_to_index": namespace_data["id_to_index"],
                    "vectors": vectors.tolist()
                }
            
            # Use temporary file for atomic write
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                json.dump(data, temp_file)
                temp_path = temp_file.name
            
            self.file_store.add_file(temp_path, self.index_file_name)
            os.unlink(temp_path)  # Clean up temp file
            
            self.logger.debug("Saved FAISS index")
        except Exception as e:
            raise IndexError(f"Failed to save index: {e}")
    
    def get_namespace(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get or create namespace."""
        if namespace is None:
            namespace = "default"
        
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {
                "index": faiss.IndexFlatIP(self.dimension),
                "id_to_index": {}
            }
            self.logger.debug(f"Created namespace: {namespace}")
        
        return self.namespaces[namespace]
    
    def insert_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Insert node embedding into FAISS (in-memory only)."""
        try:
            namespace_data = self.get_namespace(namespace)
            namespace_data["index"].add(np.array([embedding], dtype=np.float32))
            namespace_data["id_to_index"][node_id] = namespace_data["index"].ntotal - 1
            self.logger.debug(f"Inserted embedding for node: {node_id}")
        except Exception as e:
            raise IndexError(f"Failed to insert node {node_id}: {e}")
    
    def update_node_emb(self, node_id: str, embedding: np.ndarray, namespace: Optional[str] = None) -> None:
        """Update node embedding in FAISS."""
        namespace_data = self.get_namespace(namespace)
        if node_id in namespace_data["id_to_index"]:
            self.delete_node_emb(node_id, namespace)
        self.insert_node_emb(node_id, embedding, namespace)
    
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding from FAISS (in-memory only)."""
        try:
            namespace_data = self.get_namespace(namespace)
            if node_id not in namespace_data["id_to_index"]:
                raise NamespaceError(f"Node {node_id} not found in namespace {namespace}")
            
            index_to_remove = namespace_data["id_to_index"][node_id]
            vectors = namespace_data["index"].reconstruct_n(0, namespace_data["index"].ntotal)
            vectors = np.delete(vectors, index_to_remove, axis=0)
            
            # Rebuild index
            namespace_data["index"] = faiss.IndexFlatIP(self.dimension)
            if vectors.shape[0] > 0:
                namespace_data["index"].add(vectors)
            
            # Update mappings
            del namespace_data["id_to_index"][node_id]
            namespace_data["id_to_index"] = {
                node_id: (idx if idx < index_to_remove else idx - 1)
                for node_id, idx in namespace_data["id_to_index"].items()
            }
            
            self.logger.debug(f"Deleted embedding for node: {node_id}")
        except Exception as e:
            raise IndexError(f"Failed to delete node {node_id}: {e}")
    
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding from FAISS."""
        try:
            namespace_data = self.get_namespace(namespace)
            if node_id in namespace_data["id_to_index"]:
                index = namespace_data["id_to_index"][node_id]
                return namespace_data["index"].reconstruct(index)
            return None
        except Exception as e:
            raise IndexError(f"Failed to get node {node_id}: {e}")
    
    def query_similar_nodes(self, query_embedding: np.ndarray, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes from FAISS."""
        try:
            namespace_data = self.get_namespace(namespace)
            
            if namespace_data["index"].ntotal == 0:
                return []
            
            scores, indices = namespace_data["index"].search(
                np.array([query_embedding], dtype=np.float32), 
                min(top_k, namespace_data["index"].ntotal)
            )
            
            similar_nodes = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    try:
                        node_id = next(key for key, value in namespace_data["id_to_index"].items() if value == idx)
                        similar_nodes.append((node_id, float(score)))
                    except StopIteration:
                        self.logger.warning(f"No node found for index {idx} in namespace {namespace}")
            
            return similar_nodes
        except Exception as e:
            raise IndexError(f"Failed to query similar nodes: {e}")
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete FAISS index or namespace (in-memory only)."""
        try:
            if namespace is None:
                self.namespaces = {"default": {"index": faiss.IndexFlatIP(self.dimension), "id_to_index": {}}}
            elif namespace in self.namespaces:
                del self.namespaces[namespace]
            else:
                raise NamespaceError(f"Namespace {namespace} does not exist")
            
            self.logger.info(f"Deleted index/namespace: {namespace or 'all'}")
        except Exception as e:
            raise IndexError(f"Failed to delete index: {e}")


class EmbStore(IEmbService):
    """Main embedding store using Pinecone (deprecated - use EmbStoreService)."""
    
    def __init__(self, emb_api_key: str, pinecone_api_key: str, pinecone_index_name: str,
                 emb_model: str, emb_dim: int, metric: str = 'cosine',
                 cloud: str = 'aws', region: str = 'us-west-2',
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.warning("EmbStore is deprecated. Use EmbStoreService with dependency injection.")
        
        self.embedding_provider = OpenAIEmbeddingProvider(emb_api_key, emb_model, logger)
        self.vector_store = PineconeVectorStore(
            pinecone_api_key, pinecone_index_name, emb_dim, metric, cloud, region, logger
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding (delegates to provider)."""
        return self.embedding_provider.generate_embedding(text)
    

    def insert_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Insert node embedding from text."""
        embedding = self.generate_embedding(node_text)
        self.vector_store.insert_node_emb(node_id, embedding, namespace)

    def update_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Update node embedding from text."""
        embedding = self.generate_embedding(node_text)
        self.vector_store.update_node_emb(node_id, embedding, namespace)

    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding."""
        self.vector_store.delete_node_emb(node_id, namespace)

    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding."""
        return self.vector_store.get_node_emb(node_id, namespace)

    def query_similar_nodes(self, query_text: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes by text."""
        query_embedding = self.generate_embedding(query_text)
        return self.vector_store.query_similar_nodes(query_embedding, top_k, namespace)
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete index or namespace."""
        self.vector_store.delete_index(namespace)
    
    def save_index(self) -> None:
        """Save index to persistent storage."""
        self.vector_store.save_index()


class EmbStoreLocal(IEmbService):
    """Local embedding store using FAISS (deprecated - use EmbStoreService)."""
    def __init__(self, emb_api_key: str, file_store: 'IFileStore', emb_model: str, emb_dim: int, 
                 index_file_name: str = "emb_index.json", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.warning("EmbStoreLocal is deprecated. Use EmbStoreService with dependency injection.")
        
        self.embedding_provider = OpenAIEmbeddingProvider(emb_api_key, emb_model, logger)
        self.vector_store = FAISSVectorStore(emb_dim, file_store, index_file_name, logger)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding (delegates to provider)."""
        return self.embedding_provider.generate_embedding(text)

    def load_index(self) -> None:
        """Load index (delegates to vector store)."""
        pass  # Handled by FAISSVectorStore

    def save_index(self) -> None:
        """Save index (delegates to vector store)."""
        pass  # Handled by FAISSVectorStore

    def get_namespace(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get namespace (delegates to vector store)."""
        return self.vector_store.get_namespace(namespace)

    def insert_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Insert node embedding from text."""
        embedding = self.generate_embedding(node_text)
        self.vector_store.insert_node_emb(node_id, embedding, namespace)

    def update_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Update node embedding from text."""
        embedding = self.generate_embedding(node_text)
        self.vector_store.update_node_emb(node_id, embedding, namespace)

    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding."""
        self.vector_store.delete_node_emb(node_id, namespace)

    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding."""
        return self.vector_store.get_node_emb(node_id, namespace)

    def query_similar_nodes(self, query_text: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes by text."""
        query_embedding = self.generate_embedding(query_text)
        return self.vector_store.query_similar_nodes(query_embedding, top_k, namespace)

    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete index or namespace."""
        self.vector_store.delete_index(namespace)
    
    def save_index(self) -> None:
        """Save index to persistent storage."""
        self.vector_store.save_index()


class EmbStoreService(IEmbService):
    """Main embedding store service with dependency injection."""
    
    def __init__(self, embedding_provider: IEmbeddingProvider, vector_store: IVectorStore,
                 logger: Optional[logging.Logger] = None):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.logger = logger or logging.getLogger(__name__)
    
    def insert_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Insert node embedding from text."""
        embedding = self.embedding_provider.generate_embedding(node_text)
        self.vector_store.insert_node_emb(node_id, embedding, namespace)
        self.logger.debug(f"Inserted node: {node_id}")
    
    def update_node_emb(self, node_id: str, node_text: str, namespace: Optional[str] = None) -> None:
        """Update node embedding from text."""
        embedding = self.embedding_provider.generate_embedding(node_text)
        self.vector_store.update_node_emb(node_id, embedding, namespace)
        self.logger.debug(f"Updated node: {node_id}")
    
    def delete_node_emb(self, node_id: str, namespace: Optional[str] = None) -> None:
        """Delete node embedding."""
        self.vector_store.delete_node_emb(node_id, namespace)
        self.logger.debug(f"Deleted node: {node_id}")
    
    def get_node_emb(self, node_id: str, namespace: Optional[str] = None) -> Optional[np.ndarray]:
        """Get node embedding."""
        return self.vector_store.get_node_emb(node_id, namespace)
    
    def query_similar_nodes(self, query_text: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
        """Query similar nodes by text."""
        query_embedding = self.embedding_provider.generate_embedding(query_text)
        return self.vector_store.query_similar_nodes(query_embedding, top_k, namespace)
    
    def delete_index(self, namespace: Optional[str] = None) -> None:
        """Delete index or namespace."""
        self.vector_store.delete_index(namespace)
    
    def save_index(self) -> None:
        """Save index to persistent storage."""
        self.vector_store.save_index()


class EmbServiceFactory:
    """Factory for creating embedding service instances."""
    
    @staticmethod
    def create_pinecone_store(emb_api_key: str, pinecone_api_key: str, index_name: str,
                             emb_model: str, emb_dim: int, metric: str = 'cosine',
                             cloud: str = 'aws', region: str = 'us-west-2',
                             logger: Optional[logging.Logger] = None) -> EmbStoreService:
        """Create embedding store with Pinecone backend."""
        embedding_provider = OpenAIEmbeddingProvider(emb_api_key, emb_model, logger)
        vector_store = PineconeVectorStore(pinecone_api_key, index_name, emb_dim, metric, cloud, region, logger)
        return EmbStoreService(embedding_provider, vector_store, logger)
    
    @staticmethod
    def create_local_store(emb_api_key: str, file_store: 'IFileStore', emb_model: str, emb_dim: int,
                          index_file_name: str = "emb_index.json",
                          logger: Optional[logging.Logger] = None) -> EmbStoreService:
        """Create embedding store with local FAISS backend."""
        embedding_provider = OpenAIEmbeddingProvider(emb_api_key, emb_model, logger)
        vector_store = FAISSVectorStore(emb_dim, file_store, index_file_name, logger)
        return EmbStoreService(embedding_provider, vector_store, logger)
    
    @staticmethod
    def create_custom_store(embedding_provider: IEmbeddingProvider, vector_store: IVectorStore,
                           logger: Optional[logging.Logger] = None) -> EmbStoreService:
        """Create embedding store with custom providers."""
        return EmbStoreService(embedding_provider, vector_store, logger)