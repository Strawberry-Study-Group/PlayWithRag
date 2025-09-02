from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import uuid
import copy
import logging

from .file_store import IFileStore, FileStoreFactory
from .graph_store import IGraphStore, GraphStoreFactory
from .emb_store import IEmbService, EmbServiceFactory
from .constants import ConceptGraphConstants, get_default_refs, get_image_path, get_node_name_embedding_id
from .concept_operations import ConceptOperations, ConceptBuilder


class ConceptGraphError(Exception):
    """Base exception for concept graph operations."""
    pass


class ConceptNotFoundError(ConceptGraphError):
    """Raised when a requested concept is not found."""
    pass


class RelationNotFoundError(ConceptGraphError):
    """Raised when a requested relation is not found."""
    pass


class IConceptGraph(ABC):
    """Interface for concept graph operations."""
    
    @abstractmethod
    def add_concept(self, concept_name: str, concept_type: str, 
                   concept_attributes: Dict[str, Any], is_editable: bool = True,
                   image_path: Optional[str] = None, refs: Optional[Dict[str, List[str]]] = None) -> str:
        """Add a new concept to the graph."""
        pass
    
    @abstractmethod
    def update_concept(self, concept_id: str, concept_name: Optional[str] = None,
                      concept_type: Optional[str] = None, concept_attributes: Optional[Dict[str, Any]] = None,
                      image_path: Optional[str] = None, refs: Optional[Dict[str, List[str]]] = None) -> None:
        """Update an existing concept."""
        pass
    
    @abstractmethod
    def delete_concept(self, concept_id: str) -> None:
        """Delete a concept from the graph."""
        pass
    
    @abstractmethod
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID."""
        pass
    
    @abstractmethod
    def add_relation(self, source_concept_id: str, target_concept_id: str, 
                    relation_type: str, is_editable: bool = True) -> None:
        """Add a relation between concepts."""
        pass
    
    @abstractmethod
    def query_similar_concepts(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Query for similar concepts using embedding similarity."""
        pass
    
    @abstractmethod
    def get_related_concepts(self, concept_id: str, hop: int = 1, 
                           relation_type: Optional[str] = None, 
                           concept_type: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get related concepts and relations within specified hops."""
        pass


class ConceptGraphService(IConceptGraph):
    """Service for managing concept graphs with dependency injection."""
    
    def __init__(self, file_store: IFileStore, graph_store: IGraphStore, 
                 emb_store: IEmbService, logger: Optional[logging.Logger] = None):
        self.file_store = file_store
        self.graph_store = graph_store
        self.emb_store = emb_store
        self.logger = logger or logging.getLogger(__name__)
           

    def _node_to_text(self, node: Dict[str, Any]) -> str:
        """Convert node to text representation for embedding (excludes refs)."""
        return ConceptOperations.extract_text(node, include_refs=False)
    
    def _generate_concept_id(self) -> str:
        """Generate a unique concept ID."""
        concept_id = uuid.uuid4().hex
        while self.is_concept(concept_id):
            concept_id = uuid.uuid4().hex
        return concept_id
    
    def is_concept(self, concept_id: str) -> bool:
        """Check if a concept exists."""
        return self.graph_store.get_node(concept_id) is not None
    
    def is_relation(self, source_id: str, target_id: str) -> bool:
        """Check if a relation exists between two concepts."""
        return self.graph_store.get_edge(source_id, target_id) is not None 
    
    def add_concept(self, concept_name: str, concept_type: str, 
                   concept_attributes: Dict[str, Any], is_editable: bool = True,
                   image_path: Optional[str] = None, refs: Optional[Dict[str, List[str]]] = None) -> str:
        """Add a new concept to the graph."""
        concept_id = self._generate_concept_id()

        # Build concept using builder pattern with validation
        builder = ConceptBuilder(concept_id, concept_name, concept_type)
        builder.with_attributes(concept_attributes).with_editability(is_editable)
        
        # Set refs (with defaults if None)
        if refs is None:
            refs = get_default_refs()
        builder.with_refs(refs)
        
        # Handle image if provided
        if image_path:
            stored_image_path = get_image_path(concept_id)
            self.file_store.add_file(image_path, stored_image_path)
            builder.with_image_path(stored_image_path)

        # Build and validate the concept
        node = builder.build(validate=True)
        
        # Store in graph
        self.graph_store.add_node(node)
        
        # Generate embeddings (refs excluded from text representation)
        node_text = self._node_to_text(node)
        self.emb_store.insert_node_emb(concept_id, node_text, namespace=ConceptGraphConstants.NAMESPACE_FULL_NODE)
        
        node_name_embedding_id = get_node_name_embedding_id(concept_id)
        self.emb_store.insert_node_emb(node_name_embedding_id, concept_name, namespace=ConceptGraphConstants.NAMESPACE_NODE_NAME)
        
        self.logger.debug(f"Added concept: {concept_id} ({concept_name})")
        return concept_id
        
    def delete_concept(self, concept_id: str) -> None:
        """Delete a concept from the graph."""
        node = self.graph_store.get_node(concept_id)
        if not node:
            raise ConceptNotFoundError(f"Concept with ID '{concept_id}' not found")
        
        # Delete associated image file if exists
        image_path = node.get(ConceptGraphConstants.FIELD_IMAGE_PATH)
        if image_path:
            try:
                self.file_store.delete_file(image_path)
            except Exception as e:
                self.logger.warning(f"Failed to delete image file for concept {concept_id}: {e}")
        
        # Delete embeddings
        self.emb_store.delete_node_emb(concept_id, namespace=ConceptGraphConstants.NAMESPACE_FULL_NODE)
        
        node_name_embedding_id = get_node_name_embedding_id(concept_id)
        self.emb_store.delete_node_emb(node_name_embedding_id, namespace=ConceptGraphConstants.NAMESPACE_NODE_NAME)
        
        # Delete from graph
        self.graph_store.delete_node(concept_id)
        
        self.logger.debug(f"Deleted concept: {concept_id}")
        

    def update_concept(self, concept_id: str, concept_name: Optional[str] = None,
                      concept_type: Optional[str] = None, concept_attributes: Optional[Dict[str, Any]] = None,
                      image_path: Optional[str] = None, refs: Optional[Dict[str, List[str]]] = None) -> None:
        """Update an existing concept."""
        node = self.graph_store.get_node(concept_id)
        if not node:
            raise ConceptNotFoundError(f"Concept with ID '{concept_id}' not found")
        
        # Update fields if provided
        if concept_name:
            node[ConceptGraphConstants.FIELD_NODE_NAME] = concept_name
            node_name_embedding_id = get_node_name_embedding_id(concept_id)
            self.emb_store.update_node_emb(node_name_embedding_id, concept_name, namespace=ConceptGraphConstants.NAMESPACE_NODE_NAME)
        
        if concept_type:
            node[ConceptGraphConstants.FIELD_NODE_TYPE] = concept_type
        
        if concept_attributes:
            node[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES].update(concept_attributes)
        
        if refs:
            node[ConceptGraphConstants.FIELD_REFS] = refs

        if image_path:
            node[ConceptGraphConstants.FIELD_IMAGE_PATH] = image_path
        
        # Validate updated node
        if not ConceptOperations.validate_concept(node, self.logger):
            errors = ConceptOperations.get_validation_errors(node, self.logger)
            raise ValueError(f"Updated concept validation failed: {', '.join(errors)}")
        
        self.graph_store.update_node(node)
        
        # Update full node embedding (refs excluded from text)
        node_text = self._node_to_text(node)
        self.emb_store.update_node_emb(concept_id, node_text, namespace=ConceptGraphConstants.NAMESPACE_FULL_NODE)
        
        self.logger.debug(f"Updated concept: {concept_id}")
        

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID."""
        return self.graph_store.get_node(concept_id)

    def add_relation(self, source_concept_id: str, target_concept_id: str, 
                    relation_type: str, is_editable: bool = True) -> None:
        """Add a relation between concepts."""
        # Verify concepts exist
        if not self.is_concept(source_concept_id):
            raise ConceptNotFoundError(f"Source concept '{source_concept_id}' not found")
        if not self.is_concept(target_concept_id):
            raise ConceptNotFoundError(f"Target concept '{target_concept_id}' not found")
        
        edge = {
            ConceptGraphConstants.FIELD_SOURCE_NODE_ID: source_concept_id,
            ConceptGraphConstants.FIELD_TARGET_NODE_ID: target_concept_id,
            ConceptGraphConstants.FIELD_EDGE_TYPE: relation_type,
            ConceptGraphConstants.FIELD_IS_EDITABLE: is_editable,
        }
        self.graph_store.add_edge(edge)
        
        self.logger.debug(f"Added relation: {source_concept_id} -> {target_concept_id} ({relation_type})")

    def delete_relation(self, source_concept_id: str, target_concept_id: str) -> None:
        """Delete a relation between concepts."""
        if not self.is_relation(source_concept_id, target_concept_id):
            raise RelationNotFoundError(f"Relation between '{source_concept_id}' and '{target_concept_id}' not found")
        
        self.graph_store.delete_edge(source_concept_id, target_concept_id)
        
        self.logger.debug(f"Deleted relation: {source_concept_id} -> {target_concept_id}")

    def update_relation(self, source_concept_id: str, target_concept_id: str, 
                       relation_type: Optional[str] = None) -> None:
        """Update a relation between concepts."""
        edge = self.graph_store.get_edge(source_concept_id, target_concept_id)
        if not edge:
            raise RelationNotFoundError(f"Relation between '{source_concept_id}' and '{target_concept_id}' not found")
        
        if relation_type:
            edge[ConceptGraphConstants.FIELD_EDGE_TYPE] = relation_type
            self.graph_store.update_edge(edge)
            self.logger.debug(f"Updated relation: {source_concept_id} -> {target_concept_id} ({relation_type})")

    def get_relation(self, source_concept_id: str, target_concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a relation between concepts."""
        return self.graph_store.get_edge(source_concept_id, target_concept_id)

    def query_similar_concepts(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Query for similar concepts using embedding similarity."""
        similar_nodes = self.emb_store.query_similar_nodes(query_text, top_k, namespace=ConceptGraphConstants.NAMESPACE_FULL_NODE)
        similar_names = self.emb_store.query_similar_nodes(query_text, top_k, namespace=ConceptGraphConstants.NAMESPACE_NODE_NAME)
        
        similar_concepts = []
        
        # Add results from full node embeddings
        for node_id, score in similar_nodes:
            concept = self.get_concept(node_id)
            if concept:
                similar_concepts.append((concept, score))
        
        # Add results from node name embeddings
        for node_id_with_name, score in similar_names:
            node_id = node_id_with_name.split(ConceptGraphConstants.NODE_NAME_SEPARATOR)[0]
            concept = self.get_concept(node_id)
            if concept:
                similar_concepts.append((concept, score))
        
        # Remove duplicates and sort by score
        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        ranked_concepts = []
        
        for concept, score in similar_concepts:
            if concept[ConceptGraphConstants.FIELD_NODE_ID] not in seen_ids:
                ranked_concepts.append((concept, score))
                seen_ids.add(concept[ConceptGraphConstants.FIELD_NODE_ID])

        return ranked_concepts[:top_k]
    
    def get_related_concepts(self, concept_id: str, hop: int = 1, 
                           relation_type: Optional[str] = None, 
                           concept_type: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get related concepts and relations within specified hops."""
        if not self.is_concept(concept_id):
            raise ConceptNotFoundError(f"Concept with ID '{concept_id}' not found")
        
        related_concepts = []
        related_relations = []
        
        node_list, edge_list = self.graph_store.get_neighbor_info(concept_id, hop=hop)
        
        # Filter and process nodes
        for node in node_list:
            concept = copy.deepcopy(node)
            if concept:
                if concept_type and concept[ConceptGraphConstants.FIELD_NODE_TYPE] != concept_type:
                    continue
                related_concepts.append(concept)

        # Filter and process edges
        for edge in edge_list:  
            if relation_type and edge[ConceptGraphConstants.FIELD_EDGE_TYPE] != relation_type:
                continue
            
            relation = copy.deepcopy(edge)
            relation["relation_id"] = self.graph_store.parse_edge_key(
                relation[ConceptGraphConstants.FIELD_SOURCE_NODE_ID], 
                relation[ConceptGraphConstants.FIELD_TARGET_NODE_ID]
            )
            
            # Get concept names for relation
            source_node = self.graph_store.get_node(relation[ConceptGraphConstants.FIELD_SOURCE_NODE_ID])
            target_node = self.graph_store.get_node(relation[ConceptGraphConstants.FIELD_TARGET_NODE_ID])
            
            if source_node and target_node:
                relation["source_concept"] = source_node[ConceptGraphConstants.FIELD_NODE_NAME]
                relation["target_concept"] = target_node[ConceptGraphConstants.FIELD_NODE_NAME]
                related_relations.append(relation)

        return related_concepts, related_relations
    
    def get_concept_id_by_name(self, concept_name: str) -> Optional[str]:
        """Get concept ID by name."""
        # Use proper encapsulated method instead of direct graph access
        node = self.graph_store.get_node_by_name(concept_name)
        if node:
            return node[ConceptGraphConstants.FIELD_NODE_ID]
        return None
               
    def save_graph(self) -> None:
        """Save the graph and embeddings to persistent storage."""
        self.graph_store.save_graph()
        self.emb_store.save_index()
        self.logger.debug("Graph and embeddings saved")

    def empty_graph(self) -> None:
        """Empty the entire graph and reset all stores (in-memory only)."""
        try:
            self.graph_store.delete_graph()
            self.emb_store.delete_index()
            
            # Delete file prefix if file store supports it
            if hasattr(self.file_store, 'delete_prefix'):
                self.file_store.delete_prefix()
            
            self.logger.info("Graph emptied successfully (in-memory only)")
        except Exception as e:
            self.logger.error(f"Failed to empty graph: {e}")
            raise ConceptGraphError(f"Failed to empty graph: {e}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the concept graph."""
        all_nodes = self.graph_store.get_all_nodes()
        return ConceptOperations.collect_statistics(all_nodes)
    
    def search_concepts(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search concepts by criteria using visitor pattern."""
        all_nodes = self.graph_store.get_all_nodes()
        return ConceptOperations.search_concepts(all_nodes, criteria)
    
    def validate_all_concepts(self) -> Dict[str, List[str]]:
        """Validate all concepts and return any validation errors."""
        all_nodes = self.graph_store.get_all_nodes()
        validation_results = {}
        
        for node in all_nodes:
            concept_id = node[ConceptGraphConstants.FIELD_NODE_ID]
            if not ConceptOperations.validate_concept(node, self.logger):
                errors = ConceptOperations.get_validation_errors(node, self.logger)
                validation_results[concept_id] = errors
        
        return validation_results


class ConceptGraphFactory:
    """Factory for creating concept graph instances."""
    
    @staticmethod
    def create_from_memory_core(memory_core_path: str,
                               memory_core_config: Dict[str, Any],
                               logger: Optional[logging.Logger] = None) -> ConceptGraphService:
        """Create concept graph from a memory core folder path.
        
        Args:
            memory_core_path: Path to the memory core folder containing all data
            memory_core_config: Configuration including embedding store and file names
            logger: Optional logger instance
            
        Returns:
            ConceptGraphService: Configured concept graph service
            
        Note:
            All files (graph, embeddings, images) are stored in the memory core folder.
            
        Expected memory_core_config structure:
        {
            "embedding": {
                "provider": "local" | "remote",
                "api_key": "...",
                "model": "text-embedding-3-small",
                "dim": 1536,
                # for remote only:
                "pinecone_api_key": "...",
                "pinecone_index_name": "...",
                "metric": "cosine"
            },
            "files": {
                "schema_file": "schema.json" (optional),
                "graph_file": "graph.json",
                "index_file": "emb_index.json"
            }
        }
        """
        # Extract config sections
        embedding_config = memory_core_config["embedding"]
        files_config = memory_core_config.get("files", {})
        
        # Get file names with defaults
        schema_file = files_config.get("schema_file")
        graph_file_name = files_config.get("graph_file", "graph.json")
        index_file_name = files_config.get("index_file", "emb_index.json")
        
        # Create file store pointing directly to memory core
        file_store = FileStoreFactory.create_local_store(
            memory_core_path, "", logger  # Empty prefix - use memory core folder directly
        )
        
        # Create graph store
        graph_store = GraphStoreFactory.create_local_store(
            file_store, schema_file, graph_file_name, logger
        )
        
        # Create embedding store
        if embedding_config["provider"] == "remote":
            emb_store = EmbServiceFactory.create_pinecone_store(
                embedding_config["api_key"],
                embedding_config["pinecone_api_key"],
                embedding_config["pinecone_index_name"],
                embedding_config["model"],
                embedding_config["dim"],
                embedding_config.get("metric", "cosine"),
                logger=logger
            )
        elif embedding_config["provider"] == "local":
            emb_store = EmbServiceFactory.create_local_store(
                embedding_config["api_key"],
                file_store,
                embedding_config["model"],
                embedding_config["dim"],
                index_file_name,
                logger
            )
        else:
            raise ValueError(f"Invalid embedding store provider: {embedding_config['provider']}")
        
        return ConceptGraphService(file_store, graph_store, emb_store, logger)
    
    @staticmethod
    def create_custom(file_store: IFileStore, graph_store: IGraphStore, 
                     emb_store: IEmbService, logger: Optional[logging.Logger] = None) -> ConceptGraphService:
        """Create concept graph with custom components."""
        return ConceptGraphService(file_store, graph_store, emb_store, logger)
    


