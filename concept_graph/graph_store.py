from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
import logging
from pathlib import Path

from .file_store import IFileStore


class GraphStoreError(Exception):
    """Base exception for graph store operations."""
    pass


class ValidationError(GraphStoreError):
    """Raised when data validation fails."""
    pass


class NodeNotFoundError(GraphStoreError):
    """Raised when a requested node is not found."""
    pass


class EdgeNotFoundError(GraphStoreError):
    """Raised when a requested edge is not found."""
    pass


class DuplicateError(GraphStoreError):
    """Raised when attempting to create duplicate nodes/edges."""
    pass


class SchemaError(GraphStoreError):
    """Raised when schema validation fails."""
    pass


class IGraphValidator(ABC):
    """Interface for graph data validation."""
    
    @abstractmethod
    def validate_node(self, node: Dict[str, Any]) -> bool:
        """Validate node data against schema."""
        pass
    
    @abstractmethod
    def validate_edge(self, edge: Dict[str, Any]) -> bool:
        """Validate edge data against schema."""
        pass
    
    @abstractmethod
    def validate_graph(self, graph: Dict[str, Any]) -> bool:
        """Validate entire graph structure."""
        pass


class IGraphStore(ABC):
    """Interface for graph storage operations."""
    
    @abstractmethod
    def load_graph(self) -> Dict[str, Any]:
        """Load graph from storage."""
        pass
    
    @abstractmethod
    def save_graph(self) -> None:
        """Save graph to storage."""
        pass
    
    @abstractmethod
    def add_node(self, node: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        pass
    
    @abstractmethod
    def update_node(self, node: Dict[str, Any]) -> None:
        """Update an existing node."""
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        pass
    
    @abstractmethod
    def add_edge(self, edge: Dict[str, Any]) -> None:
        """Add an edge to the graph."""
        pass
    
    @abstractmethod
    def update_edge(self, edge: Dict[str, Any]) -> None:
        """Update an existing edge."""
        pass
    
    @abstractmethod
    def delete_edge(self, source_node_id: str, target_node_id: str) -> None:
        """Delete an edge from the graph."""
        pass
    
    @abstractmethod
    def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by source and target node IDs."""
        pass
    
    @abstractmethod
    def get_neighbor_info(self, node_id: str, hop: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get neighbor nodes and edges within specified hops."""
        pass
    
    @abstractmethod
    def delete_graph(self) -> None:
        """Delete the entire graph."""
        pass
    
    @abstractmethod
    def get_node_by_name(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get a node by name."""
        pass
    
    @abstractmethod
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the graph."""
        pass


class SchemaValidator(IGraphValidator):
    """JSON schema-based validator for graph data."""
    
    def __init__(self, schema_file: str, logger: Optional[logging.Logger] = None):
        self.schema_file = schema_file
        self.logger = logger or logging.getLogger(__name__)
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from JSON file."""
        try:
            with open(self.schema_file, 'r') as f:
                schema = json.load(f)
            self.logger.info(f"Loaded schema from {self.schema_file}")
            return schema
        except Exception as e:
            raise SchemaError(f"Failed to load schema from {self.schema_file}: {e}")
    
    def _validate_required_fields(self, data: Dict[str, Any], required_fields: Dict[str, Any]) -> bool:
        """Validate that all required fields are present and have correct types."""
        for field_name, field_spec in required_fields.items():
            if field_spec.get('required', True):
                if field_name not in data:
                    raise ValidationError(f"Missing required field: {field_name}")
            
            if field_name in data:
                expected_type = field_spec['type']
                value = data[field_name]
                
                if not self._validate_type(value, expected_type, field_spec):
                    raise ValidationError(f"Invalid type for field {field_name}: expected {expected_type}")
        
        return True
    
    def _validate_type(self, value: Any, expected_type: str, field_spec: Dict[str, Any]) -> bool:
        """Validate value type and constraints."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'boolean': bool,
            'object': dict,
            'array': list
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            return False
        
        # Additional validation for constraints
        if 'constraints' in field_spec:
            constraints = field_spec['constraints']
            if expected_type == 'string' and isinstance(value, str):
                if 'min_length' in constraints and len(value) < constraints['min_length']:
                    return False
                if 'max_length' in constraints and len(value) > constraints['max_length']:
                    return False
            elif expected_type == 'number' and isinstance(value, (int, float)):
                if 'min' in constraints and value < constraints['min']:
                    return False
                if 'max' in constraints and value > constraints['max']:
                    return False
        
        # Validate allowed values
        if 'allowed_values' in field_spec:
            if value not in field_spec['allowed_values']:
                return False
        
        return True
    
    def _apply_defaults(self, data: Dict[str, Any], required_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing optional fields."""
        result = data.copy()
        
        for field_name, field_spec in required_fields.items():
            if field_name not in result and 'default' in field_spec:
                if callable(field_spec['default']):
                    result[field_name] = field_spec['default']()
                else:
                    result[field_name] = field_spec['default']
        
        return result
    
    def validate_node(self, node: Dict[str, Any]) -> bool:
        """Validate node data against schema."""
        required_fields = self.schema['node']['required_fields']
        return self._validate_required_fields(node, required_fields)
    
    def validate_edge(self, edge: Dict[str, Any]) -> bool:
        """Validate edge data against schema."""
        required_fields = self.schema['edge']['required_fields']
        return self._validate_required_fields(edge, required_fields)
    
    def validate_graph(self, graph: Dict[str, Any]) -> bool:
        """Validate entire graph structure."""
        required_fields = self.schema['graph']['required_fields']
        return self._validate_required_fields(graph, required_fields)
    
    def normalize_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply defaults and normalize node data."""
        required_fields = self.schema['node']['required_fields']
        result = self._apply_defaults(node, required_fields)
        
        # Add timestamps if not present
        now = datetime.utcnow().isoformat()
        if 'created_at' not in result:
            result['created_at'] = now
        result['updated_at'] = now
        
        # Ensure refs field exists
        if 'refs' not in result:
            result['refs'] = {
                'ref_img': [],
                'ref_audio': [],
                'ref_video': [],
                'ref_docs': []
            }
        
        return result
    
    def normalize_edge(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        """Apply defaults and normalize edge data."""
        required_fields = self.schema['edge']['required_fields']
        result = self._apply_defaults(edge, required_fields)
        
        # Add timestamps if not present
        now = datetime.utcnow().isoformat()
        if 'created_at' not in result:
            result['created_at'] = now
        result['updated_at'] = now
        
        return result


class GraphStore(IGraphStore):
    def __init__(self, file_store: IFileStore, validator: Optional[IGraphValidator] = None, 
                 graph_file_name: str = "graph.json", logger: Optional[logging.Logger] = None):
        self.file_store = file_store
        self.graph_file_name = graph_file_name
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)
        self.graph = self.load_graph()

    def load_graph(self) -> Dict[str, Any]:
        """Load graph from storage."""
        try:
            if self.file_store.file_exists(self.graph_file_name):
                temp_file = f"temp_{self.graph_file_name}"
                self.file_store.get_file(self.graph_file_name, temp_file)
                with open(temp_file, "r") as file:
                    graph = json.load(file)
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)
                
                # Validate loaded graph
                if self.validator:
                    try:
                        self.validator.validate_graph(graph)
                    except ValidationError as e:
                        self.logger.warning(f"Loaded graph failed validation: {e}")
                
                self.logger.info(f"Loaded graph with {len(graph.get('node_dict', {}))} nodes")
            else:
                graph = self._create_empty_graph()
                self.logger.info("Created new empty graph")
        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            graph = self._create_empty_graph()
        
        return graph
    
    def _create_empty_graph(self) -> Dict[str, Any]:
        """Create an empty graph structure."""
        return {
            "node_dict": {},
            "edge_dict": {},
            "neighbor_dict": {},
            "node_name_to_id": {},
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "description": ""
            }
        }

    def delete_graph(self) -> None:
        """Delete the entire graph."""
        try:
            if self.file_store.file_exists(self.graph_file_name):
                self.file_store.delete_file(self.graph_file_name)
            self.graph = self._create_empty_graph()
            self.logger.info("Graph deleted successfully")
        except Exception as e:
            self.logger.warning(f"Failed to delete graph file: {e}")
            self.graph = self._create_empty_graph()

    def parse_edge_key(self, source_node_id: str, target_node_id: str, is_directed: bool = False) -> str:
        """Generate edge key for storage."""
        if is_directed:
            return f"{source_node_id}->{target_node_id}"
        else:
            keys = [source_node_id, target_node_id]
            keys.sort()
            return f"{keys[0]}<->{keys[1]}"

    def save_graph(self) -> None:
        """Save graph to storage."""
        try:
            # Update metadata
            self.graph["metadata"]["updated_at"] = datetime.utcnow().isoformat()
            
            temp_file = f"temp_{self.graph_file_name}"
            with open(temp_file, "w") as file:
                json.dump(self.graph, file, indent=2)
            
            self.file_store.update_file(temp_file, self.graph_file_name)
            Path(temp_file).unlink(missing_ok=True)  # Clean up temp file
            
            self.logger.debug("Graph saved successfully")
        except Exception as e:
            raise GraphStoreError(f"Failed to save graph: {e}")

    def is_valid_node(self, node: Dict[str, Any]) -> bool:
        """Validate node data."""
        if self.validator:
            try:
                return self.validator.validate_node(node)
            except ValidationError as e:
                self.logger.error(f"Node validation failed: {e}")
                return False
        else:
            # Fallback validation
            required_keys = ["node_id", "node_name", "node_type", "node_attributes", "is_editable"]
            return all(key in node for key in required_keys)

    def is_valid_edge(self, edge: Dict[str, Any]) -> bool:
        """Validate edge data."""
        if self.validator:
            try:
                return self.validator.validate_edge(edge)
            except ValidationError as e:
                self.logger.error(f"Edge validation failed: {e}")
                return False
        else:
            # Fallback validation
            required_keys = ["source_node_id", "target_node_id", "edge_type", "is_editable"]
            return all(key in edge for key in required_keys)

    def add_node(self, node: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        # Normalize and validate node
        if self.validator and hasattr(self.validator, 'normalize_node'):
            node = self.validator.normalize_node(node)
        
        if not self.is_valid_node(node):
            raise ValidationError("Invalid node format")
        
        node_id = node["node_id"]
        node_name = node["node_name"]
        
        if node_id in self.graph["node_dict"]:
            raise DuplicateError(f"Node with ID '{node_id}' already exists")
        if node_name in self.graph["node_name_to_id"]:
            raise DuplicateError(f"Node with name '{node_name}' already exists")
        
        self.graph["node_dict"][node_id] = node
        self.graph["neighbor_dict"][node_id] = []
        self.graph["node_name_to_id"][node_name] = node_id
        
        self.logger.debug(f"Added node: {node_id}")
        self.save_graph()

    def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        if node_id not in self.graph["node_dict"]:
            raise NodeNotFoundError(f"Node with ID '{node_id}' not found")
        
        # Remove from name mapping
        node_name = self.graph["node_dict"][node_id]["node_name"]
        if node_name in self.graph["node_name_to_id"]:
            del self.graph["node_name_to_id"][node_name]
        
        # Remove node
        del self.graph["node_dict"][node_id]
        del self.graph["neighbor_dict"][node_id]
        
        # Remove all edges involving this node
        edges_to_remove = []
        for edge_key in self.graph["edge_dict"].keys():
            if node_id in edge_key:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.graph["edge_dict"][edge_key]
        
        # Remove from neighbor lists
        for neighbor_list in self.graph["neighbor_dict"].values():
            if node_id in neighbor_list:
                neighbor_list.remove(node_id)
        
        self.logger.debug(f"Deleted node: {node_id}")
        self.save_graph()

    def update_node(self, node: Dict[str, Any]) -> None:
        """Update an existing node."""
        node_id = node.get("node_id")
        if not node_id:
            raise ValidationError("Node ID is required for update")
        
        if node_id not in self.graph["node_dict"]:
            raise NodeNotFoundError(f"Node with ID '{node_id}' not found")
        
        # Preserve created_at timestamp
        existing_node = self.graph["node_dict"][node_id]
        if 'created_at' in existing_node and 'created_at' not in node:
            node['created_at'] = existing_node['created_at']
        
        # Normalize and validate
        if self.validator and hasattr(self.validator, 'normalize_node'):
            node = self.validator.normalize_node(node)
        
        if not self.is_valid_node(node):
            raise ValidationError("Invalid node format")
        
        # Handle name changes
        old_name = existing_node["node_name"]
        new_name = node["node_name"]
        if old_name != new_name:
            if new_name in self.graph["node_name_to_id"]:
                raise DuplicateError(f"Node with name '{new_name}' already exists")
            del self.graph["node_name_to_id"][old_name]
            self.graph["node_name_to_id"][new_name] = node_id
        
        self.graph["node_dict"][node_id] = node
        self.logger.debug(f"Updated node: {node_id}")
        self.save_graph()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        return self.graph["node_dict"].get(node_id)

    def add_edge(self, edge: Dict[str, Any]) -> None:
        """Add an edge to the graph."""
        # Normalize and validate edge
        if self.validator and hasattr(self.validator, 'normalize_edge'):
            edge = self.validator.normalize_edge(edge)
        
        if not self.is_valid_edge(edge):
            raise ValidationError("Invalid edge format")
        
        source_id = edge["source_node_id"]
        target_id = edge["target_node_id"]
        is_directed = edge.get("is_directed", False)
        
        # Validate that nodes exist
        if source_id not in self.graph["node_dict"]:
            raise NodeNotFoundError(f"Source node '{source_id}' not found")
        if target_id not in self.graph["node_dict"]:
            raise NodeNotFoundError(f"Target node '{target_id}' not found")
        
        # Check for self-loops if not allowed
        if source_id == target_id:
            raise ValidationError("Self-loops are not allowed")
        
        edge_key = self.parse_edge_key(source_id, target_id, is_directed)
        if edge_key in self.graph["edge_dict"]:
            raise DuplicateError(f"Edge '{edge_key}' already exists")
        
        self.graph["edge_dict"][edge_key] = edge
        
        # Update neighbor lists
        if target_id not in self.graph["neighbor_dict"][source_id]:
            self.graph["neighbor_dict"][source_id].append(target_id)
        
        if not is_directed and source_id not in self.graph["neighbor_dict"][target_id]:
            self.graph["neighbor_dict"][target_id].append(source_id)
        
        self.logger.debug(f"Added edge: {edge_key}")
        self.save_graph()

    def delete_edge(self, source_node_id: str, target_node_id: str) -> None:
        """Delete an edge from the graph."""
        # Try both directed and undirected edge keys
        edge_key_directed = self.parse_edge_key(source_node_id, target_node_id, True)
        edge_key_undirected = self.parse_edge_key(source_node_id, target_node_id, False)
        
        edge_key = None
        edge = None
        
        if edge_key_directed in self.graph["edge_dict"]:
            edge_key = edge_key_directed
            edge = self.graph["edge_dict"][edge_key]
        elif edge_key_undirected in self.graph["edge_dict"]:
            edge_key = edge_key_undirected
            edge = self.graph["edge_dict"][edge_key]
        
        if not edge_key:
            raise EdgeNotFoundError(f"Edge between '{source_node_id}' and '{target_node_id}' not found")
        
        is_directed = edge.get("is_directed", False)
        
        # Remove edge
        del self.graph["edge_dict"][edge_key]
        
        # Update neighbor lists
        if target_node_id in self.graph["neighbor_dict"][source_node_id]:
            self.graph["neighbor_dict"][source_node_id].remove(target_node_id)
        
        if not is_directed and source_node_id in self.graph["neighbor_dict"][target_node_id]:
            self.graph["neighbor_dict"][target_node_id].remove(source_node_id)
        
        self.logger.debug(f"Deleted edge: {edge_key}")
        self.save_graph()

    def update_edge(self, edge: Dict[str, Any]) -> None:
        """Update an existing edge."""
        source_id = edge.get("source_node_id")
        target_id = edge.get("target_node_id")
        
        if not source_id or not target_id:
            raise ValidationError("Source and target node IDs are required for edge update")
        
        # Find existing edge
        edge_key_directed = self.parse_edge_key(source_id, target_id, True)
        edge_key_undirected = self.parse_edge_key(source_id, target_id, False)
        
        existing_edge = None
        edge_key = None
        
        if edge_key_directed in self.graph["edge_dict"]:
            edge_key = edge_key_directed
            existing_edge = self.graph["edge_dict"][edge_key]
        elif edge_key_undirected in self.graph["edge_dict"]:
            edge_key = edge_key_undirected
            existing_edge = self.graph["edge_dict"][edge_key]
        
        if not existing_edge:
            raise EdgeNotFoundError(f"Edge between '{source_id}' and '{target_id}' not found")
        
        # Preserve created_at timestamp
        if 'created_at' in existing_edge and 'created_at' not in edge:
            edge['created_at'] = existing_edge['created_at']
        
        # Normalize and validate
        if self.validator and hasattr(self.validator, 'normalize_edge'):
            edge = self.validator.normalize_edge(edge)
        
        if not self.is_valid_edge(edge):
            raise ValidationError("Invalid edge format")
        
        self.graph["edge_dict"][edge_key] = edge
        self.logger.debug(f"Updated edge: {edge_key}")
        self.save_graph()

    def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by source and target node IDs."""
        # Try both directed and undirected edge keys
        edge_key_directed = self.parse_edge_key(source_node_id, target_node_id, True)
        edge_key_undirected = self.parse_edge_key(source_node_id, target_node_id, False)
        
        edge = self.graph["edge_dict"].get(edge_key_directed)
        if edge:
            return edge
        
        return self.graph["edge_dict"].get(edge_key_undirected)

    def get_neighbor_info(self, node_id: str, hop: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get neighbor nodes and edges within specified hops."""
        if node_id not in self.graph["node_dict"]:
            raise NodeNotFoundError(f"Node with ID '{node_id}' not found")
        
        if hop == 1:
            neighbor_ids = self.graph["neighbor_dict"].get(node_id, [])
            node_list = [self.graph["node_dict"][neighbor_id] for neighbor_id in neighbor_ids 
                        if neighbor_id in self.graph["node_dict"]]
            
            edge_list = []
            for neighbor_id in neighbor_ids:
                edge = self.get_edge(node_id, neighbor_id)
                if edge:
                    edge_list.append(edge)
            
            return node_list, edge_list
        else:
            visited = set()
            node_list = []
            edge_list = []
            queue = [node_id]
            
            for _ in range(hop):
                new_queue = []
                for node in queue:
                    if node not in visited:
                        visited.add(node)
                        neighbor_ids = self.graph["neighbor_dict"].get(node, [])
                        for neighbor_id in neighbor_ids:
                            if neighbor_id not in visited and neighbor_id in self.graph["node_dict"]:
                                node_list.append(self.graph["node_dict"][neighbor_id])
                                edge = self.get_edge(node, neighbor_id)
                                if edge:
                                    edge_list.append(edge)
                                new_queue.append(neighbor_id)
                queue = new_queue
            
            return node_list, edge_list
    
    def get_node_by_name(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get a node by name."""
        node_id = self.graph["node_name_to_id"].get(node_name)
        if node_id:
            return self.graph["node_dict"].get(node_id)
        return None
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the graph."""
        return list(self.graph["node_dict"].values())


class GraphStoreFactory:
    """Factory for creating graph store instances."""
    
    @staticmethod
    def create_local_store(file_store: IFileStore, schema_file: Optional[str] = None,
                          graph_file_name: str = "graph.json",
                          logger: Optional[logging.Logger] = None) -> GraphStore:
        """Create a local file-based graph store."""
        validator = None
        if schema_file:
            validator = SchemaValidator(schema_file, logger)
        
        return GraphStore(file_store, validator, graph_file_name, logger)
    
    @staticmethod
    def create_cloud_store(provider: str, **config) -> IGraphStore:
        """Create a cloud-based graph store (placeholder for future implementation)."""
        raise NotImplementedError(f"Cloud graph store provider '{provider}' not yet implemented")