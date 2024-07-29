import json
from .file_store import LocalFileStore
from .graph_store import GraphStore
from .emb_store import EmbStore
import uuid
import os
import copy 

class ConceptGraph:
    def __init__(self,
                concept_graph_config, save_file_config):       
        if save_file_config["provider"] == "local":
            self.file_store = LocalFileStore(save_file_config["save_path"], "graph/")
        elif save_file_config["provider"] == "gcp":
            raise NotImplementedError("gcp file store is not implemented yet.")
        else:
            raise ValueError("Invalid file store provider")
        
        self.graph_store = GraphStore(self.file_store)
        self.emb_store = EmbStore(concept_graph_config["openai_api_key"],
                                  concept_graph_config["pinecone_api_key"],
                                  concept_graph_config["pinecone_index_name"],
                                  concept_graph_config["emb_model"],
                                  concept_graph_config["emb_dim"])

        self.openai_api_key = concept_graph_config["openai_api_key"]
        self.pinecone_api_key = concept_graph_config["pinecone_api_key"]
        self.pinecone_index_name = concept_graph_config["pinecone_index_name"]
        self.emb_model = concept_graph_config["emb_model"]
        self.emb_dim = concept_graph_config["emb_dim"]
           

    def node_to_text(self, node):
        text = f"Node: {node['node_name']}\nType: {node['node_type']}\nAttributes: {json.dumps(node['node_attributes'])}"
        return text
    
    def generate_concept_id(self):
        concept_id = uuid.uuid4().hex
        while concept_id in self.graph_store.graph["node_dict"]:
            concept_id = uuid.uuid4().hex
        return concept_id
    
    def is_concept(self, concept_id):
        return concept_id in self.graph_store.graph["node_dict"]
    
    def is_relation(self, relation_id):
        return relation_id in self.graph_store.graph["edge_dict"] 
    
    def add_concept(self, concept_name, concept_type, concept_attributes, is_editable = True,image_path=None):
        concept_id = self.generate_concept_id()

        if type(concept_attributes) is not dict:
            raise ValueError("Concept attributes should be a dictionary.")
        
        node = {
            "node_id": concept_id,
            "node_name": concept_name,
            "node_type": concept_type,
            "is_editable": is_editable,
            "node_attributes": concept_attributes
        }
        if image_path:
            self.file_store.add_file(image_path, f"{concept_id}.jpg")
            node["image_path"] = f"{concept_id}.jpg"
        else:
            node["image_path"] = None

        self.graph_store.add_node(node)
        node_text = self.node_to_text(node)
        self.emb_store.insert_node_emb(concept_id, node_text, namespace="full_node")
        node_name_text = concept_name
        self.emb_store.insert_node_emb(concept_id+"||node_name", node_name_text, namespace="node_name")
        
    def delete_concept(self, concept_id):
        if self.graph_store.graph["node_dict"][concept_id].get("image_path"):
            self.file_store.delete_file(f"{concept_id}.jpg")
        self.emb_store.delete_node_emb(concept_id, namespace="full_node")
        self.emb_store.delete_node_emb(concept_id+"||node_name", namespace="node_name")
        self.graph_store.delete_node(concept_id)
        

    def update_concept(self, concept_id, concept_name=None, concept_type=None, concept_attributes=None, image_path=None):
        node = self.graph_store.get_node(concept_id)
        if concept_name:
            node["node_name"] = concept_name
            node_name_text = node["node_name"]
            self.emb_store.update_node_emb(concept_id+"||node_name", node_name_text, namespace="node_name")
        if concept_type:
            node["node_type"] = concept_type
        if concept_attributes:
            node["node_attributes"].update(concept_attributes)

        if image_path:
            self.file_store.add_file(image_path, f"{concept_id}.jpg")
            node["image_path"] = f"{concept_id}.jpg"
        self.graph_store.update_node(node)
        node_text = self.node_to_text(node)
        self.emb_store.update_node_emb(concept_id, node_text, namespace="full_node")
        

    def get_concept(self, concept_id):
        node = self.graph_store.get_node(concept_id)
        return node

    def add_relation(self, source_concept_id, target_concept_id, relation_type, is_editable = True):
        edge = {
            "source_node_id": source_concept_id,
            "target_node_id": target_concept_id,
            "edge_type": relation_type,
            "is_editable": is_editable,
        }
        self.graph_store.add_edge(edge)

    def delete_relation(self, source_concept_id, target_concept_id):
        self.graph_store.delete_edge(source_concept_id, target_concept_id)

    def update_relation(self, source_concept_id, target_concept_id, relation_type=None, is_editable=True):
        edge = self.graph_store.get_edge(source_concept_id, target_concept_id)
        if relation_type:
            edge["edge_type"] = relation_type

        self.graph_store.update_edge(edge)

    def get_relation(self, source_concept_id, target_concept_id):
        edge = self.graph_store.get_edge(source_concept_id, target_concept_id)
        return edge

    def query_similar_concepts(self, query_text, top_k=5):
        similar_nodes = self.emb_store.query_similar_nodes(query_text, top_k, namespace="full_node")
        similar_names = self.emb_store.query_similar_nodes(query_text, top_k, namespace="node_name")
        similar_concepts = []
        for node_id, score in similar_nodes:
            concept = self.get_concept(node_id)
            if concept:
                similar_concepts.append((concept, score))
        for node_id_with_name, score in similar_names:
            node_id = node_id_with_name.split("||")[0]
            concept = self.get_concept(node_id)
            if concept:
                similar_concepts.append((concept, score))
        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        ranked_concepts = []
        for concept, score in similar_concepts:
            if concept["node_id"] not in seen_ids:
                ranked_concepts.append((concept, score))
                seen_ids.add(concept["node_id"])

        ranked_concepts = ranked_concepts[:top_k]
        return ranked_concepts
    
    def get_related_concepts(self, concept_id, hop=1, relation_type=None, concept_type=None):
        related_concepts = []
        related_relations = []
        node_list, edge_list = self.graph_store.get_neighbor_info(concept_id, hop=hop)
        for node in node_list:
            concept = copy.deepcopy(node)
            if concept:
                if concept_type and concept["node_type"] != concept_type:
                    continue
                related_concepts.append(concept)

        for edge in edge_list:  
            if relation_type and edge["edge_type"] != relation_type:
                continue
            relation = copy.deepcopy(edge)
            relation["relation_id"] = self.graph_store.parse_edge_key(relation["source_node_id"], relation["target_node_id"])
            relation["source_concept"] = self.graph_store.graph["node_dict"][relation["source_node_id"]]["node_name"]
            relation["target_concept"] = self.graph_store.graph["node_dict"][relation["target_node_id"]]["node_name"]
            related_relations.append(relation)

        return related_concepts, related_relations
    
    def get_concept_id_by_name(self, concept_name):
        return self.graph_store.graph["node_name_to_id"].get(concept_name)
               
    def save_graph(self):
        self.graph_store.save_graph()

    def empty_graph(self):
        self.graph_store.delete_graph()
        self.emb_store.delete_index()
        self.emb_store = EmbStore(self.openai_api_key,
                                   self.pinecone_api_key,
                                   self.pinecone_index_name,
                                   self.emb_model,
                                   self.emb_dim)