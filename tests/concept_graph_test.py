import unittest
from unittest.mock import Mock, patch
import sys
import json

sys.path.append("..")
from concept_graph.concept_graph import concept_graph

PROJECT_ID = "powerful-surf-415220"
LOCATION = "us-west1"
CREDENTIALS_PATH = "/home/eddy/.config/gcloud/application_default_credentials.json"
BUCKET_PATH = "concept_store"
FILE_PREFIX = "test_world1/"

EMB_API_KEY = "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"
PINECONE_API_KEY = "72f6c3a8-29ec-4c52-9661-7c92a8cf1c64"
PINECONE_INDEX_NAME = "test-graph-index"
EMB_MODEL = "text-embedding-ada-002"
EMB_DIM = 1536

class TestConceptGraph(unittest.TestCase):
    
    def setUp(self):
        self.concept_graph = concept_graph(
            EMB_API_KEY,
            PINECONE_API_KEY,
            PINECONE_INDEX_NAME,
            EMB_MODEL,
            EMB_DIM,
            PROJECT_ID,
            BUCKET_PATH,
            FILE_PREFIX,
            CREDENTIALS_PATH
        )
    
    def test_add_concept(self):
        concept_name = "Test Concept"
        concept_type = "Test Type"
        concept_attributes = {"key": "value"}
        
        with patch.object(self.concept_graph.graph_store, 'add_node'), \
             patch.object(self.concept_graph.emb_store, 'insert_node_emb'):
            self.concept_graph.add_concept(concept_name, concept_type, concept_attributes)
    
    def test_delete_concept(self):
        concept_id = "test_concept_id"
        
        with patch.object(self.concept_graph.graph_store, 'delete_node'), \
             patch.object(self.concept_graph.emb_store, 'delete_node_emb'), \
             patch.object(self.concept_graph.file_store, 'delete_file'):
            self.concept_graph.graph_store.graph["node_dict"][concept_id] = {"image_path": "test.jpg"}
            self.concept_graph.delete_concept(concept_id)
    
    def test_update_concept(self):
        concept_id = "test_concept_id"
        concept_name = "Updated Concept"
        concept_type = "Updated Type"
        concept_attributes = {"updated_key": "updated_value"}
        
        with patch.object(self.concept_graph.graph_store, 'get_node') as mock_get_node, \
             patch.object(self.concept_graph.graph_store, 'update_node'), \
             patch.object(self.concept_graph.emb_store, 'update_node_emb'), \
             patch.object(self.concept_graph.file_store, 'add_file'):
            mock_get_node.return_value = {"node_attributes": {}}
            self.concept_graph.update_concept(concept_id, concept_name, concept_type, concept_attributes)
    
    def test_get_concept(self):
        concept_id = "test_concept_id"
        
        with patch.object(self.concept_graph.graph_store, 'get_node'):
            self.concept_graph.get_concept(concept_id)
    
    def test_add_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        relation_type = "Test Relation"
        relation_attributes = {"key": "value"}
        
        with patch.object(self.concept_graph.graph_store, 'add_edge'):
            self.concept_graph.add_relation(source_concept_id, target_concept_id, relation_type, relation_attributes)
    
    def test_delete_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        
        with patch.object(self.concept_graph.graph_store, 'delete_edge'):
            self.concept_graph.delete_relation(source_concept_id, target_concept_id)
    
    def test_update_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        relation_type = "Updated Relation"
        relation_attributes = {"updated_key": "updated_value"}
        
        with patch.object(self.concept_graph.graph_store, 'get_edge') as mock_get_edge, \
             patch.object(self.concept_graph.graph_store, 'update_edge'):
            mock_get_edge.return_value = {"edge_attributes": {}}
            self.concept_graph.update_relation(source_concept_id, target_concept_id, relation_type, relation_attributes)
    
    def test_get_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        
        with patch.object(self.concept_graph.graph_store, 'get_edge'):
            self.concept_graph.get_relation(source_concept_id, target_concept_id)
    
    def test_query_similar_concepts(self):
        query_text = "Test query"
        top_k = 5
        with patch.object(self.concept_graph.emb_store, 'query_similar_nodes') as mock_query_similar_nodes, \
                patch.object(self.concept_graph, 'get_concept') as mock_get_concept:
            mock_query_similar_nodes.return_value = [("concept_id_1", 0.9), ("concept_id_2", 0.8)]
            mock_get_concept.side_effect = lambda node_id: {
                "concept_id_1": {"node_id": "concept_id_1", "node_name": "Concept 1"},
                "concept_id_2": {"node_id": "concept_id_2", "node_name": "Concept 2"}
            }[node_id]
            similar_concepts = self.concept_graph.query_similar_concepts(query_text, top_k)
            self.assertEqual(len(similar_concepts), 2)
            self.assertEqual(similar_concepts[0][0]["node_id"], "concept_id_1")
            self.assertEqual(similar_concepts[0][0]["node_name"], "Concept 1")
            self.assertEqual(similar_concepts[1][0]["node_id"], "concept_id_2")
        self.assertEqual(similar_concepts[1][0]["node_name"], "Concept 2")
    
    def test_get_related_concepts(self):
        # Create test data
        self.concept_graph.add_concept("Test Concept", "Test Node Type", {"key": "value"})
        self.concept_graph.add_concept("Related Concept", "Test Node Type", {"key": "value"})
        self.concept_graph.add_concept("Unrelated Concept", "Unrelated Node Type", {"key": "value"})
        test_concept_id = self.concept_graph.get_concept_id_by_name("Test Concept")
        related_concept_id = self.concept_graph.get_concept_id_by_name("Related Concept")
        unrelated_concept_id = self.concept_graph.get_concept_id_by_name("Unrelated Concept")
        self.concept_graph.add_relation(test_concept_id, related_concept_id, "Test Relation")
        self.concept_graph.add_relation(test_concept_id, unrelated_concept_id, "Unrelated Relation")

        # Set up mock objects
        mock_node_dict = {
            test_concept_id: {"node_id": test_concept_id, "node_name": "Test Concept", "node_type": "Test Node Type"},
            related_concept_id: {"node_id": related_concept_id, "node_name": "Related Concept", "node_type": "Test Node Type"},
            unrelated_concept_id: {"node_id": unrelated_concept_id, "node_name": "Unrelated Concept", "node_type": "Unrelated Node Type"}
        }
        mock_get_neighbor_info = Mock(return_value=(
            [mock_node_dict[related_concept_id], mock_node_dict[unrelated_concept_id]],
            [
                {"source_node_id": test_concept_id, "target_node_id": related_concept_id, "edge_type": "Test Relation"},
                {"source_node_id": test_concept_id, "target_node_id": unrelated_concept_id, "edge_type": "Unrelated Relation"}
            ]
        ))

        # Call the function with mocked objects
        with patch.object(self.concept_graph.graph_store, 'get_neighbor_info', mock_get_neighbor_info):
            related_concepts, related_relations = self.concept_graph.get_related_concepts(
                test_concept_id, hop=1, relation_type="Test Relation", concept_type="Test Node Type"
            )

        # Assert the expected results
        self.assertEqual(len(related_concepts), 1)
        self.assertEqual(related_concepts[0]["node_type"], "Test Node Type")
        self.assertEqual(len(related_relations), 1)
        self.assertEqual(related_relations[0]["edge_type"], "Test Relation")
        self.assertEqual(related_relations[0]["source_concept"], "Test Concept")
        self.assertEqual(related_relations[0]["target_concept"], "Related Concept")

    
    def test_save_graph(self):
        with patch.object(self.concept_graph.graph_store, 'save_graph'):
            self.concept_graph.save_graph()

if __name__ == '__main__':
    unittest.main()