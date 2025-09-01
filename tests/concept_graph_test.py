import unittest
from unittest.mock import Mock, patch
import sys
import json

sys.path.append("..")
from concept_graph.concept_graph import ConceptGraph

file_store_config = {
    "provider": "local",
    "save_path": "unit_test_save/",
    "file_prefix": "concept_graph/"
}

concept_graph_config = {
    "provider": "local",  # Use local FAISS instead of Pinecone for testing
    "embedding_api_key": "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",
    "openai_api_key": "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",
    "pinecone_api_key": "72f6c3a8-29ec-4c52-9661-7c92a8cf1c64",
    "pinecone_index_name": "test-graph-index",
    "emb_model": "text-embedding-3-small",
    "emb_dim": 1536
}

class TestConceptGraph(unittest.TestCase):
    
    def setUp(self):
        self.ConceptGraph = ConceptGraph(concept_graph_config, file_store_config)
    
    def test_add_concept(self):
        concept_name = "Test Concept"
        concept_type = "Test Type"
        concept_attributes = {"key": "value"}
        
        with patch.object(self.ConceptGraph.graph_store, 'add_node'), \
             patch.object(self.ConceptGraph.emb_store, 'insert_node_emb'):
            self.ConceptGraph.add_concept(concept_name, concept_type, concept_attributes)
    
    def test_delete_concept(self):
        concept_id = "test_concept_id"
        
        with patch.object(self.ConceptGraph.graph_store, 'delete_node'), \
             patch.object(self.ConceptGraph.emb_store, 'delete_node_emb'), \
             patch.object(self.ConceptGraph.file_store, 'delete_file'):
            self.ConceptGraph.graph_store.graph["node_dict"][concept_id] = {"image_path": "test.jpg"}
            self.ConceptGraph.delete_concept(concept_id)
    
    def test_update_concept(self):
        concept_id = "test_concept_id"
        concept_name = "Updated Concept"
        concept_type = "Updated Type"
        concept_attributes = {"updated_key": "updated_value"}
        
        with patch.object(self.ConceptGraph.graph_store, 'get_node') as mock_get_node, \
             patch.object(self.ConceptGraph.graph_store, 'update_node'), \
             patch.object(self.ConceptGraph.emb_store, 'update_node_emb'), \
             patch.object(self.ConceptGraph.file_store, 'add_file'):
            mock_get_node.return_value = {"node_attributes": {}}
            self.ConceptGraph.update_concept(concept_id, concept_name, concept_type, concept_attributes)
    
    def test_get_concept(self):
        concept_id = "test_concept_id"
        
        with patch.object(self.ConceptGraph.graph_store, 'get_node'):
            self.ConceptGraph.get_concept(concept_id)
    
    def test_add_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        relation_type = "Test Relation"
        is_editable = True
        
        with patch.object(self.ConceptGraph.graph_store, 'add_edge'):
            self.ConceptGraph.add_relation(source_concept_id, target_concept_id, relation_type, is_editable=is_editable)
    
    def test_delete_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        
        with patch.object(self.ConceptGraph.graph_store, 'delete_edge'):
            self.ConceptGraph.delete_relation(source_concept_id, target_concept_id)
    
    def test_update_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        relation_type = "Updated Relation"
        is_editable = True
        
        with patch.object(self.ConceptGraph.graph_store, 'get_edge') as mock_get_edge, \
             patch.object(self.ConceptGraph.graph_store, 'update_edge'):
            mock_get_edge.return_value = {"edge_attributes": {}}
            self.ConceptGraph.update_relation(source_concept_id, target_concept_id, relation_type, is_editable)
    
    def test_get_relation(self):
        source_concept_id = "test_source_concept_id"
        target_concept_id = "test_target_concept_id"
        
        with patch.object(self.ConceptGraph.graph_store, 'get_edge'):
            self.ConceptGraph.get_relation(source_concept_id, target_concept_id)
    
    def test_query_similar_concepts(self):
        query_text = "Test query"
        top_k = 5
        with patch.object(self.ConceptGraph.emb_store, 'query_similar_nodes') as mock_query_similar_nodes, \
                patch.object(self.ConceptGraph, 'get_concept') as mock_get_concept:
            mock_query_similar_nodes.return_value = [("concept_id_1", 0.9), ("concept_id_2", 0.8)]
            mock_get_concept.side_effect = lambda node_id: {
                "concept_id_1": {"node_id": "concept_id_1", "node_name": "Concept 1"},
                "concept_id_2": {"node_id": "concept_id_2", "node_name": "Concept 2"}
            }[node_id]
            similar_concepts = self.ConceptGraph.query_similar_concepts(query_text, top_k)
            self.assertEqual(len(similar_concepts), 2)
            self.assertEqual(similar_concepts[0][0]["node_id"], "concept_id_1")
            self.assertEqual(similar_concepts[0][0]["node_name"], "Concept 1")
            self.assertEqual(similar_concepts[1][0]["node_id"], "concept_id_2")
        self.assertEqual(similar_concepts[1][0]["node_name"], "Concept 2")
    
    def test_get_related_concepts(self):
        # Create test data
        self.ConceptGraph.add_concept("Test Concept", "Test Node Type", {"key": "value"})
        self.ConceptGraph.add_concept("Related Concept", "Test Node Type", {"key": "value"})
        self.ConceptGraph.add_concept("Unrelated Concept", "Unrelated Node Type", {"key": "value"})
        test_concept_id = self.ConceptGraph.get_concept_id_by_name("Test Concept")
        related_concept_id = self.ConceptGraph.get_concept_id_by_name("Related Concept")
        unrelated_concept_id = self.ConceptGraph.get_concept_id_by_name("Unrelated Concept")
        self.ConceptGraph.add_relation(test_concept_id, related_concept_id, "Test Relation")
        self.ConceptGraph.add_relation(test_concept_id, unrelated_concept_id, "Unrelated Relation")

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
        with patch.object(self.ConceptGraph.graph_store, 'get_neighbor_info', mock_get_neighbor_info):
            related_concepts, related_relations = self.ConceptGraph.get_related_concepts(
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
        with patch.object(self.ConceptGraph.graph_store, 'save_graph'):
            self.ConceptGraph.save_graph()

if __name__ == '__main__':
    unittest.main()