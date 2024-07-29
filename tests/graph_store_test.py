import unittest
import tempfile
import os
import sys
import json

sys.path.append("..")
from concept_graph.file_store import LocalFileStore
from concept_graph.graph_store import GraphStore


BASE_PATH = "concept_store"
FILE_PREFIX = "test_world1/"

class TestGraphStore(unittest.TestCase):
    def setUp(self):
        self.file_store = LocalFileStore(BASE_PATH, FILE_PREFIX)
        self.file_store.delete_prefix()
        self.graph_store = GraphStore(self.file_store)
        
        
    def test_add_node(self):
        node = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        self.graph_store.add_node(node)
        self.assertEqual(self.graph_store.graph["node_dict"]["node1"], node)
        self.assertIn("node1", self.graph_store.graph["neighbor_dict"])
        self.graph_store.save_graph()

    def test_delete_node(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "is_editable": True,
            "edge_type": "Edge Type",
            "edge_weight": 1
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_edge(edge)
        self.graph_store.delete_node("node1")
        self.assertNotIn("node1", self.graph_store.graph["node_dict"])
        self.assertNotIn("node1", self.graph_store.graph["neighbor_dict"])
        self.assertNotIn(("node1", "node2"), self.graph_store.graph["edge_dict"])
        self.assertNotIn("node1", self.graph_store.graph["neighbor_dict"]["node2"])
        self.graph_store.save_graph()

    def test_update_node(self):
        node = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        updated_node = {
            "node_id": "node1",
            "node_name": "Updated Node 1",
            "node_type": "Updated Type 1",
            "is_editable": True,
            "node_attributes": {"updated_key": "updated_value"}
        }
        self.graph_store.add_node(node)
        self.graph_store.update_node(updated_node)
        self.assertEqual(self.graph_store.graph["node_dict"]["node1"], updated_node)
        self.graph_store.save_graph()

    def test_get_node(self):
        node = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        self.graph_store.add_node(node)
        retrieved_node = self.graph_store.get_node("node1")
        self.assertEqual(retrieved_node, node)

    def test_add_edge(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_edge(edge)
        edge_key = self.graph_store.parse_edge_key("node1", "node2")
        self.assertEqual(self.graph_store.graph["edge_dict"][edge_key], edge)
        self.assertIn("node2", self.graph_store.graph["neighbor_dict"]["node1"])
        self.assertIn("node1", self.graph_store.graph["neighbor_dict"]["node2"])
        self.graph_store.save_graph()

    def test_delete_edge(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_edge(edge)
        self.graph_store.delete_edge("node1", "node2")
        self.assertNotIn(("node1", "node2"), self.graph_store.graph["edge_dict"])
        self.assertNotIn("node2", self.graph_store.graph["neighbor_dict"]["node1"])
        self.assertNotIn("node1", self.graph_store.graph["neighbor_dict"]["node2"])
        self.graph_store.save_graph()

    def test_update_edge(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        updated_edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Updated Edge Type",
            "is_editable": True,
            "edge_weight": 2
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_edge(edge)
        self.graph_store.update_edge(updated_edge)
        edge_key = self.graph_store.parse_edge_key("node1", "node2")
        self.assertEqual(self.graph_store.graph["edge_dict"][edge_key], updated_edge)
        self.graph_store.save_graph()

    def test_get_edge(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_edge(edge)
        retrieved_edge = self.graph_store.get_edge("node1", "node2")
        self.assertEqual(retrieved_edge, edge)

    def test_get_neighbor_info(self):
        node1 = {
            "node_id": "node1",
            "node_name": "Node 1",
            "node_type": "Type 1",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node2 = {
            "node_id": "node2",
            "node_name": "Node 2",
            "node_type": "Type 2",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        node3 = {
            "node_id": "node3",
            "node_name": "Node 3",
            "node_type": "Type 3",
            "is_editable": True,
            "node_attributes": {"key": "value"}
        }
        edge1 = {
            "source_node_id": "node1",
            "target_node_id": "node2",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        edge2 = {
            "source_node_id": "node2",
            "target_node_id": "node3",
            "edge_type": "Edge Type",
            "is_editable": True,
            "edge_weight": 1
        }
        self.graph_store.add_node(node1)
        self.graph_store.add_node(node2)
        self.graph_store.add_node(node3)
        self.graph_store.add_edge(edge1)
        self.graph_store.add_edge(edge2)
        neighbors_1hop, edges_1hop = self.graph_store.get_neighbor_info("node1", hop=1)
        neighbors_2hop, edges_2hop = self.graph_store.get_neighbor_info("node1", hop=2)
        
        # Check the neighbors and edges for 1-hop
        self.assertEqual(len(neighbors_1hop), 1)
        self.assertEqual(neighbors_1hop[0]["node_id"], "node2")
        self.assertEqual(len(edges_1hop), 1)
        self.assertEqual(edges_1hop[0]["source_node_id"], "node1")
        self.assertEqual(edges_1hop[0]["target_node_id"], "node2")
        
        # Check the neighbors and edges for 2-hop
        self.assertEqual(len(neighbors_2hop), 2)
        neighbor_ids_2hop = [node["node_id"] for node in neighbors_2hop]
        self.assertCountEqual(neighbor_ids_2hop, ["node2", "node3"])
        self.assertEqual(len(edges_2hop), 2)
        edge_source_target_pairs_2hop = [(edge["source_node_id"], edge["target_node_id"]) for edge in edges_2hop]
        self.assertCountEqual(edge_source_target_pairs_2hop, [("node1", "node2"), ("node2", "node3")])

if __name__ == "__main__":
    unittest.main()