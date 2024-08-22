import unittest
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import faiss
sys.path.append("..")
from concept_graph.emb_store import EmbStoreLocal
from concept_graph.file_store import LocalFileStore

EMB_API_KEY = "sk-v9yxT3BlbkFJPJ56JGoZkNUUvyQsP0A"
EMB_MODEL = "text-embedding-ada-002"
EMB_DIM = 1536

class TestEmbStoreLocal(unittest.TestCase):
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        self.mock_file_store = LocalFileStore("concept_store", "test_world1/")
        self.mock_openai_client = mock_openai.return_value
        self.EmbStoreLocal = EmbStoreLocal(
            EMB_API_KEY,
            self.mock_file_store,
            EMB_MODEL,
            EMB_DIM
        )
        # Mock load_index to avoid file operations
        self.EmbStoreLocal.load_index = MagicMock()

    def test_generate_embedding(self):
        text = "Sample text"
        expected_embedding = np.random.rand(EMB_DIM).astype(np.float32)

        self.mock_openai_client.embeddings.create.return_value.data = [MagicMock(embedding=expected_embedding.tolist())]
        embedding = self.EmbStoreLocal.generate_embedding(text)

        self.assertEqual(embedding.shape, (EMB_DIM,))

    def test_insert_node_emb(self):
        node_id = "node_1"
        node_text = "Sample node text"
        namespace = "test_namespace"

        with patch.object(self.EmbStoreLocal, 'generate_embedding') as mock_generate_embedding:
            mock_generate_embedding.return_value = np.random.rand(EMB_DIM).astype(np.float32)
            self.EmbStoreLocal.insert_node_emb(node_id, node_text, namespace)

        self.assertIn(namespace, self.EmbStoreLocal.namespaces)
        self.assertIn(node_id, self.EmbStoreLocal.namespaces[namespace]["id_to_index"])

    def test_update_node_emb(self):
        node_id = "node_1"
        node_text = "Updated node text"
        namespace = "test_namespace"

        # First insert a node
        self.EmbStoreLocal.insert_node_emb(node_id, "Original text", namespace)

        with patch.object(self.EmbStoreLocal, 'generate_embedding') as mock_generate_embedding:
            mock_generate_embedding.return_value = np.random.rand(EMB_DIM).astype(np.float32)
            self.EmbStoreLocal.update_node_emb(node_id, node_text, namespace)

        self.assertIn(node_id, self.EmbStoreLocal.namespaces[namespace]["id_to_index"])

    def test_delete_node_emb(self):
        node_id = "node_1"
        namespace = "test_namespace"

        # First insert a node
        self.EmbStoreLocal.insert_node_emb(node_id, "Sample text", namespace)

        self.EmbStoreLocal.delete_node_emb(node_id, namespace)

        self.assertNotIn(node_id, self.EmbStoreLocal.namespaces[namespace]["id_to_index"])

    def test_get_node_emb(self):
        node_id = "node_1"
        namespace = "test_namespace"
        expected_embedding = np.random.rand(EMB_DIM).astype(np.float32)

        # Insert a node
        self.EmbStoreLocal.insert_node_emb(node_id, "Sample text", namespace)
        
        # Mock the Faiss index reconstruct method
        self.EmbStoreLocal.namespaces[namespace]["index"].reconstruct = MagicMock(return_value=expected_embedding)

        embedding = self.EmbStoreLocal.get_node_emb(node_id, namespace)

        np.testing.assert_array_almost_equal(embedding, expected_embedding)

    def test_query_similar_nodes(self):
        query_text = "Sample query text"
        top_k = 3
        namespace = "test_namespace"

        # Insert some nodes
        for i in range(5):
            self.EmbStoreLocal.insert_node_emb(f"node_{i}", f"Sample text {i}", namespace)

        with patch.object(self.EmbStoreLocal, 'generate_embedding') as mock_generate_embedding:
            mock_generate_embedding.return_value = np.random.rand(EMB_DIM).astype(np.float32)
            
            # Mock the Faiss index search method
            self.EmbStoreLocal.namespaces[namespace]["index"].search = MagicMock(return_value=(
                np.array([[0.9, 0.8, 0.7]]),
                np.array([[0, 1, 2]])
            ))
            
            similar_nodes = self.EmbStoreLocal.query_similar_nodes(query_text, top_k=top_k, namespace=namespace)

        self.assertEqual(len(similar_nodes), top_k)

    def test_delete_index(self):
        namespace = "test_namespace"

        # Insert a node to create the namespace
        self.EmbStoreLocal.insert_node_emb("node_1", "Sample text", namespace)

        self.EmbStoreLocal.delete_index(namespace)

        self.assertNotIn(namespace, self.EmbStoreLocal.namespaces)

if __name__ == '__main__':
    unittest.main()