import unittest
from unittest.mock import patch, MagicMock
from pinecone import Pinecone
import sys
sys.path.append("..")
from concept_graph.emb_store import emb_store



EMB_API_KEY = "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"
PINECONE_API_KEY = "72f6c3a8-29ec-4c52-9661-7c92a8cf1c64"
PINECONE_INDEX_NAME = "test-graph-index"
EMB_MODEL = "text-embedding-ada-002"
EMB_DIM = 1536

class TestEmbStore(unittest.TestCase):
    @patch('openai.Embedding.create')
    @patch('pinecone.Pinecone')
    def setUp(self, mock_pinecone, mock_openai_embedding):
        mock_pinecone.return_value.list_indexes.return_value.names = [PINECONE_INDEX_NAME]
        mock_pinecone.return_value.Index.return_value = mock_pinecone.return_value

        self.emb_store = emb_store(
            EMB_API_KEY,
            PINECONE_API_KEY,
            PINECONE_INDEX_NAME,
            EMB_MODEL,
            EMB_DIM
        )

    def test_generate_embedding(self):
        text = "Sample text"
        expected_embedding = [0.1] * EMB_DIM

        with patch('openai.Embedding.create') as mock_openai_embedding:
            mock_openai_embedding.return_value = {'data': [{'embedding': expected_embedding}]}
            embedding = self.emb_store.generate_embedding(text)

        self.assertEqual(embedding, expected_embedding)

    def test_insert_node_emb(self):
        node_id = "node_1"
        node_text = "Sample node text"

        with patch.object(self.emb_store.index, 'upsert') as mock_upsert:
            self.emb_store.insert_node_emb(node_id, node_text)

        mock_upsert.assert_called_once()

    def test_update_node_emb(self):
        node_id = "node_1"
        node_text = "Updated node text"

        with patch.object(self.emb_store.index, 'upsert') as mock_upsert:
            self.emb_store.update_node_emb(node_id, node_text)

        mock_upsert.assert_called_once()

    def test_delete_node_emb(self):
        node_id = "node_1"

        with patch.object(self.emb_store.index, 'delete') as mock_delete:
            self.emb_store.delete_node_emb(node_id)

        mock_delete.assert_called_once_with(ids=[node_id])

    def test_get_node_emb(self):
        node_id = "node_1"
        expected_embedding = [0.1] * EMB_DIM

        with patch.object(self.emb_store.index, 'fetch') as mock_fetch:
            mock_fetch.return_value.vectors = {node_id: type('obj', (object,), {'values': expected_embedding})}
            embedding = self.emb_store.get_node_emb(node_id)

        self.assertEqual(embedding, expected_embedding)

    def test_query_similar_nodes(self):
        query_text = "Sample query text"
        top_k = 3

        with patch.object(self.emb_store.index, 'query') as mock_query:
            mock_query.return_value.matches = [
                type('obj', (object,), {'id': "node_1", 'score': 0.9}),
                type('obj', (object,), {'id': "node_2", 'score': 0.8}),
                type('obj', (object,), {'id': "node_3", 'score': 0.7})
            ]
            similar_nodes = self.emb_store.query_similar_nodes(query_text, top_k=top_k)

        self.assertEqual(len(similar_nodes), top_k)

if __name__ == '__main__':
    unittest.main()