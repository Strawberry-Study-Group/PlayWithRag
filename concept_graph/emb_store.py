
import openai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import re

class EmbStore:
    def __init__(self, emb_api_key,
                 pinecone_api_key,
                 pinecone_index_name,
                 emb_model,
                 emb_dim,
                 metric='cosine',
                 cloud='aws',
                 region='us-west-2'):
        if pinecone_index_name is None:
            raise ValueError("Pinecone index name is required")
        if not re.match(r'^[a-z0-9-]+$', pinecone_index_name):
            raise ValueError("Index name must consist of lowercase alphanumeric characters or '-'")
        openai.api_key = emb_api_key
        pc = Pinecone(api_key = pinecone_api_key)
        self.emb_model = emb_model
        self.emb_dim = emb_dim
        if pinecone_index_name not in pc.list_indexes().names():
            pc.create_index(
                name=pinecone_index_name,
                dimension=emb_dim,
                metric= metric,
                spec=ServerlessSpec(
                    cloud= cloud,
                    region= region
                )
            )
        self.client = pc
        self.index_name = pinecone_index_name
        self.index = pc.Index(pinecone_index_name)
        self.openai = OpenAI(api_key = emb_api_key)
    def generate_embedding(self, text):
        response = self.openai.embeddings.create(
            input=text,
            model=self.emb_model
        )
        embedding = response.data[0].embedding
        return embedding
    

    def insert_node_emb(self, node_id, node_text, namespace=None):
        embedding = self.generate_embedding(node_text)
        if namespace is not None:
            self.index.upsert(vectors=[{"id": node_id, "values": embedding}], namespace=namespace)
        else:
            self.index.upsert(vectors=[{"id": node_id, "values": embedding}])

    def update_node_emb(self, node_id, node_text, namespace=None):
        embedding = self.generate_embedding(node_text)
        if namespace is not None:
            self.index.upsert(vectors=[{"id": node_id, "values": embedding}], namespace=namespace)
        else:
            self.index.upsert(vectors=[{"id": node_id, "values": embedding}])

    def delete_node_emb(self, node_id, namespace=None):
        if namespace is not None:
            self.index.delete(ids=[node_id], namespace=namespace)
        else:
            self.index.delete(ids=[node_id])

    def get_node_emb(self, node_id):
        response = self.index.fetch(ids=[node_id])
        if response.vectors.get(node_id) is not None:
            return response.vectors[node_id].values
        else:
            return None

    def query_similar_nodes(self, query_text, top_k=5, namespace=None):
        query_embedding = self.generate_embedding(query_text)
        if namespace is not None:
            response = self.index.query(vector=query_embedding, top_k=top_k, include_values=True, namespace=namespace)
        else:
            response = self.index.query(vector=query_embedding, top_k=top_k, include_values=True)
        similar_nodes = [(match.id, match.score) for match in response.matches]
        return similar_nodes
    
    def delete_index(self):
        self.client.delete_index(self.index_name)