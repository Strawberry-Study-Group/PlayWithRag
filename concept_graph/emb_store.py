
import openai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import re
import json
import numpy as np
import faiss

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


class EmbStoreLocal:
    def __init__(self, emb_api_key, file_store, emb_model, emb_dim, index_file_name="emb_index.json"):
        self.openai = OpenAI(api_key=emb_api_key)
        self.file_store = file_store
        self.emb_model = emb_model
        self.emb_dim = emb_dim
        self.index_file_name = index_file_name
        self.namespaces = {}
        self.load_index()
    
    def generate_embedding(self, text):
        response = self.openai.embeddings.create(
            input=text,
            model=self.emb_model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

    def load_index(self):
        try:
            with open(self.file_store.base_path + self.file_store.file_prefix + self.index_file_name, "r") as f:
                data = json.load(f)
            for namespace, namespace_data in data.items():
                self.namespaces[namespace] = {
                    "index": faiss.IndexFlatIP(self.emb_dim),
                    "id_to_index": namespace_data["id_to_index"]
                }
                vectors = np.array(namespace_data["vectors"], dtype=np.float32)
                if vectors.shape[1] != self.emb_dim:
                    raise ValueError(f"Dimension mismatch in the loaded index for namespace {namespace}.")
                self.namespaces[namespace]["index"].add(vectors)
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty default namespace
            self.namespaces["default"] = {
                "index": faiss.IndexFlatIP(self.emb_dim),
                "id_to_index": {}
            }

    def save_index(self):
        data = {}
        for namespace, namespace_data in self.namespaces.items():
            vectors = namespace_data["index"].reconstruct_n(0, namespace_data["index"].ntotal)
            data[namespace] = {
                "id_to_index": namespace_data["id_to_index"],
                "vectors": vectors.tolist()
            }
        with open("temp_index.json", "w") as f:
            json.dump(data, f)
        self.file_store.add_file("temp_index.json", self.index_file_name)

    def get_namespace(self, namespace=None):
        if namespace is None:
            namespace = "default"
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {
                "index": faiss.IndexFlatIP(self.emb_dim),
                "id_to_index": {}
            }
        return self.namespaces[namespace]

    def insert_node_emb(self, node_id, node_text, namespace=None):
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {
                "index": faiss.IndexFlatIP(self.emb_dim),
                "id_to_index": {}
            }
        namespace_data = self.get_namespace(namespace)
        embedding = self.generate_embedding(node_text)
        namespace_data["index"].add(np.array([embedding], dtype=np.float32))
        namespace_data["id_to_index"][node_id] = namespace_data["index"].ntotal - 1
        self.save_index()

    def update_node_emb(self, node_id, node_text, namespace=None):
        if namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist.")
        namespace_data = self.get_namespace(namespace)
        if node_id in namespace_data["id_to_index"]:
            self.delete_node_emb(node_id, namespace)
        self.insert_node_emb(node_id, node_text, namespace)

    def delete_node_emb(self, node_id, namespace=None):
        if namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist.")
        namespace_data = self.get_namespace(namespace)
        if node_id in namespace_data["id_to_index"]:
            index_to_remove = namespace_data["id_to_index"][node_id]
            vectors = namespace_data["index"].reconstruct_n(0, namespace_data["index"].ntotal)
            vectors = np.delete(vectors, index_to_remove, axis=0)
            namespace_data["index"] = faiss.IndexFlatIP(self.emb_dim)
            namespace_data["index"].add(vectors)
            del namespace_data["id_to_index"][node_id]
            # Update the id_to_index mapping
            namespace_data["id_to_index"] = {id: (idx if idx < index_to_remove else idx - 1) 
                                             for id, idx in namespace_data["id_to_index"].items()}
            self.save_index()

    def get_node_emb(self, node_id, namespace=None):
        if  namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist.")
        namespace_data = self.get_namespace(namespace)
        if node_id in namespace_data["id_to_index"]:
            index = namespace_data["id_to_index"][node_id]
            return namespace_data["index"].reconstruct(index)
        return None

    def query_similar_nodes(self, query_text, top_k=5, namespace=None):
        namespace_data = self.get_namespace(namespace)
        
        query_embedding = self.generate_embedding(query_text)
        
        _, indices = namespace_data["index"].search(np.array([query_embedding], dtype=np.float32), top_k)
        
        similar_nodes = []
        for idx in indices[0]:
            try:
                node_id = next(key for key, value in namespace_data["id_to_index"].items() if value == idx)
                vector = namespace_data["index"].reconstruct(int(idx))
                score = np.dot(query_embedding, vector)
                similar_nodes.append((node_id, float(score)))
            except StopIteration:
                print(f"Warning: No node found for index {idx} in namespace {namespace}. Skipping.")
            except Exception as e:
                print(f"Error processing index {idx}: {str(e)}")
        
        return similar_nodes

    def delete_index(self, namespace=None):
        if namespace is None:
            self.namespaces = {"default": {"index": faiss.IndexFlatIP(self.emb_dim), "id_to_index": {}}}
        elif namespace in self.namespaces:
            del self.namespaces[namespace]
        elif namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist.")
        self.save_index()
        self.file_store.delete_file(self.index_file_name)