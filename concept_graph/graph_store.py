import json

class GraphStore:
    def __init__(self, file_store):
        self.file_store = file_store
        self.graph_file_name = "graph.json"
        self.graph = self.load_graph()

    def load_graph(self):
        try:
            self.file_store.get_file(self.graph_file_name, "graph.json")
            with open("graph.json", "r") as file:
                graph = json.load(file)
        except Exception:
            graph = {"node_dict": {}, "edge_dict": {}, "neighbor_dict": {}, "node_name_to_id": {}}
        return graph
    
    def delete_graph(self):
        try:
            self.file_store.delete_file("graph.json")
            self.graph = {"node_dict": {}, "edge_dict": {}, "neighbor_dict": {}, "node_name_to_id": {}}
        except Exception:
            print("Graph file is already empty.")

    def parse_edge_key(self, source_node_id, target_node_id):
        keys = [source_node_id, target_node_id]
        keys.sort()
        edge_key = keys[0] + "<->" + keys[1]
        return edge_key

    def save_graph(self):
        with open("graph.json", "w") as file:
            json.dump(self.graph, file)
        self.file_store.add_file("graph.json", self.graph_file_name)

    def is_valid_node(self, node):
        required_keys = ["node_id", "node_name", "node_type", "node_attributes", "is_editable"]
        return all(key in node for key in required_keys)

    def is_valid_edge(self, edge) -> bool:
        required_keys = ["source_node_id", "target_node_id", "edge_type", "is_editable"]
        return all(key in edge for key in required_keys)

    def add_node(self, node):
        if not self.is_valid_node(node):
            raise ValueError("Invalid node format")
        if node["node_id"] in self.graph["node_dict"]:
            raise ValueError("Node already exists")
        if node["node_name"] in self.graph["node_name_to_id"]:
            raise ValueError("Node name already exists")
        self.graph["node_dict"][node["node_id"]] = node
        self.graph["neighbor_dict"][node["node_id"]] = []
        self.graph["node_name_to_id"][node["node_name"]] = node["node_id"]

    def delete_node(self, node_id):
        if node_id in self.graph["node_dict"]:
            del self.graph["node_name_to_id"][self.graph["node_dict"][node_id]["node_name"]]
            del self.graph["node_dict"][node_id]
            del self.graph["neighbor_dict"][node_id]
            for edge_key in list(self.graph["edge_dict"].keys()):
                if node_id in edge_key:
                    del self.graph["edge_dict"][edge_key]
            for neighbor_list in self.graph["neighbor_dict"].values():
                if node_id in neighbor_list:
                    neighbor_list.remove(node_id)

    def update_node(self, node):
        if not self.is_valid_node(node):
            raise ValueError("Invalid node format")
        self.graph["node_dict"][node["node_id"]] = node

    def get_node(self, node_id):
        return self.graph["node_dict"].get(node_id)

    def add_edge(self, edge):
        if not self.is_valid_edge(edge):
            raise ValueError("Invalid edge format")
        edge_key = self.parse_edge_key(edge["source_node_id"], edge["target_node_id"])
        if edge_key in self.graph["edge_dict"]:
            raise ValueError("Edge already exists")
        self.graph["edge_dict"][edge_key] = edge
        self.graph["neighbor_dict"][edge["source_node_id"]].append(edge["target_node_id"])
        self.graph["neighbor_dict"][edge["target_node_id"]].append(edge["source_node_id"])

    def delete_edge(self, source_node_id, target_node_id):
        edge_key = self.parse_edge_key(source_node_id, target_node_id)
        if edge_key in self.graph["edge_dict"]:
            del self.graph["edge_dict"][edge_key]
            self.graph["neighbor_dict"][source_node_id].remove(target_node_id)
            self.graph["neighbor_dict"][target_node_id].remove(source_node_id)

    def update_edge(self, edge):
        if not self.is_valid_edge(edge):
            raise ValueError("Invalid edge format")
        edge_key = self.parse_edge_key(edge["source_node_id"], edge["target_node_id"])
        self.graph["edge_dict"][edge_key] = edge

    def get_edge(self, source_node_id, target_node_id):
        edge_key = self.parse_edge_key(source_node_id, target_node_id)
        return self.graph["edge_dict"].get(edge_key)

    def get_neighbor_info(self, node_id, hop=1):
        if hop == 1:
            neighbor_ids = self.graph["neighbor_dict"].get(node_id, [])
            node_list = [self.graph["node_dict"][neighbor_id] for neighbor_id in neighbor_ids]
            edge_keys  = [self.parse_edge_key(node_id, neighbor_id) for neighbor_id in neighbor_ids]
            edge_list = [self.graph["edge_dict"][edge_key] for edge_key in edge_keys]
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
                            if neighbor_id not in visited:
                                node_list.append(self.graph["node_dict"][neighbor_id])
                                edge_key = self.parse_edge_key(node, neighbor_id)
                                edge_list.append(self.graph["edge_dict"][edge_key])
                                new_queue.append(neighbor_id)
                queue = new_queue
            return node_list, edge_list