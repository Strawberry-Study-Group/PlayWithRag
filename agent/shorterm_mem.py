import sys
sys.path.append("..")
import json
from concept_graph.file_store import LocalFileStore
from collections import OrderedDict

class ShortTermMemory:
    """s
    Class representing the short-term memory of an agent.
    """
    
    def __init__(self, config, long_term_memory, save_file_config): 
        """
        Initialize the short-term memory.
        
        Args:
            max_context_length (int): The maximum number of tokens to store in the context window.
        """
        self.attributes = {}
        self.play_history = []
        self.max_history_rounds = config["max_history_rounds"]

        self.max_lru_cache_size = config.get("lru_cache_size", 40)
        self.lru_cache = OrderedDict()

        self.file_store = LocalFileStore(save_file_config["save_path"], "short_term_memory/")
        self.short_term_memory_file = "short_term_memory.json"
        self.load_short_term_memory()
        self.long_term_memory = long_term_memory

    def load_short_term_memory(self):
        try:
            self.file_store.get_file(self.short_term_memory_file, self.short_term_memory_file)
            with open(self.short_term_memory_file) as f:
                data = json.load(f, object_hook=lambda d: OrderedDict(d.items()))
                self.attributes = data["attributes"]
                self.play_history = data["play_history"]
                self.lru_cache = data["lru_cache"]
        except Exception:
            self.attributes = {}
            self.play_history = []
            self.lru_cache = OrderedDict()

    def save_short_term_memory(self):
        data = {
            "attributes": self.attributes,
            "play_history": self.play_history,
            "lru_cache": self.lru_cache
        }
        with open(self.short_term_memory_file, "w") as f:
            json.dump(data, f)
        self.file_store.add_file(self.short_term_memory_file, self.short_term_memory_file)

    def empty_short_term_memory(self):
        self.attributes = {}
        self.play_history = []
        self.lru_cache = OrderedDict()
        self.save_short_term_memory()

    def set_attribute(self, key, value):
        """
        Set a custom attribute in the short-term memory.
        
        Args:
            key (str): The attribute key.
            value (any): The attribute value.
        """
        self.attributes[key] = value
    
    def get_attribute(self, key):
        """
        Get the value of a custom attribute from the short-term memory.
        
        Args:
            key (str): The attribute key.
        
        Returns:
            any: The attribute value, or None if the attribute doesn't exist.
        """
        return self.attributes.get(key)
    
    def update_attribute(self, key, value):
        """
        Update the value of a custom attribute in the short-term memory.
        
        Args:
            key (str): The attribute key.
            value (any): The new attribute value.
        """
        if key in self.attributes:
            self.attributes[key] = value
    
    def remove_attribute(self, key):
        """
        Remove a custom attribute from the short-term memory.
        
        Args:
            key (str): The attribute key.
        """
        if key in self.attributes:
            del self.attributes[key]
    
    def append_play_history(self, text):
        """
        Append text to the context window.
        
        Args:
            text (str): The text to append.
        """
        self.play_history.append(text)
        if len(self.play_history) > self.max_history_rounds:
            self.play_history = self.play_history[1:]
    
    def get_play_history(self):
        """
        Get the current context window.
        
        Returns:
            str: The context window as a string.
        """
        return ' '.join(self.play_history)
    
    def get_attributes_text(self):
        """
        Get the custom attributes as a string.
        
        Returns:
            str: The custom attributes as a string.
        """
        return ', '.join([f"{key}: {value}" for key, value in self.attributes.items()])
    

    def get_item(self, item_id):
        """
        Get a concept from the LRU cache or load it from the file store.
        
        Args:
            concept_id (str): The ID of the concept to retrieve.
        
        Returns:
            dict: The concept dictionary.
        """
        if item_id in self.lru_cache:
            # Move the accessed concept to the end (most recently used)
            self.lru_cache.move_to_end(item_id)
            if self.lru_cache[item_id] == "concept":
                return self.long_term_memory.get_concept(item_id)
            if self.lru_cache[item_id] == "relation":
                return self.long_term_memory.get_relation(item_id[0], item_id[1])
        
        return None

    def mention_concept(self, concept_name):
        """
        Mention a concept, updating the LRU cache.
        
        Args:
            concept_id (str): The ID of the concept to mention.
        """
        concept_id = self.long_term_memory.get_concept_id_by_name(concept_name)
        if not concept_id:
            return

        if concept_id in self.lru_cache:
            # Move the accessed concept to the end (most recently used)
            self.lru_cache.move_to_end(concept_id)
        else:
            # Add the concept to the cache
            self.add_item_to_cache(concept_id, "concept")
    
    def mention_relation(self, source_concept_name, target_concept_named):
        """
        Mention a relation, updating the LRU cache.
        
        Args:
            source_concept_id (str): The ID of the source concept.
            target_concept_id (str): The ID of the target concept.
        """


        relation_id = self.long_term_memory.graph_store.parse_edge_key(source_concept_name, target_concept_named)
        if not self.long_term_memory.is_relation(relation_id):
            return 
        if relation_id in self.lru_cache:
            # Move the accessed relation to the end (most recently used)
            self.lru_cache.move_to_end(relation_id)
        else:
            # Add the relation to the cache
            self.add_item_to_cache(relation_id, "relation")

    def add_item_to_cache(self, item_id, item_type):
        """
        Add a concept to the LRU cache.
        
        Args:
            concept_id (str): The ID of the concept.
            concept (dict): The concept dictionary.
        """
        if item_type not in ["concept", "relation"]:
            raise ValueError("Invalid item type. Must be 'concept' or 'relation'.")

        if item_id in self.lru_cache:
            # Remove the existing entry if it exists
            del self.lru_cache[item_id]
        
        # Add the concept to the cache
        self.lru_cache[item_id] = item_type
        
        # Remove the least recently used concept if the cache size is exceeded
        if len(self.lru_cache) > self.max_lru_cache_size:
            self.lru_cache.popitem(last=False)

    def get_lru_text(self) -> str: 
        """
        Get the least recently used concepts as a string.
        
        Returns:
            str: The least recently used concepts as a string.
        """
        text = ""

        for key in self.lru_cache:
            if self.lru_cache[key] == "concept":
                concept = self.long_term_memory.get_concept(key)
                text += f"Concept: {concept}\n"
            if self.lru_cache[key] == "relation":
                relation = self.long_term_memory.get_relation(key[0], key[1])
                text += f"Relation: {relation}\n"

        return text