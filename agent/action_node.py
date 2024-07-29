from abc import ABC, abstractmethod
import json

class ActionNode(ABC):
    """
    Abstract base class for action nodes.
    """
    
    def __init__(self, action_description, child_actions=None):
        """
        Initialize the action node.
        
        Args:
            child_actions (list): List of child action nodes.
        """
        self.parent_action = None
        self.child_actions = child_actions or []
        self.action_description = action_description
    
    @abstractmethod
    def execute(self, information_input):
        """
        Execute the action node.
        
        Args:
            agent (Agent): The AI agent instance.
            game_state (dict): The current game state.
            
        Returns:
            dict: The updated game state after executing the action.
        """
        pass
    
    @abstractmethod
    def describe_action(self):
        """
        Return a description of the action node.
        
        Returns:
            str: Description of the action node.
        """
        pass

class InformationRetrievalAction(ActionNode):
    """
    Action node for retrieving information from the game state or agent's memory.
    """
    def __init__(self, action_description, concept_graph, llm, retrieval_prompt, 
                 match_threshold=0.5, concept_type=None, relation_type=None, related_hop=1):
        super().__init__(action_description)
        self.concept_graph = concept_graph
        self.retrieval_prompt = retrieval_prompt
        self.match_threshold = match_threshold
        self.llm = llm
        self.concept_type = concept_type
        self.relation_type = relation_type
        self.related_hop = related_hop
    
    def execute(self, input_text):

        input_prompt = self.retrieval_prompt
        input_prompt += "Player input: \n"
        input_prompt += input_text + "\n"
        input_prompt += "Concepts: \n"

        llm_output = self.llm.completion(input_prompt)
        concept_names = json.loads(llm_output)
        concept_names = concept_names["concepts"]
        print("concept_names: ", concept_names)
        matched_concepts = []
        for concept_name in concept_names:
            retrieved_concepts = self.concept_graph.query_similar_concepts(concept_name)
            for concept, score in retrieved_concepts:
                if score > self.match_threshold:
                    matched_concepts.append(concept)

        related_concepts = []
        related_relations = []
        for concept in matched_concepts:
            related_concepts, related_relations = self.concept_graph.get_related_concepts(concept["node_id"], 
                                                                                          hop=self.related_hop,
                                                                                          concept_type=self.concept_type,
                                                                                          relation_type=self.relation_type)

        retrieved_information_list = []
        for concept in related_concepts:
            retrieved_information_list.append((concept["node_id"], "concept"))
        for relation in related_relations:
            retrieved_information_list.append((relation["relation_id"], "relation"))
        
        return retrieved_information_list
    
    def describe_action(self):
        return self.action_description

class BaseReasoning(ActionNode):
    """
    Action node for performing reasoning based on the available information.
    """
    def __init__(self, action_description, reasoning_prompt, llm):  
        super().__init__(action_description) 
        self.reasoning_prompt = reasoning_prompt
        self.llm = llm

    def execute(self, information_input, 
                short_term_context, 
                play_history,
                player_and_world_state):    
        input_prompt = dict()

        input_prompt["backgound"] = short_term_context
        input_prompt["play_history"] = play_history
        input_prompt["player_and_world_state"] = player_and_world_state
        input_prompt["player_input"] = information_input
        input_prompt["instructions"] = self.reasoning_prompt

        prompt_text = ""
        for key in input_prompt:
            prompt_text += key.upper() + "\n" + input_prompt[key] + "\n"
            prompt_text += "\n"


        print("====================prompt_text======================================")
        print("backgound: ", short_term_context)
        print("play_history", play_history)
        print("player_and_world_state", player_and_world_state)
        print("=====================================================================")
        reasoning_output = self.llm.completion(prompt_text)

        return reasoning_output

    
    def describe_action(self):
        return self.action_description

class LongTermMemoryUpdateAction(ActionNode):
    """
    Action node for updating the agent's memory based on the current game state.
    """
    def __init__(self, action_description, concept_graph):
        super().__init__(action_description)
        self.concept_graph = concept_graph
    
    def execute(self, concept_update_dict, relation_update_dict):
        for concept_name in concept_update_dict:
            if concept_update_dict[concept_name]["manipulation"] == "add":
                if not self.concept_graph.get_concept_id_by_name(concept_name):
                    self.concept_graph.add_concept(concept_name,
                                                concept_type=concept_update_dict[concept_name]["concept_type"],
                                                concept_attributes=concept_update_dict[concept_name]["concept_attributes"])
            if concept_update_dict[concept_name]["manipulation"] == "delete":
                concept_id = self.concept_graph.get_concept_id_by_name(concept_name)
                self.concept_graph.delete_concept(concept_id)
            if concept_update_dict[concept_name]["manipulation"] == "update":
                concept_id = self.concept_graph.get_concept_id_by_name(concept_name)
                if concept_id:
                    self.concept_graph.update_concept(concept_id,
                                                    concept_type=concept_update_dict[concept_name]["concept_type"],
                                                    concept_attributes=concept_update_dict[concept_name]["concept_attributes"])
            
        for relation in relation_update_dict:
            source_concept_id = self.concept_graph.get_concept_id_by_name(relation["source_concept_name"])
            target_concept_id = self.concept_graph.get_concept_id_by_name(relation["target_concept_name"]) 
            if relation["manipulation"] == "add":
                if source_concept_id and target_concept_id:
                    relation_id = self.concept_graph.graph_store.parse_edge_key(source_concept_id, target_concept_id)  
                    if not self.concept_graph.graph_store.graph["edge_dict"].get(relation_id):
                        self.concept_graph.add_relation(source_concept_id,
                                                        target_concept_id,
                                                        relation["relation_type"],
                                                        relation["is_editable"],)
            if relation["manipulation"] == "delete":
                self.concept_graph.delete_relation(source_concept_id, target_concept_id)
            if relation["manipulation"] == "update":
                self.concept_graph.update_relation(source_concept_id,
                                                target_concept_id,
                                                relation["relation_type"],
                                                relation["is_editable"],)
    
    def describe_action(self):
        return self.action_description
    
class ShortTermMemoryUpdateAction(ActionNode):
    """
    Action node for updating the agent's short-term memory based on the current game state.
    """
    def __init__(self, action_description, short_term_memory):
        super().__init__(action_description)
        self.short_term_memory = short_term_memory
        self.long_term_memory = short_term_memory.long_term_memory
    
    def execute(self, attribute_update_dict, lru_cache_update_list, history_text):
        for attribute in attribute_update_dict:
            self.short_term_memory.update_attribute(attribute, attribute_update_dict[attribute])


        for item in lru_cache_update_list:
            if isinstance(item,list) and len(item) == 2:
                self.short_term_memory.mention_relation(item[0], item[1])
            else:
                self.short_term_memory.mention_concept(item)

        
        self.short_term_memory.append_play_history(history_text)
    
    def describe_action(self):
        return self.action_description

class ImageRenderAction(ActionNode):
    """
    Action node for rendering an image based on the current game state.
    """
    def __init__(self, action_description, image_renderer):
        super().__init__(action_description)
        self.image_renderer = image_renderer
    
    def execute(self, image_generation_prompt):
        image_path = self.image_renderer.generate(image_generation_prompt)
        return image_path
    
    def describe_action(self):
        return self.action_description
# Add more action node classes as needed