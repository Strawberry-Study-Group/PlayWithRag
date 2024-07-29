import sys
sys.path.append("..")
from .shorterm_mem import ShortTermMemory
from concept_graph.concept_graph import ConceptGraph
from .action_node import *
from .llm import LLM
from render.render import Render
import json
import re

class Agent:
    """
    Class representing an AI agent.
    """
    
    def __init__(self, config_path):
        """
        Initialize the agent.
        
        Args:
            action_tree (ActionTree): The action tree for the agent.
            shorterm_mem (ShortTermMemory): The short-term memory of the agent.
            longterm_mem (object): The long-term memory of the agent (placeholder).
        """
        #read json config file
        with open(config_path) as f:
            self.config = json.load(f)

        self.actions = dict()

        self.longterm_mem = ConceptGraph(self.config["concept_graph_config"], self.config["save_file_config"])
        self.shorterm_mem = ShortTermMemory(self.config["shorterm_mem_config"], self.longterm_mem, self.config["save_file_config"])
        self.llm = LLM(config=self.config["llm_config"])
        with open(self.config["agent_config"]["retrieval_prompt_path"], encoding="utf-8") as f:
            retrieval_prompt = f.read()
        self.actions["long_term_mem"] = InformationRetrievalAction("Long-term memory retrieval", 
                                                                   self.longterm_mem,
                                                                   llm=self.llm,
                                                                   retrieval_prompt= retrieval_prompt)
        with open(self.config["agent_config"]["reasoning_prompt_path"], encoding="utf-8") as f:
            reasoning_prompt = f.read()
        self.actions["reasoning"] = BaseReasoning("Reasoning", llm=self.llm, 
                                                    reasoning_prompt=reasoning_prompt)
        self.actions["update_long_term_mem"] = LongTermMemoryUpdateAction("Update long-term memory", self.longterm_mem)
        self.actions["update_short_term_mem"] = ShortTermMemoryUpdateAction("Update short-term memory", self.shorterm_mem)
        self.actions["rendering"] = ImageRenderAction("Rendering", image_renderer=Render(self.config["render_config"]))
    

    def reason_one_round(self, player_input):
        
        retrieved_long_term_mem_info = self.actions["long_term_mem"].execute(player_input)
        print("retrieved_long_term_mem_info: ",retrieved_long_term_mem_info)
        for item_id,item_type in retrieved_long_term_mem_info:
            self.shorterm_mem.add_item_to_cache(item_id, item_type)
        short_term_context = self.shorterm_mem.get_lru_text()
        play_history = self.shorterm_mem.get_play_history()
        short_term_attributes = self.shorterm_mem.get_attributes_text()

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            llm_output = self.actions["reasoning"].execute(player_input, short_term_context, play_history, short_term_attributes)
            print("llm_output: ",llm_output)
            try:
                llm_output = json.loads(llm_output)
                print("JSON successfully parsed.")
                break
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                
                llm_output = llm_output.replace("'", '"')
                try:
                    llm_output = json.loads(llm_output)
                    print("JSON successfully parsed after fixing quotes.")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error after fixing quotes: {e}")
                    retry_count += 1
                    print(f"Retrying... (Attempt {retry_count}/{max_retries})")
                    
                    if retry_count == max_retries:
                        print("Max retries reached. Unable to parse JSON.")



        self.actions["update_long_term_mem"].execute(llm_output["concept_update_dict"], llm_output["relation_update_dict"])
        new_history = "Player input: " + player_input + "\n" + "Message to player: " + llm_output["message_to_player"] + "\n" + "Image generation prompt: " + llm_output["image_generation_prompt"] + "\n" 
        
        
        self.actions["update_short_term_mem"].execute(llm_output["short_term_mem_update_dict"],
                                                      llm_output["mentioned_relations_and_concepts"],
                                                      new_history)
        self.shorterm_mem.save_short_term_memory()
        self.longterm_mem.save_graph()
        rendered_image_path = self.actions["rendering"].execute(llm_output["image_generation_prompt"])

        return llm_output["message_to_player"] , rendered_image_path

