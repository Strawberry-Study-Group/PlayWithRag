import os
import logging
import json

def check_game_save(save_path) -> bool:
    """
    Check if the game save is valid.
    read in game save path and check if it is valid
    will check if has graph folder, short term memory folder, and config.json
    """
    
    if not os.path.exists(save_path):
        raise FileNotFoundError("Save path does not exist")
    if not os.path.exists(os.path.join(save_path, "graph")):
        raise FileNotFoundError("graph folder does not exist")
    if not os.path.exists(os.path.join(save_path, "short_term_memory")):
        raise FileNotFoundError("short term memory folder does not exist")
    if not os.path.exists(os.path.join(save_path, "config.json")):
        raise FileNotFoundError("config.json does not exist")
    
    if not os.path.exists(os.path.join(save_path + "/graph", "graph.json")):
        raise FileNotFoundError("graph.json does not exist")
    
    