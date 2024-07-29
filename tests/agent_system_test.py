import sys
import json
sys.path.append("..")

from agent.agent import Agent

config_path = "../agent/config_local.json"

agent = Agent(config_path)
agent.shorterm_mem.set_attribute("player_name", "Dabolaw")
agent.shorterm_mem.set_attribute("player_age", 25)
agent.shorterm_mem.set_attribute("player_location", "The Arcane Tower")
agent.shorterm_mem.set_attribute("player_health", 100)
agent.shorterm_mem.set_attribute("player_energy", 100)
agent.shorterm_mem.set_attribute("player_gold", 0)
agent.shorterm_mem.set_attribute("player_inventory", [])
agent.shorterm_mem.append_play_history("Game started.")
print("Welcome to the AI-powered text-based game!")
print("Type 'quit' to exit the game.")

while True:
    player_input = input("> ")
    
    if player_input.lower() == 'quit':
        print("Thank you for playing. Goodbye!")
        break
    
    message_to_player, image_generation_prompt = agent.reason_one_round(player_input)
    print("=============================================")
    print(message_to_player)
    
    if image_generation_prompt:
        print("Image Generation Prompt:", image_generation_prompt)
    
    print()