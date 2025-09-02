"""System tests for complete game world scenario based on the notebook example."""

import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_graph.concept_graph import ConceptGraphFactory, ConceptGraphService
from .config import get_test_config, check_test_readiness


class TestGameWorldScenario:
    """Test complete game world creation and interaction scenarios."""
    
    @pytest.fixture
    def game_world(self):
        """Create a populated game world for testing."""
        if not check_test_readiness(use_remote=False):
            pytest.skip("API keys not configured for testing")
        
        config = get_test_config(use_remote=False)
        graph = ConceptGraphFactory.create_from_config(
            config["concept_graph_config"], 
            config["file_store_config"],
            world_name="test_game_world"
        )
        graph.empty_graph()
        
        # Create the game world based on the notebook scenario
        self._populate_game_world(graph)
        
        yield graph
        
        # Cleanup
        try:
            graph.empty_graph()
        except Exception:
            pass
    
    def _populate_game_world(self, graph: ConceptGraphService) -> Dict[str, str]:
        """Populate the game world with NPCs, events, and locations."""
        concept_ids = {}
        
        # Create NPCs
        npcs = [
            ("kobuko", {"gender": "male", "age": "young", "npc_description": "kobuko is very tanky npc, it can heal himself"}),
            ("Eldrin", {"gender": "male", "age": "old", "npc_description": "Eldrin is a wise and powerful wizard who guides the player through their quest."}),
            ("Lyra", {"gender": "female", "age": "young", "npc_description": "Lyra is a skilled archer and a loyal companion to the player."}),
            ("Thorgrim", {"gender": "male", "age": "middle-aged", "npc_description": "Thorgrim is a mighty dwarven warrior who joins the player's party."}),
            ("Elara", {"gender": "female", "age": "young", "npc_description": "Elara is a mystical elf who possesses the ability to communicate with nature."}),
            ("Rognark", {"gender": "male", "age": "old", "npc_description": "Rognark is an ancient dragon who has taken human form."}),
            ("Aria", {"gender": "female", "age": "young", "npc_description": "Aria is a skilled bard who travels with the player."}),
            ("Zephyr", {"gender": "male", "age": "young", "npc_description": "Zephyr is a mischievous wind spirit who enjoys playing tricks."}),
            ("Naia", {"gender": "female", "age": "middle-aged", "npc_description": "Naia is a wise and caring healer who runs a sanctuary."}),
            ("Grom", {"gender": "male", "age": "old", "npc_description": "Grom is a retired gladiator who now serves as a mentor."}),
            ("Lumi", {"gender": "female", "age": "young", "npc_description": "Lumi is a mysterious figure who appears in dreams."})
        ]
        
        for name, attributes in npcs:
            concept_ids[name] = graph.add_concept(name, "npc", attributes)
        
        # Create events
        events = [
            ("The Betrayal of Eldrin", {"event_description": "Eldrin reveals his true intentions and betrays the party.", "location": "The Arcane Tower"}),
            ("The Forging of Destiny", {"event_description": "Thorgrim forges a legendary weapon known as the Sword of Destiny.", "location": "The Dwarven Forge"}),
            ("The Ritual of the Ancients", {"event_description": "Elara discovers an ancient ritual that can summon forest spirits.", "location": "The Whispering Woods"}),
            ("The Lost City of Gold", {"event_description": "Rognark reveals the existence of a lost city filled with treasures.", "location": "The Desert of Mirages"}),
            ("The Siren's Song", {"event_description": "Aria's music attracts a powerful siren who seeks to lure the party.", "location": "The Enchanted Cove"}),
            ("The Trickster's Gambit", {"event_description": "Zephyr leads the party into a trap set by a cunning trickster.", "location": "The Labyrinth of Illusions"}),
            ("The Plague of Shadows", {"event_description": "A mysterious plague sweeps through the land, turning people into shadows.", "location": "The Forsaken Village"}),
            ("The Tournament of Champions", {"event_description": "Grom invites the party to participate in a grand tournament.", "location": "The Coliseum of Heroes"}),
            ("The Dreamscape of Prophecy", {"event_description": "Lumi appears in dreams, revealing a cryptic prophecy.", "location": "The Ethereal Plane"})
        ]
        
        for name, attributes in events:
            concept_ids[name] = graph.add_concept(name, "event", attributes)
        
        # Create locations
        locations = [
            ("The Arcane Tower", {"location_description": "A tall, mysterious tower that pulses with arcane energy."}),
            ("The Dwarven Forge", {"location_description": "A massive underground forge where dwarves craft legendary weapons."}),
            ("The Whispering Woods", {"location_description": "An enchanted forest filled with ancient trees that whisper secrets."}),
            ("The Desert of Mirages", {"location_description": "A vast desert known for its illusions and hidden treasures."}),
            ("The Enchanted Cove", {"location_description": "A secluded cove with crystal-clear waters and a mesmerizing aura."}),
            ("The Labyrinth of Illusions", {"location_description": "A maze-like dungeon filled with deceptive traps."}),
            ("The Forsaken Village", {"location_description": "A once-thriving village now plagued by shadows."}),
            ("The Coliseum of Heroes", {"location_description": "A grand arena where champions compete for glory."}),
            ("The Ethereal Plane", {"location_description": "A mystical realm accessible only through dreams."})
        ]
        
        for name, attributes in locations:
            concept_ids[name] = graph.add_concept(name, "location", attributes)
        
        # Add NPC relations
        npc_relations = [
            ("Eldrin", "kobuko", "mentors"),
            ("Lyra", "kobuko", "companions_with"),
            ("Thorgrim", "kobuko", "allies_with"),
            ("Elara", "kobuko", "guides"),
            ("Rognark", "kobuko", "quests_for"),
            ("Aria", "kobuko", "travels_with"),
            ("Zephyr", "kobuko", "informs"),
            ("Naia", "kobuko", "heals"),
            ("Grom", "kobuko", "trains"),
            ("Lumi", "kobuko", "prophesies_to"),
            ("Lyra", "Eldrin", "learns_from"),
            ("Thorgrim", "Eldrin", "befriends"),
            ("Elara", "Lyra", "sisterhood"),
            ("Aria", "Lyra", "sisterhood"),
            ("Naia", "Elara", "mentors"),
            ("Grom", "Thorgrim", "rivals_with"),
            ("Rognark", "Grom", "employs"),
            ("Zephyr", "Aria", "admires")
        ]
        
        for source, target, relation_type in npc_relations:
            graph.add_relation(concept_ids[source], concept_ids[target], relation_type)
        
        # Add event-NPC relations
        event_npc_relations = [
            ("The Betrayal of Eldrin", "kobuko", "person-event"),
            ("The Betrayal of Eldrin", "Eldrin", "person-event"),
            ("The Forging of Destiny", "kobuko", "person-event"),
            ("The Forging of Destiny", "Thorgrim", "person-event"),
            ("The Ritual of the Ancients", "kobuko", "person-event"),
            ("The Ritual of the Ancients", "Elara", "person-event"),
            ("The Lost City of Gold", "kobuko", "person-event"),
            ("The Lost City of Gold", "Rognark", "person-event"),
            ("The Siren's Song", "kobuko", "person-event"),
            ("The Siren's Song", "Aria", "person-event"),
            ("The Trickster's Gambit", "kobuko", "person-event"),
            ("The Trickster's Gambit", "Zephyr", "person-event"),
            ("The Plague of Shadows", "kobuko", "person-event"),
            ("The Plague of Shadows", "Naia", "person-event"),
            ("The Tournament of Champions", "kobuko", "person-event"),
            ("The Tournament of Champions", "Grom", "person-event"),
            ("The Dreamscape of Prophecy", "kobuko", "person-event"),
            ("The Dreamscape of Prophecy", "Lumi", "person-event")
        ]
        
        for event, npc, relation_type in event_npc_relations:
            graph.add_relation(concept_ids[event], concept_ids[npc], relation_type)
        
        # Add event-location relations
        event_location_relations = [
            ("The Betrayal of Eldrin", "The Arcane Tower", "event-location"),
            ("The Forging of Destiny", "The Dwarven Forge", "event-location"),
            ("The Ritual of the Ancients", "The Whispering Woods", "event-location"),
            ("The Lost City of Gold", "The Desert of Mirages", "event-location"),
            ("The Siren's Song", "The Enchanted Cove", "event-location"),
            ("The Trickster's Gambit", "The Labyrinth of Illusions", "event-location"),
            ("The Plague of Shadows", "The Forsaken Village", "event-location"),
            ("The Tournament of Champions", "The Coliseum of Heroes", "event-location"),
            ("The Dreamscape of Prophecy", "The Ethereal Plane", "event-location")
        ]
        
        for event, location, relation_type in event_location_relations:
            graph.add_relation(concept_ids[event], concept_ids[location], relation_type)
        
        graph.save_graph()
        return concept_ids
    
    def test_character_similarity_search(self, game_world):
        """Test similarity search for character concepts."""
        # Search for warrior-type characters
        warrior_results = game_world.query_similar_concepts("mighty warrior", top_k=5)
        
        assert len(warrior_results) > 0
        warrior_names = [result[0]["node_name"] for result in warrior_results]
        
        # Should find Thorgrim (dwarven warrior) and Grom (gladiator)
        assert "Thorgrim" in warrior_names or "Grom" in warrior_names
        
        # Search for magic users
        magic_results = game_world.query_similar_concepts("wizard magic user", top_k=5)
        
        assert len(magic_results) > 0
        magic_names = [result[0]["node_name"] for result in magic_results]
        
        # Should find Eldrin (wizard) and Elara (mystical elf)
        assert "Eldrin" in magic_names or "Elara" in magic_names
        
        # Search for healers
        healer_results = game_world.query_similar_concepts("healer healing sanctuary", top_k=5)
        
        assert len(healer_results) > 0
        healer_names = [result[0]["node_name"] for result in healer_results]
        
        # Should find Naia (healer) and possibly kobuko (can heal himself)
        assert "Naia" in healer_names or "kobuko" in healer_names
    
    def test_event_similarity_search(self, game_world):
        """Test similarity search for event concepts."""
        # Search for betrayal/conflict events
        betrayal_results = game_world.query_similar_concepts("betrayal conflict treachery", top_k=5)
        
        assert len(betrayal_results) > 0
        betrayal_names = [result[0]["node_name"] for result in betrayal_results]
        
        # Should find "The Betrayal of Eldrin"
        assert "The Betrayal of Eldrin" in betrayal_names
        
        # Search for magical/mystical events
        mystical_results = game_world.query_similar_concepts("ritual magic prophecy mystical", top_k=5)
        
        assert len(mystical_results) > 0
        mystical_names = [result[0]["node_name"] for result in mystical_results]
        
        # Should find ritual and prophecy events
        assert any(name in ["The Ritual of the Ancients", "The Dreamscape of Prophecy"] 
                  for name in mystical_names)
        
        # Search for combat/competition events
        combat_results = game_world.query_similar_concepts("tournament competition battle", top_k=5)
        
        assert len(combat_results) > 0
        combat_names = [result[0]["node_name"] for result in combat_results]
        
        # Should find "The Tournament of Champions"
        assert "The Tournament of Champions" in combat_names
    
    def test_location_similarity_search(self, game_world):
        """Test similarity search for location concepts."""
        # Search for mystical/magical places
        mystical_results = game_world.query_similar_concepts("mystical magical enchanted", top_k=5)
        
        assert len(mystical_results) > 0
        mystical_names = [result[0]["node_name"] for result in mystical_results]
        
        # Should find mystical locations
        assert any(name in ["The Arcane Tower", "The Whispering Woods", "The Enchanted Cove", "The Ethereal Plane"] 
                  for name in mystical_names)
        
        # Search for dangerous/challenging places
        dangerous_results = game_world.query_similar_concepts("dangerous labyrinth desert plague", top_k=5)
        
        assert len(dangerous_results) > 0
        dangerous_names = [result[0]["node_name"] for result in dangerous_results]
        
        # Should find challenging locations
        assert any(name in ["The Labyrinth of Illusions", "The Desert of Mirages", "The Forsaken Village"] 
                  for name in dangerous_names)
    
    def test_character_relationship_traversal(self, game_world):
        """Test traversing relationships between characters."""
        # Get kobuko's direct relationships
        kobuko_id = game_world.get_concept_id_by_name("kobuko")
        assert kobuko_id is not None
        
        related_concepts, related_relations = game_world.get_related_concepts(kobuko_id, hop=1)
        
        # kobuko should be connected to many NPCs
        assert len(related_concepts) >= 8  # Should have relationships with most NPCs
        
        # Check for specific relationships
        related_names = [concept["node_name"] for concept in related_concepts]
        assert "Eldrin" in related_names  # mentor
        assert "Lyra" in related_names   # companion
        assert "Naia" in related_names   # healer
        
        # Test multi-hop traversal
        related_concepts_2hop, related_relations_2hop = game_world.get_related_concepts(kobuko_id, hop=2)
        
        # Should find more concepts at 2 hops
        assert len(related_concepts_2hop) >= len(related_concepts)
    
    def test_event_location_relationships(self, game_world):
        """Test relationships between events and locations."""
        # Get "The Betrayal of Eldrin" event
        betrayal_id = game_world.get_concept_id_by_name("The Betrayal of Eldrin")
        assert betrayal_id is not None
        
        related_concepts, related_relations = game_world.get_related_concepts(betrayal_id, hop=1)
        
        # Should be connected to Arcane Tower and involved NPCs
        related_names = [concept["node_name"] for concept in related_concepts]
        assert "The Arcane Tower" in related_names
        assert "kobuko" in related_names
        assert "Eldrin" in related_names
        
        # Check relation types
        relation_types = [relation["edge_type"] for relation in related_relations]
        assert "event-location" in relation_types
        assert "person-event" in relation_types
    
    def test_concept_updates_in_game_world(self, game_world):
        """Test updating concepts within the game world context."""
        # Get Thorgrim and update his description
        thorgrim_id = game_world.get_concept_id_by_name("Thorgrim")
        assert thorgrim_id is not None
        
        # Update Thorgrim's attributes
        game_world.update_concept(
            thorgrim_id,
            concept_attributes={
                "gender": "male",
                "age": "middle-aged", 
                "npc_description": "Thorgrim is a legendary dwarven weaponsmith and master warrior.",
                "special_ability": "legendary crafting"
            }
        )
        
        # Verify update
        updated_thorgrim = game_world.get_concept(thorgrim_id)
        assert "weaponsmith" in updated_thorgrim["node_attributes"]["npc_description"]
        assert updated_thorgrim["node_attributes"]["special_ability"] == "legendary crafting"
        
        # Test similarity search after update
        crafting_results = game_world.query_similar_concepts("legendary weaponsmith crafting", top_k=5)
        
        # Thorgrim should now appear in crafting-related searches
        crafting_names = [result[0]["node_name"] for result in crafting_results]
        assert "Thorgrim" in crafting_names
    
    def test_concept_deletion_impact(self, game_world):
        """Test the impact of deleting concepts on the game world."""
        # Get "The Forging of Destiny" event
        forging_id = game_world.get_concept_id_by_name("The Forging of Destiny")
        assert forging_id is not None
        
        # Verify it exists and has relationships
        forging_concept = game_world.get_concept(forging_id)
        assert forging_concept is not None
        
        related_concepts, related_relations = game_world.get_related_concepts(forging_id, hop=1)
        initial_relation_count = len(related_relations)
        assert initial_relation_count > 0
        
        # Delete the event
        game_world.delete_concept(forging_id)
        
        # Verify it's gone
        assert game_world.get_concept(forging_id) is None
        
        # Verify it doesn't appear in searches
        search_results = game_world.query_similar_concepts("forging destiny sword", top_k=10)
        result_names = [result[0]["node_name"] for result in search_results]
        assert "The Forging of Destiny" not in result_names
        
        # Verify related concepts still exist but relations are cleaned up
        thorgrim_id = game_world.get_concept_id_by_name("Thorgrim")
        thorgrim_relations = game_world.get_related_concepts(thorgrim_id, hop=1)[1]
        
        # Thorgrim should no longer be connected to the deleted event
        thorgrim_relation_targets = [rel["target_node_id"] for rel in thorgrim_relations] + \
                                   [rel["source_node_id"] for rel in thorgrim_relations]
        assert forging_id not in thorgrim_relation_targets
    
    def test_cross_type_similarity_search(self, game_world):
        """Test similarity searches that span multiple concept types."""
        # Search for "dragon" - should find Rognark (NPC) and possibly dragon-related events
        dragon_results = game_world.query_similar_concepts("dragon ancient powerful", top_k=5)
        
        assert len(dragon_results) > 0
        dragon_names = [result[0]["node_name"] for result in dragon_results]
        
        # Should find Rognark (ancient dragon in human form)
        assert "Rognark" in dragon_names
        
        # Search for "forge" - should find Dwarven Forge (location) and Forging event
        forge_results = game_world.query_similar_concepts("forge crafting weapons", top_k=5)
        
        assert len(forge_results) > 0
        forge_names = [result[0]["node_name"] for result in forge_results]
        
        # Should find forge-related concepts
        assert "The Dwarven Forge" in forge_names
        
        # Search for "music" - should find Aria and Siren's Song event
        music_results = game_world.query_similar_concepts("music song enchanting", top_k=5)
        
        assert len(music_results) > 0
        music_names = [result[0]["node_name"] for result in music_results]
        
        # Should find music-related concepts
        assert any(name in ["Aria", "The Siren's Song"] for name in music_names)