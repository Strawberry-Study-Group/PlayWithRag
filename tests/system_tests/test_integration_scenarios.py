"""System tests for integration scenarios and real-world usage patterns."""

import pytest
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from memory.memory import ConceptGraphFactory, ConceptGraphService
from .config import get_test_config, check_test_readiness


class TestIntegrationScenarios:
    """Test real-world integration scenarios and usage patterns."""
    
    @pytest.fixture
    def persistent_graph(self):
        """Create a graph that persists across test operations."""
        if not check_test_readiness(use_remote=False):
            pytest.skip("API keys not configured for testing")
        
        config = get_test_config(use_remote=False)
        graph = ConceptGraphFactory.create_from_config(
            config["concept_graph_config"], 
            config["file_store_config"],
            memory_core_name="test_integration"
        )
        graph.empty_graph()
        yield graph
        
        # Cleanup
        try:
            graph.empty_graph()
        except Exception:
            pass
    
    def test_game_session_simulation(self, persistent_graph):
        """Simulate a complete game session with progressive world building."""
        # Phase 1: Initial world setup
        world_concepts = self._create_initial_world(persistent_graph)
        
        # Phase 2: Player actions and world evolution
        self._simulate_player_actions(persistent_graph, world_concepts)
        
        # Phase 3: Dynamic content generation
        self._simulate_dynamic_content(persistent_graph, world_concepts)
        
        # Phase 4: World state queries
        self._verify_world_state(persistent_graph, world_concepts)
    
    def _create_initial_world(self, graph: ConceptGraphService) -> Dict[str, str]:
        """Create initial game world concepts."""
        concepts = {}
        
        # Create starting location
        concepts["village"] = graph.add_concept(
            "Riverside Village", "location",
            {
                "description": "A peaceful village by the river where the adventure begins",
                "population": 200,
                "resources": ["water", "fish", "crops"]
            }
        )
        
        # Create starting NPCs
        concepts["blacksmith"] = graph.add_concept(
            "Gareth the Blacksmith", "npc",
            {
                "profession": "blacksmith",
                "personality": "gruff but helpful",
                "services": ["weapon repair", "armor crafting"]
            }
        )
        
        concepts["innkeeper"] = graph.add_concept(
            "Martha the Innkeeper", "npc", 
            {
                "profession": "innkeeper",
                "personality": "warm and welcoming",
                "services": ["lodging", "meals", "local information"]
            }
        )
        
        # Create initial quests
        concepts["missing_caravan"] = graph.add_concept(
            "The Missing Caravan", "quest",
            {
                "type": "investigation",
                "description": "A merchant caravan has gone missing on the forest road",
                "difficulty": "medium",
                "rewards": ["gold", "reputation"]
            }
        )
        
        # Create relationships
        graph.add_relation(concepts["blacksmith"], concepts["village"], "lives_in")
        graph.add_relation(concepts["innkeeper"], concepts["village"], "lives_in")
        graph.add_relation(concepts["missing_caravan"], concepts["village"], "originates_from")
        graph.add_relation(concepts["innkeeper"], concepts["missing_caravan"], "provides_info_about")
        
        graph.save_graph()
        return concepts
    
    def _simulate_player_actions(self, graph: ConceptGraphService, world_concepts: Dict[str, str]):
        """Simulate player actions that modify the world."""
        # Player investigates the missing caravan
        forest_road_id = graph.add_concept(
            "Forest Road", "location",
            {
                "description": "A winding road through dense woods, known for bandit activity",
                "danger_level": "moderate",
                "features": ["ancient trees", "hidden paths", "abandoned camp"]
            }
        )
        world_concepts["forest_road"] = forest_road_id
        
        # Player discovers bandit camp
        bandit_camp_id = graph.add_concept(
            "Bandit Hideout", "location",
            {
                "description": "A hidden camp where the bandits store their loot",
                "danger_level": "high", 
                "loot": ["stolen goods", "weapons", "gold"]
            }
        )
        world_concepts["bandit_camp"] = bandit_camp_id
        
        # Create bandit leader
        bandit_leader_id = graph.add_concept(
            "Scarred Jake", "npc",
            {
                "type": "antagonist",
                "description": "The ruthless leader of the forest bandits",
                "abilities": ["intimidation", "sword fighting"],
                "motivation": "greed and power"
            }
        )
        world_concepts["bandit_leader"] = bandit_leader_id
        
        # Update quest with new information
        graph.update_concept(
            world_concepts["missing_caravan"],
            concept_attributes={
                "type": "investigation",
                "description": "A merchant caravan was attacked by bandits on the forest road",
                "difficulty": "medium",
                "rewards": ["gold", "reputation"],
                "status": "in_progress",
                "clues_found": ["bandit tracks", "torn fabric", "hidden camp location"]
            }
        )
        
        # Add new relationships
        graph.add_relation(forest_road_id, world_concepts["village"], "connects_to")
        graph.add_relation(bandit_camp_id, forest_road_id, "hidden_along")
        graph.add_relation(bandit_leader_id, bandit_camp_id, "commands_from")
        graph.add_relation(world_concepts["missing_caravan"], bandit_leader_id, "caused_by")
        
        graph.save_graph()
    
    def _simulate_dynamic_content(self, graph: ConceptGraphService, world_concepts: Dict[str, str]):
        """Simulate dynamic content generation based on player progress."""
        # Generate consequence of player actions - village reaction
        village_guard_id = graph.add_concept(
            "Captain Marcus", "npc",
            {
                "profession": "guard captain",
                "personality": "dutiful and strategic", 
                "equipment": ["steel sword", "chainmail armor"],
                "mission": "protect the village from bandit threats"
            }
        )
        world_concepts["village_guard"] = village_guard_id
        
        # Create follow-up quest
        cleanup_quest_id = graph.add_concept(
            "Securing the Roads", "quest",
            {
                "type": "combat",
                "description": "Help Captain Marcus eliminate the remaining bandit threat",
                "difficulty": "hard",
                "prerequisites": ["The Missing Caravan"],
                "rewards": ["village recognition", "guard equipment", "safe passage"]
            }
        )
        world_concepts["cleanup_quest"] = cleanup_quest_id
        
        # Create new location unlocked by progress
        ancient_ruins_id = graph.add_concept(
            "Ancient Ruins", "location",
            {
                "description": "Mysterious ruins discovered behind the bandit camp",
                "age": "centuries old",
                "features": ["crumbling walls", "magical inscriptions", "hidden chamber"],
                "danger_level": "unknown"
            }
        )
        world_concepts["ancient_ruins"] = ancient_ruins_id
        
        # Add relationships for dynamic content
        graph.add_relation(village_guard_id, world_concepts["village"], "protects")
        graph.add_relation(cleanup_quest_id, village_guard_id, "given_by")
        graph.add_relation(ancient_ruins_id, world_concepts["bandit_camp"], "discovered_near")
        graph.add_relation(world_concepts["blacksmith"], cleanup_quest_id, "supports_with_equipment")
        
        graph.save_graph()
    
    def _verify_world_state(self, graph: ConceptGraphService, world_concepts: Dict[str, str]):
        """Verify the current state of the world through queries."""
        # Test location-based queries
        village_results = graph.query_similar_concepts("peaceful village community", top_k=5)
        village_names = [result[0]["node_name"] for result in village_results]
        assert "Riverside Village" in village_names
        
        # Test character role queries
        npc_results = graph.query_similar_concepts("helpful village blacksmith", top_k=5)
        npc_names = [result[0]["node_name"] for result in npc_results]
        assert "Gareth the Blacksmith" in npc_names
        
        # Test quest progression queries  
        quest_results = graph.query_similar_concepts("bandit investigation missing caravan", top_k=5)
        quest_names = [result[0]["node_name"] for result in quest_results]
        assert "The Missing Caravan" in quest_names
        
        # Test relationship traversal for story connections
        village_related, village_relations = graph.get_related_concepts(world_concepts["village"], hop=2)
        
        # Should find NPCs, quests, and connected locations
        related_names = [concept["node_name"] for concept in village_related]
        assert "Gareth the Blacksmith" in related_names
        assert "Martha the Innkeeper" in related_names
        assert "The Missing Caravan" in related_names
        
        # Test antagonist discovery
        bandit_results = graph.query_similar_concepts("bandit leader forest threat", top_k=5)
        bandit_names = [result[0]["node_name"] for result in bandit_results]
        assert "Scarred Jake" in bandit_names
    
    def test_knowledge_base_evolution(self, persistent_graph):
        """Test how the knowledge base evolves with new information."""
        # Initial knowledge state
        initial_concepts = self._create_academic_knowledge_base(persistent_graph)
        
        # Add new research and discoveries
        self._simulate_knowledge_discovery(persistent_graph, initial_concepts)
        
        # Test knowledge connections and inference
        self._verify_knowledge_connections(persistent_graph, initial_concepts)
    
    def _create_academic_knowledge_base(self, graph: ConceptGraphService) -> Dict[str, str]:
        """Create an academic knowledge base scenario."""
        concepts = {}
        
        # Create research areas
        concepts["ai_research"] = graph.add_concept(
            "Artificial Intelligence Research", "field",
            {
                "description": "Study of intelligent agents and machine learning",
                "subfields": ["machine learning", "natural language processing", "computer vision"],
                "applications": ["automation", "data analysis", "robotics"]
            }
        )
        
        concepts["neuroscience"] = graph.add_concept(
            "Neuroscience", "field",
            {
                "description": "Scientific study of the nervous system and brain",
                "methods": ["brain imaging", "electrophysiology", "behavioral studies"],
                "focus_areas": ["cognition", "perception", "memory"]
            }
        )
        
        # Create researchers
        concepts["dr_smith"] = graph.add_concept(
            "Dr. Sarah Smith", "researcher",
            {
                "specialization": "machine learning algorithms",
                "institution": "Tech University",
                "research_focus": "neural network optimization"
            }
        )
        
        concepts["dr_jones"] = graph.add_concept(
            "Dr. Michael Jones", "researcher", 
            {
                "specialization": "cognitive neuroscience",
                "institution": "Medical Institute",
                "research_focus": "brain-computer interfaces"
            }
        )
        
        # Create initial research projects
        concepts["neural_net_project"] = graph.add_concept(
            "Advanced Neural Networks", "project",
            {
                "description": "Development of more efficient neural network architectures",
                "funding": "government grant",
                "timeline": "3 years",
                "goals": ["improved accuracy", "reduced computation"]
            }
        )
        
        # Create relationships
        graph.add_relation(concepts["dr_smith"], concepts["ai_research"], "researches_in")
        graph.add_relation(concepts["dr_jones"], concepts["neuroscience"], "researches_in") 
        graph.add_relation(concepts["dr_smith"], concepts["neural_net_project"], "leads")
        graph.add_relation(concepts["neural_net_project"], concepts["ai_research"], "contributes_to")
        
        graph.save_graph()
        return concepts
    
    def _simulate_knowledge_discovery(self, graph: ConceptGraphService, concepts: Dict[str, str]):
        """Simulate new discoveries and research connections."""
        # New breakthrough discovery
        breakthrough_id = graph.add_concept(
            "Bio-Inspired Neural Architecture", "discovery",
            {
                "description": "Neural network design inspired by biological brain structures",
                "significance": "major breakthrough",
                "applications": ["brain-computer interfaces", "cognitive modeling"],
                "implications": ["better AI-brain understanding", "medical applications"]
            }
        )
        concepts["breakthrough"] = breakthrough_id
        
        # Collaborative research project
        collab_project_id = graph.add_concept(
            "AI-Neuroscience Collaboration", "project",
            {
                "description": "Joint project combining AI research with neuroscience insights",
                "type": "interdisciplinary collaboration",
                "funding": "joint institutional grant",
                "expected_outcomes": ["novel algorithms", "brain understanding"]
            }
        )
        concepts["collaboration"] = collab_project_id
        
        # New researcher joining
        new_researcher_id = graph.add_concept(
            "Dr. Lisa Chen", "researcher",
            {
                "specialization": "computational neuroscience",
                "background": "PhD in both computer science and neuroscience",
                "unique_skills": ["interdisciplinary research", "algorithm development"]
            }
        )
        concepts["dr_chen"] = new_researcher_id
        
        # Update existing project with new discoveries
        graph.update_concept(
            concepts["neural_net_project"],
            concept_attributes={
                "description": "Development of bio-inspired neural network architectures",
                "funding": "government grant + industry partnership",
                "timeline": "3 years", 
                "goals": ["improved accuracy", "reduced computation", "biological plausibility"],
                "recent_breakthrough": "bio-inspired architecture discovery"
            }
        )
        
        # Add new relationships showing knowledge evolution
        graph.add_relation(breakthrough_id, concepts["neural_net_project"], "emerged_from")
        graph.add_relation(breakthrough_id, concepts["ai_research"], "advances")
        graph.add_relation(breakthrough_id, concepts["neuroscience"], "bridges_to")
        graph.add_relation(collab_project_id, concepts["dr_smith"], "includes")
        graph.add_relation(collab_project_id, concepts["dr_jones"], "includes")
        graph.add_relation(new_researcher_id, collab_project_id, "joins")
        graph.add_relation(new_researcher_id, breakthrough_id, "contributes_to")
        
        graph.save_graph()
    
    def _verify_knowledge_connections(self, graph: ConceptGraphService, concepts: Dict[str, str]):
        """Verify knowledge connections and cross-field relationships."""
        # Test interdisciplinary discovery
        interdisciplinary_results = graph.query_similar_concepts(
            "bio-inspired AI neuroscience collaboration", top_k=5
        )
        
        interdisciplinary_names = [result[0]["node_name"] for result in interdisciplinary_results]
        assert any(name in ["Bio-Inspired Neural Architecture", "AI-Neuroscience Collaboration"] 
                  for name in interdisciplinary_names)
        
        # Test researcher network traversal
        dr_smith_network, smith_relations = graph.get_related_concepts(concepts["dr_smith"], hop=2)
        smith_network_names = [concept["node_name"] for concept in dr_smith_network]
        
        # Should connect to other researchers through projects
        assert "Dr. Michael Jones" in smith_network_names
        assert "Dr. Lisa Chen" in smith_network_names
        
        # Test knowledge flow through projects
        project_related, project_relations = graph.get_related_concepts(
            concepts["neural_net_project"], hop=2
        )
        
        project_network_names = [concept["node_name"] for concept in project_related]
        
        # Should connect research fields through discoveries
        assert "Artificial Intelligence Research" in project_network_names
        assert "Bio-Inspired Neural Architecture" in project_network_names
    
    def test_content_recommendation_system(self, persistent_graph):
        """Test using the concept graph as a content recommendation system."""
        # Create content library
        content_concepts = self._create_content_library(persistent_graph)
        
        # Simulate user interactions
        self._simulate_user_preferences(persistent_graph, content_concepts)
        
        # Test recommendation generation
        self._verify_content_recommendations(persistent_graph, content_concepts)
    
    def _create_content_library(self, graph: ConceptGraphService) -> Dict[str, str]:
        """Create a content library for recommendation testing."""
        concepts = {}
        
        # Create content categories
        content_items = [
            ("Sci-Fi Novel: Dune", "book", {
                "genre": "science fiction",
                "themes": ["politics", "ecology", "mysticism"],
                "setting": "desert planet",
                "complexity": "high"
            }),
            ("Fantasy Novel: Lord of the Rings", "book", {
                "genre": "fantasy", 
                "themes": ["good vs evil", "friendship", "heroism"],
                "setting": "medieval fantasy world",
                "complexity": "high"
            }),
            ("Documentary: Planet Earth", "video", {
                "genre": "nature documentary",
                "themes": ["wildlife", "conservation", "natural beauty"],
                "format": "video series",
                "educational_value": "high"
            }),
            ("Action Movie: Mad Max", "movie", {
                "genre": "action",
                "themes": ["survival", "post-apocalyptic", "vehicles"],
                "setting": "wasteland",
                "intensity": "high"
            }),
            ("Podcast: Science Talk", "audio", {
                "genre": "educational podcast",
                "themes": ["scientific discoveries", "interviews", "research"],
                "format": "weekly episodes",
                "educational_value": "high"
            })
        ]
        
        for name, content_type, attributes in content_items:
            concepts[name.lower().replace(" ", "_")] = graph.add_concept(
                name, content_type, attributes
            )
        
        # Create user profiles
        concepts["user_alice"] = graph.add_concept(
            "Alice Johnson", "user",
            {
                "age": 28,
                "interests": ["science fiction", "technology", "space exploration"],
                "education": "engineering degree",
                "preferences": ["complex narratives", "scientific accuracy"]
            }
        )
        
        concepts["user_bob"] = graph.add_concept(
            "Bob Smith", "user",
            {
                "age": 35,
                "interests": ["fantasy", "adventure", "mythology"], 
                "education": "literature degree",
                "preferences": ["character development", "world building"]
            }
        )
        
        graph.save_graph()
        return concepts
    
    def _simulate_user_preferences(self, graph: ConceptGraphService, concepts: Dict[str, str]):
        """Simulate user interactions and preference learning."""
        # Alice likes sci-fi content
        graph.add_relation(concepts["user_alice"], concepts["sci-fi_novel:_dune"], "likes")
        graph.add_relation(concepts["user_alice"], concepts["podcast:_science_talk"], "likes")
        graph.add_relation(concepts["user_alice"], concepts["documentary:_planet_earth"], "somewhat_likes")
        
        # Bob likes fantasy content
        graph.add_relation(concepts["user_bob"], concepts["fantasy_novel:_lord_of_the_rings"], "loves")
        graph.add_relation(concepts["user_bob"], concepts["action_movie:_mad_max"], "likes")
        graph.add_relation(concepts["user_bob"], concepts["sci-fi_novel:_dune"], "dislikes")
        
        # Create content similarity relationships
        graph.add_relation(
            concepts["sci-fi_novel:_dune"], 
            concepts["podcast:_science_talk"], 
            "shares_scientific_themes"
        )
        graph.add_relation(
            concepts["fantasy_novel:_lord_of_the_rings"],
            concepts["action_movie:_mad_max"],
            "shares_adventure_themes"
        )
        
        graph.save_graph()
    
    def _verify_content_recommendations(self, graph: ConceptGraphService, concepts: Dict[str, str]):
        """Verify content recommendation generation."""
        # Test recommendations for Alice (sci-fi fan)
        alice_related, alice_relations = graph.get_related_concepts(concepts["user_alice"], hop=2)
        
        alice_liked_content = []
        for relation in alice_relations:
            if relation["edge_type"] in ["likes", "loves", "somewhat_likes"]:
                target_concept = graph.get_concept(relation["target_node_id"])
                if target_concept and target_concept["node_type"] in ["book", "video", "audio", "movie"]:
                    alice_liked_content.append(target_concept["node_name"])
        
        # Should find Alice's preferred content
        assert "Sci-Fi Novel: Dune" in alice_liked_content
        assert "Podcast: Science Talk" in alice_liked_content
        
        # Test content similarity search for Alice
        alice_recommendations = graph.query_similar_concepts(
            "science fiction technology space educational", top_k=5
        )
        
        alice_rec_names = [result[0]["node_name"] for result in alice_recommendations]
        
        # Should recommend content matching Alice's interests
        assert any(name in ["Sci-Fi Novel: Dune", "Podcast: Science Talk", "Documentary: Planet Earth"] 
                  for name in alice_rec_names)
        
        # Test recommendations for Bob (fantasy fan)
        bob_related, bob_relations = graph.get_related_concepts(concepts["user_bob"], hop=2)
        
        bob_recommendations = graph.query_similar_concepts(
            "fantasy adventure heroism mythology", top_k=5
        )
        
        bob_rec_names = [result[0]["node_name"] for result in bob_recommendations]
        
        # Should recommend content matching Bob's interests
        assert any(name in ["Fantasy Novel: Lord of the Rings", "Action Movie: Mad Max"] 
                  for name in bob_rec_names)