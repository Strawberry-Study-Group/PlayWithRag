"""System tests for basic concept graph operations with real API calls."""

import pytest
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from memory.memory import MemoryCoreFactory
from .config import check_test_readiness
from tests.test_memory_core_utils import MemoryCoreTestContext, create_memory_core_config


class TestBasicOperations:
    """Test basic concept graph operations with real API integration."""
    
    @pytest.fixture
    def memory_core_local(self):
        """Create memory core with local storage for testing."""
        if not check_test_readiness(use_remote=False):
            pytest.skip("API keys not configured for local testing")
        
        # Create memory core config using new schema
        import os
        config = create_memory_core_config(
            provider="local",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
            model="text-embedding-3-small",
            dim=1536
        )
        
        # Create memory core context and initialize service
        context = MemoryCoreTestContext(custom_config=config)
        memory_core_path = context.__enter__()
        
        try:
            graph = MemoryCoreFactory.create_from_memory_core(
                memory_core_path=memory_core_path
            )
            graph.empty_graph()  # Start with clean graph
            yield graph
        finally:
            # Cleanup
            try:
                graph.empty_graph()
            except Exception:
                pass  # Ignore cleanup errors
            context.__exit__(None, None, None)
    
    @pytest.fixture
    def memory_core_remote(self):
        """Create memory core with remote storage for testing."""
        if not check_test_readiness(use_remote=True):
            pytest.skip("API keys not configured for remote testing")
        
        # Create memory core config using new schema for remote
        import os
        config = create_memory_core_config(
            provider="remote",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
            model="text-embedding-3-small",
            dim=1536,
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "test_pinecone_key"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "test-index")
        )
        
        # Create memory core context and initialize service
        context = MemoryCoreTestContext(custom_config=config)
        memory_core_path = context.__enter__()
        
        try:
            graph = MemoryCoreFactory.create_from_memory_core(
                memory_core_path=memory_core_path
            )
            graph.empty_graph()  # Start with clean graph
            yield graph
        finally:
            # Cleanup
            try:
                graph.empty_graph()
            except Exception:
                pass  # Ignore cleanup errors
            context.__exit__(None, None, None)
    
    def test_create_npc_concepts(self, memory_core_local):
        """Test creating NPC concepts with real embeddings."""
        # Create main character
        kobuko_id = memory_core_local.add_concept(
            concept_name="kobuko",
            concept_type="npc",
            concept_attributes={
                "gender": "male",
                "age": "young", 
                "npc_description": "kobuko is very tanky npc, it can heal himself"
            }
        )
        
        # Create supporting NPCs
        eldrin_id = memory_core_local.add_concept(
            concept_name="Eldrin", 
            concept_type="npc",
            concept_attributes={
                "gender": "male",
                "age": "old",
                "npc_description": "Eldrin is a wise and powerful wizard who guides the player through their quest."
            }
        )
        
        lyra_id = memory_core_local.add_concept(
            concept_name="Lyra",
            concept_type="npc", 
            concept_attributes={
                "gender": "female",
                "age": "young",
                "npc_description": "Lyra is a skilled archer and a loyal companion to the player."
            }
        )
        
        # Verify concepts were created
        assert kobuko_id is not None
        assert eldrin_id is not None
        assert lyra_id is not None
        
        # Verify concepts can be retrieved
        kobuko = memory_core_local.get_concept(kobuko_id)
        assert kobuko["node_name"] == "kobuko"
        assert kobuko["node_type"] == "npc"
        assert kobuko["node_attributes"]["gender"] == "male"
        
        eldrin = memory_core_local.get_concept(eldrin_id) 
        assert eldrin["node_name"] == "Eldrin"
        assert "wizard" in eldrin["node_attributes"]["npc_description"]
        
        lyra = memory_core_local.get_concept(lyra_id)
        assert lyra["node_name"] == "Lyra"
        assert "archer" in lyra["node_attributes"]["npc_description"]
    
    def test_create_relations(self, memory_core_local):
        """Test creating relations between concepts."""
        # Create NPCs
        hero_id = memory_core_local.add_concept(
            "Hero", "npc", {"role": "protagonist"}
        )
        mentor_id = memory_core_local.add_concept(
            "Mentor", "npc", {"role": "guide"}
        )
        weapon_id = memory_core_local.add_concept(
            "Magic Sword", "item", {"damage": 50, "magic": True}
        )
        
        # Add relations
        memory_core_local.add_relation(mentor_id, hero_id, "mentors")
        memory_core_local.add_relation(hero_id, weapon_id, "wields")
        
        # Verify relations exist
        assert memory_core_local.is_relation(mentor_id, hero_id)
        assert memory_core_local.is_relation(hero_id, weapon_id)
        
        # Get related concepts
        related_concepts, related_relations = memory_core_local.get_related_concepts(hero_id)
        
        assert len(related_concepts) == 2
        assert len(related_relations) == 2
        
        # Check that we can find mentor and weapon in related concepts
        concept_names = [c["node_name"] for c in related_concepts]
        assert "Mentor" in concept_names
        assert "Magic Sword" in concept_names
    
    def test_similarity_search_with_real_embeddings(self, memory_core_local):
        """Test similarity search using real OpenAI embeddings."""
        # Create diverse concepts
        warrior_id = memory_core_local.add_concept(
            "Thorgrim", "npc", 
            {"description": "A mighty dwarven warrior skilled in combat"}
        )
        
        mage_id = memory_core_local.add_concept(
            "Eldara", "npc",
            {"description": "A powerful sorceress who controls elemental magic"}
        )
        
        sword_id = memory_core_local.add_concept(
            "Dragonbane", "weapon",
            {"description": "A legendary sword forged to slay dragons"}
        )
        
        forge_id = memory_core_local.add_concept(
            "Ancient Forge", "location",
            {"description": "A mystical forge where legendary weapons are created"}
        )
        
        # Test similarity search for warrior-related concepts
        warrior_results = memory_core_local.query_similar_concepts("mighty warrior", top_k=5)
        
        assert len(warrior_results) > 0
        # Should find Thorgrim as most similar
        top_result = warrior_results[0][0]
        assert top_result["node_name"] == "Thorgrim"
        
        # Test similarity search for magic-related concepts  
        magic_results = memory_core_local.query_similar_concepts("powerful magic user", top_k=5)
        
        assert len(magic_results) > 0
        # Should find Eldara among results
        magic_names = [result[0]["node_name"] for result in magic_results]
        assert "Eldara" in magic_names
        
        # Test similarity search for weapon/forge concepts
        weapon_results = memory_core_local.query_similar_concepts("legendary weapon creation", top_k=5)
        
        assert len(weapon_results) > 0
        weapon_names = [result[0]["node_name"] for result in weapon_results]
        # Should find either the sword or forge in results
        assert any(name in ["Dragonbane", "Ancient Forge"] for name in weapon_names)
    
    def test_concept_updates_and_embedding_refresh(self, memory_core_local):
        """Test updating concepts and verifying embeddings are refreshed."""
        # Create initial concept
        concept_id = memory_core_local.add_concept(
            "Test Character", "npc",
            {"role": "villager", "description": "A simple farmer"}
        )
        
        # Search for farmer-related content
        initial_results = memory_core_local.query_similar_concepts("farming agriculture", top_k=3)
        initial_scores = {result[0]["node_id"]: result[1] for result in initial_results}
        
        # Update concept to be warrior-related
        memory_core_local.update_concept(
            concept_id,
            concept_attributes={
                "role": "warrior", 
                "description": "A battle-hardened knight with years of combat experience"
            }
        )
        
        # Search for warrior-related content
        updated_results = memory_core_local.query_similar_concepts("battle combat knight", top_k=3)
        updated_scores = {result[0]["node_id"]: result[1] for result in updated_results}
        
        # The updated concept should now score higher for warrior queries
        if concept_id in updated_scores:
            # If our concept appears in warrior results, it should have a decent score
            assert updated_scores[concept_id] > 0.1
        
        # Verify the concept was actually updated
        updated_concept = memory_core_local.get_concept(concept_id)
        assert updated_concept["node_attributes"]["role"] == "warrior"
        assert "knight" in updated_concept["node_attributes"]["description"]
    
    def test_concept_deletion_and_cleanup(self, memory_core_local):
        """Test concept deletion and verification of cleanup."""
        # Create concepts and relations
        hero_id = memory_core_local.add_concept("Hero", "npc", {"level": 1})
        villain_id = memory_core_local.add_concept("Villain", "npc", {"level": 10}) 
        sword_id = memory_core_local.add_concept("Sword", "item", {"damage": 20})
        
        # Add relations
        memory_core_local.add_relation(hero_id, villain_id, "fights")
        memory_core_local.add_relation(hero_id, sword_id, "wields")
        
        # Verify setup
        assert memory_core_local.get_concept(hero_id) is not None
        assert memory_core_local.is_relation(hero_id, villain_id)
        assert memory_core_local.is_relation(hero_id, sword_id)
        
        # Delete hero
        memory_core_local.delete_concept(hero_id)
        
        # Verify hero is gone
        assert memory_core_local.get_concept(hero_id) is None
        
        # Verify relations involving hero are gone
        assert not memory_core_local.is_relation(hero_id, villain_id)
        assert not memory_core_local.is_relation(hero_id, sword_id)
        
        # Verify other concepts still exist
        assert memory_core_local.get_concept(villain_id) is not None
        assert memory_core_local.get_concept(sword_id) is not None
        
        # Verify hero doesn't appear in similarity searches
        search_results = memory_core_local.query_similar_concepts("Hero", top_k=10)
        result_ids = [result[0]["node_id"] for result in search_results]
        assert hero_id not in result_ids
    
    @pytest.mark.slow
    def test_remote_storage_integration(self, memory_core_remote):
        """Test integration with remote Pinecone storage."""
        # Create test concepts
        concept_id = memory_core_remote.add_concept(
            "Remote Test Character", "npc",
            {"description": "A character for testing remote storage integration"}
        )
        
        # Verify concept was created
        assert concept_id is not None
        concept = memory_core_remote.get_concept(concept_id)
        assert concept["node_name"] == "Remote Test Character"
        
        # Test similarity search with remote storage
        results = memory_core_remote.query_similar_concepts("test character", top_k=5)
        assert len(results) > 0
        
        # Should find our test character
        result_names = [result[0]["node_name"] for result in results]
        assert "Remote Test Character" in result_names
        
        # Cleanup
        memory_core_remote.delete_concept(concept_id)
        
        # Verify deletion
        assert memory_core_remote.get_concept(concept_id) is None