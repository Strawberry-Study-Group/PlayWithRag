"""System tests for advanced concept graph operations."""

import pytest
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_graph.concept_graph import ConceptGraphFactory
from .config import get_test_config, check_test_readiness


class TestAdvancedOperations:
    """Test advanced concept graph operations and edge cases."""
    
    @pytest.fixture
    def concept_graph(self):
        """Create concept graph for advanced testing."""
        if not check_test_readiness(use_remote=False):
            pytest.skip("API keys not configured for testing")
        
        config = get_test_config(use_remote=False)
        graph = ConceptGraphFactory.create_from_config(
            config["concept_graph_config"], 
            config["file_store_config"],
            world_name="test_advanced"
        )
        graph.empty_graph()
        yield graph
        
        # Cleanup
        try:
            graph.empty_graph()
        except Exception:
            pass
    
    def test_multi_hop_relationship_traversal(self, concept_graph):
        """Test traversing relationships across multiple hops."""
        # Create a chain of relationships: A -> B -> C -> D
        concepts = []
        for i, name in enumerate(["Alpha", "Beta", "Gamma", "Delta"]):
            concept_id = concept_graph.add_concept(
                name, "test_type", {"position": i, "description": f"Concept {name}"}
            )
            concepts.append((name, concept_id))
        
        # Create chain relationships
        for i in range(len(concepts) - 1):
            concept_graph.add_relation(concepts[i][1], concepts[i+1][1], "leads_to")
        
        # Test 1-hop traversal from Alpha
        related_1hop, relations_1hop = concept_graph.get_related_concepts(concepts[0][1], hop=1)
        assert len(related_1hop) == 1
        assert related_1hop[0]["node_name"] == "Beta"
        
        # Test 2-hop traversal from Alpha
        related_2hop, relations_2hop = concept_graph.get_related_concepts(concepts[0][1], hop=2)
        related_names_2hop = [c["node_name"] for c in related_2hop]
        assert "Beta" in related_names_2hop
        assert "Gamma" in related_names_2hop
        assert len(related_2hop) >= 2
        
        # Test 3-hop traversal from Alpha
        related_3hop, relations_3hop = concept_graph.get_related_concepts(concepts[0][1], hop=3)
        related_names_3hop = [c["node_name"] for c in related_3hop]
        assert "Beta" in related_names_3hop
        assert "Gamma" in related_names_3hop
        assert "Delta" in related_names_3hop
        assert len(related_3hop) >= 3
    
    def test_large_scale_similarity_search(self, concept_graph):
        """Test similarity search with larger numbers of concepts."""
        # Create a larger set of diverse concepts
        concept_types = {
            "warriors": [
                ("Spartan", "A disciplined Greek warrior"),
                ("Viking", "A fierce Norse raider"),
                ("Samurai", "An honorable Japanese warrior"),
                ("Knight", "A chivalrous medieval warrior"),
                ("Berserker", "A frenzied battle warrior")
            ],
            "mages": [
                ("Wizard", "A wise practitioner of arcane magic"),
                ("Sorcerer", "A natural-born magic wielder"),
                ("Warlock", "A magic user bound by pacts"),
                ("Enchanter", "A specialist in magical enchantments"),
                ("Necromancer", "A master of death magic")
            ]
        }
        
        created_concepts = {}
        for category, concepts in concept_types.items():
            for name, description in concepts:
                concept_id = concept_graph.add_concept(
                    name, category[:-1], {"description": description}
                )
                created_concepts[name] = concept_id
        
        # Test search for warrior concepts
        warrior_results = concept_graph.query_similar_concepts("fierce battle warrior", top_k=10)
        warrior_names = [result[0]["node_name"] for result in warrior_results]
        
        # Should find multiple warrior types
        warrior_concepts = ["Spartan", "Viking", "Samurai", "Knight", "Berserker"]
        found_warriors = [name for name in warrior_names if name in warrior_concepts]
        assert len(found_warriors) >= 2  # Should find at least 2 warrior types
        
        # Test search for magic concepts
        magic_results = concept_graph.query_similar_concepts("magic spells arcane", top_k=10)
        magic_names = [result[0]["node_name"] for result in magic_results]
        
        # Should find multiple mage types
        mage_concepts = ["Wizard", "Sorcerer", "Warlock", "Enchanter", "Necromancer"]
        found_mages = [name for name in magic_names if name in mage_concepts]
        assert len(found_mages) >= 2  # Should find at least 2 mage types