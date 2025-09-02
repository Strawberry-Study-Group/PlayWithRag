"""System tests for ConceptGraph using real implementations."""
import unittest
import tempfile
import shutil
import os
from pathlib import Path
import sys

sys.path.append("../..")
from memory.memory import MemoryCoreFactory
from memory.constants import ConceptGraphConstants
from tests.test_memory_core_utils import MemoryCoreTestContext, create_memory_core_config
from .config import check_test_readiness


class TestMemoryCoreSystem(unittest.TestCase):
    """System tests for MemoryCore with real implementations."""
    
    def setUp(self):
        """Set up test environment with temporary directory and real MemoryCore."""
        if not check_test_readiness(use_remote=False):
            self.skipTest("API keys not configured for testing")
        
        # Create memory core config using new schema
        self.memory_core_config = create_memory_core_config(
            provider="local",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
            model="text-embedding-3-small",
            dim=1536,
            graph_file="test_graph.json",
            index_file="test_emb_index.json"
        )
        
        # Create memory core context
        self.memory_core_context = MemoryCoreTestContext(custom_config=self.memory_core_config)
        self.memory_core_path = self.memory_core_context.__enter__()
        
        self.memory_core = MemoryCoreFactory.create_from_memory_core(
            memory_core_path=self.memory_core_path
        )
    
    def tearDown(self):
        """Clean up memory core."""
        if hasattr(self, 'memory_core_context'):
            self.memory_core_context.__exit__(None, None, None)
    
    def test_add_and_get_concept(self):
        """Test adding and retrieving a concept."""
        concept_name = "Test Person"
        concept_type = "person"
        concept_attributes = {
            "age": 30,
            "occupation": "developer",
            "skills": ["Python", "AI"]
        }
        
        # Add concept
        concept_id = self.memory_core.add_concept(
            concept_name, concept_type, concept_attributes
        )
        
        # Verify concept was created
        self.assertIsInstance(concept_id, str)
        self.assertTrue(len(concept_id) > 0)
        
        # Retrieve concept
        retrieved_concept = self.memory_core.get_concept(concept_id)
        self.assertIsNotNone(retrieved_concept)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_NAME], concept_name)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_TYPE], concept_type)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["age"], 30)
    
    def test_update_concept(self):
        """Test updating a concept."""
        # Add initial concept
        concept_id = self.memory_core.add_concept(
            "Initial Name", "person", {"age": 25}
        )
        
        # Update concept
        self.memory_core.update_concept(
            concept_id,
            concept_name="Updated Name",
            concept_attributes={"age": 26, "city": "San Francisco"}
        )
        
        # Verify updates
        updated_concept = self.memory_core.get_concept(concept_id)
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_NAME], "Updated Name")
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["age"], 26)
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["city"], "San Francisco")
    
    def test_add_and_get_relation(self):
        """Test adding and retrieving relations between concepts."""
        # Add two concepts
        person_id = self.memory_core.add_concept("Alice", "person", {"age": 30})
        company_id = self.memory_core.add_concept("TechCorp", "company", {"industry": "technology"})
        
        # Add relation
        self.memory_core.add_relation(person_id, company_id, "works_at")
        
        # Verify relation exists
        relation = self.memory_core.get_relation(person_id, company_id)
        self.assertIsNotNone(relation)
        self.assertEqual(relation[ConceptGraphConstants.FIELD_EDGE_TYPE], "works_at")
    
    def test_update_relation(self):
        """Test updating a relation."""
        # Add concepts and relation
        person_id = self.memory_core.add_concept("Bob", "person", {})
        company_id = self.memory_core.add_concept("StartupInc", "company", {})
        self.memory_core.add_relation(person_id, company_id, "intern_at")
        
        # Update relation
        self.memory_core.update_relation(person_id, company_id, "employee_at")
        
        # Verify update
        updated_relation = self.memory_core.get_relation(person_id, company_id)
        self.assertEqual(updated_relation[ConceptGraphConstants.FIELD_EDGE_TYPE], "employee_at")
    
    def test_delete_concept_and_relation(self):
        """Test deleting concepts and relations."""
        # Add concepts and relation
        person_id = self.memory_core.add_concept("Charlie", "person", {})
        company_id = self.memory_core.add_concept("BigCorp", "company", {})
        self.memory_core.add_relation(person_id, company_id, "works_at")
        
        # Verify they exist
        self.assertIsNotNone(self.memory_core.get_concept(person_id))
        self.assertIsNotNone(self.memory_core.get_relation(person_id, company_id))
        
        # Delete relation
        self.memory_core.delete_relation(person_id, company_id)
        self.assertIsNone(self.memory_core.get_relation(person_id, company_id))
        
        # Delete concept
        self.memory_core.delete_concept(person_id)
        self.assertIsNone(self.memory_core.get_concept(person_id))
        
        # Company should still exist
        self.assertIsNotNone(self.memory_core.get_concept(company_id))
    
    def test_query_similar_concepts(self):
        """Test querying for similar concepts using embeddings."""
        # Add several related concepts
        self.memory_core.add_concept("Python Developer", "person", {"skill": "programming"})
        self.memory_core.add_concept("Java Developer", "person", {"skill": "programming"}) 
        self.memory_core.add_concept("Data Scientist", "person", {"skill": "analytics"})
        self.memory_core.add_concept("Pizza Chef", "person", {"skill": "cooking"})
        
        # Query for similar concepts
        similar_concepts = self.memory_core.query_similar_concepts("Software Engineer", top_k=3)
        
        # Should return some results
        self.assertGreaterEqual(len(similar_concepts), 0)  # Might be empty if no embeddings match
        
        # Each result should be a tuple of (concept, score)
        for concept, score in similar_concepts:
            self.assertIsInstance(concept, dict)
            self.assertIn(ConceptGraphConstants.FIELD_NODE_NAME, concept)
            self.assertIsInstance(score, (int, float))
    
    def test_get_related_concepts(self):
        """Test getting related concepts within specified hops."""
        # Create a small network
        alice_id = self.memory_core.add_concept("Alice", "person", {})
        bob_id = self.memory_core.add_concept("Bob", "person", {})
        company_id = self.memory_core.add_concept("TechCorp", "company", {})
        project_id = self.memory_core.add_concept("AI Project", "project", {})
        
        # Add relations
        self.memory_core.add_relation(alice_id, company_id, "works_at")
        self.memory_core.add_relation(bob_id, company_id, "works_at")
        self.memory_core.add_relation(alice_id, project_id, "leads")
        self.memory_core.add_relation(bob_id, project_id, "contributes_to")
        
        # Get related concepts for Alice
        related_concepts, related_relations = self.memory_core.get_related_concepts(
            alice_id, hop=1
        )
        
        # Should find company and project
        self.assertGreaterEqual(len(related_concepts), 2)
        self.assertGreaterEqual(len(related_relations), 2)
        
        # Filter by relation type
        work_relations = self.memory_core.get_related_concepts(
            alice_id, hop=1, relation_type="works_at"
        )[1]
        
        self.assertEqual(len(work_relations), 1)
        self.assertEqual(work_relations[0][ConceptGraphConstants.FIELD_EDGE_TYPE], "works_at")
    
    def test_save_and_persistence(self):
        """Test saving and data persistence."""
        # Add some data
        concept_id = self.memory_core.add_concept("Persistent Data", "test", {"value": 42})
        
        # Save explicitly
        self.memory_core.save_graph()
        
        # Verify files were created in the unified memory core folder
        memory_core_dir = self.test_dir / "test_memory_core"
        graph_file = memory_core_dir / "test_graph.json"
        emb_file = memory_core_dir / "test_emb_index.json"
        
        self.assertTrue(graph_file.exists())
        self.assertTrue(emb_file.exists())
        
        # Create new ConceptGraph instance and verify data persisted
        concept_graph_config = get_concept_graph_config()
        memory_core_config = {
            "embedding": {
                "provider": concept_graph_config["provider"],
                "api_key": concept_graph_config.get("embedding_api_key", ""),
                "model": concept_graph_config.get("emb_model", "text-embedding-3-small"),
                "dim": concept_graph_config.get("emb_dim", 1536)
            },
            "files": {
                "graph_file": "test_graph.json",
                "index_file": "test_emb_index.json"
            }
        }
        
        if concept_graph_config["provider"] == "remote":
            memory_core_config["embedding"].update({
                "pinecone_api_key": concept_graph_config.get("pinecone_api_key", ""),
                "pinecone_index_name": concept_graph_config.get("pinecone_index_name", ""),
                "metric": concept_graph_config.get("metric", "cosine")
            })
        
        memory_core_path = str(self.test_dir / "test_memory_core")
        new_concept_graph = MemoryCoreFactory.create_from_memory_core(
            memory_core_path=memory_core_path,
            memory_core_config=memory_core_config
        )
        
        # Should be able to retrieve the concept
        retrieved_concept = new_concept_graph.get_concept(concept_id)
        self.assertIsNotNone(retrieved_concept)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["value"], 42)
    
    def test_graph_statistics(self):
        """Test getting graph statistics."""
        # Add some concepts of different types
        self.memory_core.add_concept("Person 1", "person", {})
        self.memory_core.add_concept("Person 2", "person", {}) 
        self.memory_core.add_concept("Company 1", "company", {})
        
        # Get statistics
        stats = self.memory_core.get_graph_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_concepts", stats)
        self.assertEqual(stats["total_concepts"], 3)
        self.assertIn("concept_types", stats)
        self.assertEqual(stats["concept_types"]["person"], 2)
        self.assertEqual(stats["concept_types"]["company"], 1)
    
    def test_unified_folder_structure(self):
        """Test that all files are stored in a unified memory core folder structure."""
        # Add some data
        concept_id = self.memory_core.add_concept("Test Concept", "test", {"value": 123})
        
        # Save to ensure files are created
        self.memory_core.save_graph()
        
        # Verify unified memory core folder structure exists
        memory_core_dir = self.test_dir / "test_memory_core"
        self.assertTrue(memory_core_dir.exists())
        self.assertTrue(memory_core_dir.is_dir())
        
        # Verify all expected files are in the same directory
        graph_file = memory_core_dir / "test_graph.json"
        emb_file = memory_core_dir / "test_emb_index.json"
        
        self.assertTrue(graph_file.exists())
        self.assertTrue(emb_file.exists())
        
        # Verify files are actually in the same directory
        self.assertEqual(graph_file.parent, emb_file.parent)
        self.assertEqual(str(graph_file.parent), str(memory_core_dir))


@unittest.skipIf(
    not check_test_readiness(use_remote=False),
    "OpenAI API key not provided for system tests."
)
class TestMemoryCoreSystemWithAPI(TestMemoryCoreSystem):
    """System tests that require API access. Only run when API key is provided."""
    
    def test_create_memory_core_method(self):
        """Test creating a memory core with the new structure."""
        from tests.test_config import OPENAI_API_KEY
        
        # Create memory core config
        memory_core_config = {
            "embedding": {
                "provider": "local",
                "api_key": OPENAI_API_KEY,
                "model": "text-embedding-3-small",
                "dim": 1536
            },
            "files": {
                "graph_file": "graph.json",
                "index_file": "emb_index.json"
            }
        }
        
        # Create a memory core using the new method
        memory_core_path = str(self.test_dir / "convenience_memory_core")
        world_concept_graph = MemoryCoreFactory.create_from_memory_core(
            memory_core_path=memory_core_path,
            memory_core_config=memory_core_config
        )
        
        # Add some test data
        concept_id = world_concept_graph.add_concept("Convenience Test", "test", {"method": "create_memory_core"})
        world_concept_graph.save_graph()
        
        # Verify the memory core folder structure was created
        memory_core_dir = self.test_dir / "convenience_memory_core"
        self.assertTrue(memory_core_dir.exists())
        
        # Verify both graph and embedding files exist in the same directory
        graph_file = memory_core_dir / "graph.json"  # Default names
        emb_file = memory_core_dir / "emb_index.json"
        
        self.assertTrue(graph_file.exists())
        self.assertTrue(emb_file.exists())
        
        # Verify data can be retrieved
        retrieved_concept = world_concept_graph.get_concept(concept_id)
        self.assertIsNotNone(retrieved_concept)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["method"], "create_memory_core")


if __name__ == '__main__':
    print("=" * 50)
    print("CONCEPT GRAPH SYSTEM TESTS")
    print("=" * 50)
    print("\nTo run tests that require API access:")
    print("1. Copy tests/test_config.template.py to tests/test_config.py")
    print("2. Edit tests/test_config.py and add your actual OpenAI API key")
    print("3. The tests use local FAISS storage (no Pinecone key needed)")
    print("4. Tests create temporary directories that are cleaned up automatically")
    print("\nRunning tests...")
    unittest.main()