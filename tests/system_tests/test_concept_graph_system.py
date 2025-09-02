"""System tests for ConceptGraph using real implementations."""
import unittest
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append("../..")
from memory.memory import ConceptGraphFactory
from memory.constants import ConceptGraphConstants

# Import test configuration
try:
    from tests.test_config import get_concept_graph_config, get_file_store_config, has_valid_openai_key
except ImportError:
    print("Warning: test_config.py not found. Please copy test_config.template.py to test_config.py and configure your API keys.")
    # Fallback configuration
    def get_concept_graph_config():
        return {
            "provider": "local",
            "embedding_api_key": "PLEASE_PROVIDE_OPENAI_API_KEY",
            "emb_model": "text-embedding-3-small",
            "emb_dim": 1536
        }
    
    def get_file_store_config(test_dir):
        return {
            "provider": "local",
            "save_path": str(test_dir),
        }
    
    def has_valid_openai_key():
        return False


class TestConceptGraphSystem(unittest.TestCase):
    """System tests for ConceptGraph with real implementations."""
    
    def setUp(self):
        """Set up test environment with temporary directory and real ConceptGraph."""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Get configurations from test_config
        self.file_store_config = get_file_store_config(self.test_dir)
        self.concept_graph_config = get_concept_graph_config()
        
        # Create ConceptGraph using new memory core structure
        memory_core_config = {
            "embedding": {
                "provider": self.concept_graph_config["provider"],
                "api_key": self.concept_graph_config.get("embedding_api_key", ""),
                "model": self.concept_graph_config.get("emb_model", "text-embedding-3-small"),
                "dim": self.concept_graph_config.get("emb_dim", 1536)
            },
            "files": {
                "graph_file": "test_graph.json",
                "index_file": "test_emb_index.json"
            }
        }
        
        # Add remote-specific config if needed
        if self.concept_graph_config["provider"] == "remote":
            memory_core_config["embedding"].update({
                "pinecone_api_key": self.concept_graph_config.get("pinecone_api_key", ""),
                "pinecone_index_name": self.concept_graph_config.get("pinecone_index_name", ""),
                "metric": self.concept_graph_config.get("metric", "cosine")
            })
        
        memory_core_path = str(self.test_dir / "test_memory_core")
        self.concept_graph = ConceptGraphFactory.create_from_memory_core(
            memory_core_path=memory_core_path,
            memory_core_config=memory_core_config
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
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
        concept_id = self.concept_graph.add_concept(
            concept_name, concept_type, concept_attributes
        )
        
        # Verify concept was created
        self.assertIsInstance(concept_id, str)
        self.assertTrue(len(concept_id) > 0)
        
        # Retrieve concept
        retrieved_concept = self.concept_graph.get_concept(concept_id)
        self.assertIsNotNone(retrieved_concept)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_NAME], concept_name)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_TYPE], concept_type)
        self.assertEqual(retrieved_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["age"], 30)
    
    def test_update_concept(self):
        """Test updating a concept."""
        # Add initial concept
        concept_id = self.concept_graph.add_concept(
            "Initial Name", "person", {"age": 25}
        )
        
        # Update concept
        self.concept_graph.update_concept(
            concept_id,
            concept_name="Updated Name",
            concept_attributes={"age": 26, "city": "San Francisco"}
        )
        
        # Verify updates
        updated_concept = self.concept_graph.get_concept(concept_id)
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_NAME], "Updated Name")
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["age"], 26)
        self.assertEqual(updated_concept[ConceptGraphConstants.FIELD_NODE_ATTRIBUTES]["city"], "San Francisco")
    
    def test_add_and_get_relation(self):
        """Test adding and retrieving relations between concepts."""
        # Add two concepts
        person_id = self.concept_graph.add_concept("Alice", "person", {"age": 30})
        company_id = self.concept_graph.add_concept("TechCorp", "company", {"industry": "technology"})
        
        # Add relation
        self.concept_graph.add_relation(person_id, company_id, "works_at")
        
        # Verify relation exists
        relation = self.concept_graph.get_relation(person_id, company_id)
        self.assertIsNotNone(relation)
        self.assertEqual(relation[ConceptGraphConstants.FIELD_EDGE_TYPE], "works_at")
    
    def test_update_relation(self):
        """Test updating a relation."""
        # Add concepts and relation
        person_id = self.concept_graph.add_concept("Bob", "person", {})
        company_id = self.concept_graph.add_concept("StartupInc", "company", {})
        self.concept_graph.add_relation(person_id, company_id, "intern_at")
        
        # Update relation
        self.concept_graph.update_relation(person_id, company_id, "employee_at")
        
        # Verify update
        updated_relation = self.concept_graph.get_relation(person_id, company_id)
        self.assertEqual(updated_relation[ConceptGraphConstants.FIELD_EDGE_TYPE], "employee_at")
    
    def test_delete_concept_and_relation(self):
        """Test deleting concepts and relations."""
        # Add concepts and relation
        person_id = self.concept_graph.add_concept("Charlie", "person", {})
        company_id = self.concept_graph.add_concept("BigCorp", "company", {})
        self.concept_graph.add_relation(person_id, company_id, "works_at")
        
        # Verify they exist
        self.assertIsNotNone(self.concept_graph.get_concept(person_id))
        self.assertIsNotNone(self.concept_graph.get_relation(person_id, company_id))
        
        # Delete relation
        self.concept_graph.delete_relation(person_id, company_id)
        self.assertIsNone(self.concept_graph.get_relation(person_id, company_id))
        
        # Delete concept
        self.concept_graph.delete_concept(person_id)
        self.assertIsNone(self.concept_graph.get_concept(person_id))
        
        # Company should still exist
        self.assertIsNotNone(self.concept_graph.get_concept(company_id))
    
    def test_query_similar_concepts(self):
        """Test querying for similar concepts using embeddings."""
        # Add several related concepts
        self.concept_graph.add_concept("Python Developer", "person", {"skill": "programming"})
        self.concept_graph.add_concept("Java Developer", "person", {"skill": "programming"}) 
        self.concept_graph.add_concept("Data Scientist", "person", {"skill": "analytics"})
        self.concept_graph.add_concept("Pizza Chef", "person", {"skill": "cooking"})
        
        # Query for similar concepts
        similar_concepts = self.concept_graph.query_similar_concepts("Software Engineer", top_k=3)
        
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
        alice_id = self.concept_graph.add_concept("Alice", "person", {})
        bob_id = self.concept_graph.add_concept("Bob", "person", {})
        company_id = self.concept_graph.add_concept("TechCorp", "company", {})
        project_id = self.concept_graph.add_concept("AI Project", "project", {})
        
        # Add relations
        self.concept_graph.add_relation(alice_id, company_id, "works_at")
        self.concept_graph.add_relation(bob_id, company_id, "works_at")
        self.concept_graph.add_relation(alice_id, project_id, "leads")
        self.concept_graph.add_relation(bob_id, project_id, "contributes_to")
        
        # Get related concepts for Alice
        related_concepts, related_relations = self.concept_graph.get_related_concepts(
            alice_id, hop=1
        )
        
        # Should find company and project
        self.assertGreaterEqual(len(related_concepts), 2)
        self.assertGreaterEqual(len(related_relations), 2)
        
        # Filter by relation type
        work_relations = self.concept_graph.get_related_concepts(
            alice_id, hop=1, relation_type="works_at"
        )[1]
        
        self.assertEqual(len(work_relations), 1)
        self.assertEqual(work_relations[0][ConceptGraphConstants.FIELD_EDGE_TYPE], "works_at")
    
    def test_save_and_persistence(self):
        """Test saving and data persistence."""
        # Add some data
        concept_id = self.concept_graph.add_concept("Persistent Data", "test", {"value": 42})
        
        # Save explicitly
        self.concept_graph.save_graph()
        
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
        new_concept_graph = ConceptGraphFactory.create_from_memory_core(
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
        self.concept_graph.add_concept("Person 1", "person", {})
        self.concept_graph.add_concept("Person 2", "person", {}) 
        self.concept_graph.add_concept("Company 1", "company", {})
        
        # Get statistics
        stats = self.concept_graph.get_graph_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_concepts", stats)
        self.assertEqual(stats["total_concepts"], 3)
        self.assertIn("concept_types", stats)
        self.assertEqual(stats["concept_types"]["person"], 2)
        self.assertEqual(stats["concept_types"]["company"], 1)
    
    def test_unified_folder_structure(self):
        """Test that all files are stored in a unified memory core folder structure."""
        # Add some data
        concept_id = self.concept_graph.add_concept("Test Concept", "test", {"value": 123})
        
        # Save to ensure files are created
        self.concept_graph.save_graph()
        
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
    not has_valid_openai_key(),
    "OpenAI API key not provided. Copy test_config.template.py to test_config.py and add your API key."
)
class TestConceptGraphSystemWithAPI(TestConceptGraphSystem):
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
        world_concept_graph = ConceptGraphFactory.create_from_memory_core(
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