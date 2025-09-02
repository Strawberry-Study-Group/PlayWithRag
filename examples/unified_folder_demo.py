#!/usr/bin/env python3
"""
Demo of the unified folder structure for ConceptGraph.

This example shows how all data (graph, embeddings, images) are stored
in a single folder for easy sharing and organization.
"""

import tempfile
from pathlib import Path
import sys
import os

# Add the parent directory to path so we can import from concept_graph
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_graph.concept_graph import ConceptGraphFactory


def demo_unified_folder_structure():
    """Demonstrate the unified folder structure."""
    print("=" * 60)
    print("UNIFIED FOLDER STRUCTURE DEMO")
    print("=" * 60)
    
    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Demo directory: {temp_dir}")
    
    # Demo 1: Using create_world convenience method
    print("\n🌍 Creating world using convenience method...")
    try:
        # This would work with real API key - using dummy for structure demo
        concept_graph = ConceptGraphFactory.create_world(
            base_path=str(temp_dir),
            world_name="my_game_world",
            openai_api_key="dummy-key-for-demo",  # Replace with real key
            use_pinecone=False  # Use local FAISS storage
        )
        
        world_dir = temp_dir / "my_game_world"
        print(f"✅ World created at: {world_dir}")
        print(f"   Directory exists: {world_dir.exists()}")
        
    except Exception as e:
        print(f"⚠️  Note: This demo uses a dummy API key, so embedding operations will fail.")
        print(f"   But the folder structure is still created: {e}")
    
    # Demo 2: Manual configuration
    print("\n🔧 Creating world using manual configuration...")
    
    concept_graph_config = {
        "provider": "local",  # Use local FAISS
        "embedding_api_key": "dummy-key-for-demo",
        "emb_model": "text-embedding-3-small",
        "emb_dim": 1536
    }
    
    file_store_config = {
        "provider": "local",
        "save_path": str(temp_dir)
    }
    
    # Create another world
    try:
        concept_graph2 = ConceptGraphFactory.create_from_config(
            concept_graph_config=concept_graph_config,
            save_file_config=file_store_config,
            world_name="another_world",  # Different world name
            graph_file_name="game_graph.json",
            index_file_name="game_embeddings.json"
        )
        
        world2_dir = temp_dir / "another_world"
        print(f"✅ Second world created at: {world2_dir}")
        print(f"   Directory exists: {world2_dir.exists()}")
        
    except Exception as e:
        print(f"⚠️  Folder structure created despite API error: {e}")
    
    # Show the directory structure
    print("\n📂 Directory structure created:")
    print(f"{temp_dir}/")
    
    for world_dir in temp_dir.iterdir():
        if world_dir.is_dir():
            print(f"├── {world_dir.name}/")
            for file_path in world_dir.iterdir():
                print(f"│   ├── {file_path.name}")
    
    # Demo 3: Show sharing benefits
    print(f"\n📦 SHARING BENEFITS:")
    print(f"   • Each world is completely self-contained")
    print(f"   • Copy entire folder to share: cp -r {temp_dir}/my_game_world /destination/")
    print(f"   • All data together: graph.json, emb_index.json, images/")
    print(f"   • No scattered files across different directories")
    
    # Demo 4: Multiple worlds
    print(f"\n🌐 MULTIPLE WORLDS:")
    print(f"   • Each world_name creates separate folder")
    print(f"   • No conflicts between different games/scenarios")
    print(f"   • Easy to organize: worlds/rpg_game/, worlds/story_mode/, etc.")
    
    print(f"\n🧹 Cleaning up demo directory: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n✅ Demo complete!")
    
    return True


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
# Example 1: Simple world creation
from concept_graph.concept_graph import ConceptGraphFactory

concept_graph = ConceptGraphFactory.create_world(
    base_path="/data/games",
    world_name="fantasy_rpg", 
    openai_api_key="your-api-key",
    use_pinecone=False
)

# All data stored in: /data/games/fantasy_rpg/
# - /data/games/fantasy_rpg/graph.json
# - /data/games/fantasy_rpg/emb_index.json  
# - /data/games/fantasy_rpg/images/

# Example 2: Multiple worlds
world1 = ConceptGraphFactory.create_world(
    base_path="/data", world_name="world1", openai_api_key="key"
)
world2 = ConceptGraphFactory.create_world(
    base_path="/data", world_name="world2", openai_api_key="key"  
)

# Creates:
# /data/world1/ - completely independent
# /data/world2/ - completely independent

# Example 3: Custom configuration
config = {
    "provider": "local",
    "embedding_api_key": "your-key",
    "emb_model": "text-embedding-3-small", 
    "emb_dim": 1536
}
file_config = {"provider": "local", "save_path": "/data"}

cg = ConceptGraphFactory.create_from_config(
    concept_graph_config=config,
    save_file_config=file_config, 
    world_name="custom_world"
)
""")


if __name__ == "__main__":
    try:
        demo_unified_folder_structure()
        show_usage_examples()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()