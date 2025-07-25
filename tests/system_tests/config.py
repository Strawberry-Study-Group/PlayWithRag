"""Configuration for system tests with real API integration."""

import os
import uuid
from typing import Dict, Any

# API Configuration - Users should set these environment variables or update values directly
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY_HERE")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "task-graph-index")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

# Test configuration for local storage (FAISS)
LOCAL_CONFIG = {
    "concept_graph_config": {
        "provider": "local",
        "embedding_api_key": OPENAI_API_KEY,
        "emb_model": "text-embedding-3-small",
        "emb_dim": 1536
    },
    "file_store_config": {
        "provider": "local",
        "save_path": "system_test_data"
    }
}

# Test configuration for remote storage (Pinecone)
REMOTE_CONFIG = {
    "concept_graph_config": {
        "provider": "remote",
        "embedding_api_key": OPENAI_API_KEY,
        "pinecone_api_key": PINECONE_API_KEY,
        "pinecone_index_name": PINECONE_INDEX_NAME,
        "emb_model": "text-embedding-3-small",
        "emb_dim": 1536,
        "metric": "cosine"
    },
    "file_store_config": {
        "provider": "local",
        "save_path": "system_test_data"
    }
}

def get_test_config(use_remote: bool = False, unique_index: bool = True) -> Dict[str, Any]:
    """Get test configuration for local or remote embedding storage."""
    config = REMOTE_CONFIG.copy() if use_remote else LOCAL_CONFIG.copy()
    
    if use_remote and unique_index:
        # Create a unique test index name that's clearly for testing
        import time
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        config["concept_graph_config"] = config["concept_graph_config"].copy()
        config["concept_graph_config"]["pinecone_index_name"] = f"pytest-{timestamp}-{unique_id}"
    
    return config

def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are configured."""
    return {
        "openai": OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE" and OPENAI_API_KEY is not None,
        "pinecone": PINECONE_API_KEY != "YOUR_PINECONE_API_KEY_HERE" and PINECONE_API_KEY is not None,
        "anthropic": ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY_HERE" and ANTHROPIC_API_KEY is not None
    }

def check_test_readiness(use_remote: bool = False) -> bool:
    """Check if system tests can run with current configuration."""
    validation = validate_api_keys()
    
    if not validation["openai"]:
        print("❌ OpenAI API key not configured")
        return False
    
    if use_remote and not validation["pinecone"]:
        print("❌ Pinecone API key not configured for remote tests")
        return False
    
    print("✅ API keys configured correctly")
    return True
