"""
Test configuration template file.

To use this:
1. Copy this file to test_config.py
2. Fill in your actual API keys in test_config.py
3. The test_config.py file will be ignored by git

Usage:
    cp tests/test_config.template.py tests/test_config.py
    # Edit tests/test_config.py with your API keys
"""

# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"
OPENAI_EMB_MODEL = "text-embedding-3-small"
OPENAI_EMB_DIM = 1536

# Pinecone API Configuration (optional, for remote vector store tests)
PINECONE_API_KEY = "your-pinecone-api-key-here"
PINECONE_INDEX_NAME = "test-concept-graph"
PINECONE_ENVIRONMENT = "us-west-2"

# Anthropic API Configuration (optional)
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"

# Test Configuration
TEST_DATA_DIR = "test_data"
USE_LOCAL_STORAGE = True  # Set to False to use Pinecone instead of FAISS

def get_concept_graph_config():
    """Get configuration for ConceptGraph tests."""
    if USE_LOCAL_STORAGE:
        return {
            "provider": "local",
            "embedding_api_key": OPENAI_API_KEY,
            "emb_model": OPENAI_EMB_MODEL,
            "emb_dim": OPENAI_EMB_DIM
        }
    else:
        return {
            "provider": "remote",
            "embedding_api_key": OPENAI_API_KEY,
            "pinecone_api_key": PINECONE_API_KEY,
            "pinecone_index_name": PINECONE_INDEX_NAME,
            "emb_model": OPENAI_EMB_MODEL,
            "emb_dim": OPENAI_EMB_DIM
        }

def get_file_store_config(test_dir):
    """Get configuration for file store tests."""
    return {
        "provider": "local",
        "save_path": str(test_dir),
    }

def has_valid_openai_key():
    """Check if a valid OpenAI API key is configured."""
    return (OPENAI_API_KEY and 
            OPENAI_API_KEY != "your-openai-api-key-here" and 
            len(OPENAI_API_KEY) > 10)

def has_valid_pinecone_key():
    """Check if a valid Pinecone API key is configured."""
    return (PINECONE_API_KEY and 
            PINECONE_API_KEY != "your-pinecone-api-key-here" and 
            len(PINECONE_API_KEY) > 10)

def has_valid_anthropic_key():
    """Check if a valid Anthropic API key is configured."""
    return (ANTHROPIC_API_KEY and 
            ANTHROPIC_API_KEY != "your-anthropic-api-key-here" and 
            len(ANTHROPIC_API_KEY) > 10)