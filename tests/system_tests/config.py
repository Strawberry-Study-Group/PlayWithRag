"""Configuration for system tests with real API integration."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from test_config import OPENAI_API_KEY, has_valid_openai_key
from test_config import PINECONE_API_KEY, has_valid_pinecone_key  
from test_config import ANTHROPIC_API_KEY, has_valid_anthropic_key

def check_test_readiness(use_remote: bool = False) -> bool:
    """Check if system tests can run with current configuration."""
    
    if not has_valid_openai_key():
        print("❌ OpenAI API key not configured")
        return False
    
    if use_remote and not has_valid_pinecone_key():
        print("❌ Pinecone API key not configured for remote tests")
        return False
    
    print("✅ API keys configured correctly")
    return True

def validate_api_keys() -> dict:
    """Validate that required API keys are configured."""
    return {
        "openai": has_valid_openai_key(),
        "pinecone": has_valid_pinecone_key(), 
        "anthropic": has_valid_anthropic_key()
    }