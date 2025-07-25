# Concept Graph System Tests

This directory contains comprehensive system tests for the concept graph functionality with real API integration.

## Overview

These tests verify the complete concept graph system using actual API calls to OpenAI for embeddings and optionally Pinecone for remote vector storage. They simulate real-world usage scenarios and test the system end-to-end.

## Setup

### 1. API Key Configuration

Before running the tests, configure your API keys using one of these methods:

#### Option A: Environment Variables (Recommended)
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export PINECONE_API_KEY="your_pinecone_api_key_here"  # Optional, for remote tests
export PINECONE_INDEX_NAME="concept-graph-test"       # Optional, for remote tests
export ANTHROPIC_API_KEY="your_anthropic_api_key_here" # Optional, future use
```

#### Option B: Edit Configuration File
Edit `config.py` and replace the placeholder values:
```python
OPENAI_API_KEY = "your_actual_openai_api_key_here"
PINECONE_API_KEY = "your_actual_pinecone_api_key_here"  # Optional
```

### 2. Required API Keys

- **OpenAI API Key**: Required for all tests (embeddings)
- **Pinecone API Key**: Optional, only needed for remote storage tests
- **Anthropic API Key**: Optional, reserved for future functionality

### 3. Dependencies

Make sure you have the required packages installed:
```bash
pip install openai anthropic pinecone-client numpy faiss-cpu pytest
```

## Test Structure

### Test Files

1. **`test_basic_operations.py`**
   - Basic CRUD operations with real API calls
   - Concept creation, updates, deletions
   - Similarity search with real embeddings
   - Local and remote storage testing

2. **`test_game_world_scenario.py`**
   - Complete game world simulation based on the notebook example
   - NPCs, events, locations with complex relationships
   - Multi-hop relationship traversal
   - Character, event, and location similarity searches

3. **`test_advanced_operations.py`**
   - Advanced operations and edge cases
   - Multi-hop relationship traversal
   - Complex relationship networks
   - Performance testing with larger datasets
   - Concurrent operations simulation

4. **`test_integration_scenarios.py`**
   - Real-world integration scenarios
   - Game session simulation
   - Knowledge base evolution
   - Content recommendation system

### Test Categories

- **Basic Operations**: Core functionality with real APIs
- **Game World**: Complete RPG world simulation
- **Advanced**: Complex operations and edge cases  
- **Integration**: Real-world usage patterns

## Running the Tests

### All System Tests
```bash
cd /home/eddy/PlayWithRag
python -m pytest tests/system_tests/ -v
```

### Specific Test Files
```bash
# Basic operations only
python -m pytest tests/system_tests/test_basic_operations.py -v

# Game world scenario
python -m pytest tests/system_tests/test_game_world_scenario.py -v

# Advanced operations
python -m pytest tests/system_tests/test_advanced_operations.py -v

# Integration scenarios
python -m pytest tests/system_tests/test_integration_scenarios.py -v
```

### Test Markers

Some tests have special markers:

```bash
# Run only local storage tests (no Pinecone required)
python -m pytest tests/system_tests/ -m "not slow" -v

# Run performance tests
python -m pytest tests/system_tests/ -m "performance" -v

# Skip slow tests
python -m pytest tests/system_tests/ -m "not slow" -v
```

## Test Scenarios

### 1. Basic Operations (`test_basic_operations.py`)

- **NPC Creation**: Creates game characters with embeddings
- **Relationship Management**: Tests relations between concepts
- **Similarity Search**: Tests semantic search with real embeddings
- **Updates and Cleanup**: Tests concept updates and deletions
- **Remote Storage**: Tests Pinecone integration (requires API key)

### 2. Game World Scenario (`test_game_world_scenario.py`)

Based on the original notebook example:
- **Character Network**: 11 NPCs with complex relationships
- **Event System**: 9 interconnected game events
- **Location System**: 9 locations connected to events
- **Multi-hop Traversal**: Tests relationship chains
- **Semantic Queries**: Cross-type similarity searches

### 3. Advanced Operations (`test_advanced_operations.py`)

- **Multi-hop Traversal**: Tests 1, 2, 3+ hop relationship chains
- **Complex Networks**: Social networks with multiple relationship types
- **Relationship Filtering**: Filter by type and concept category
- **Large Scale Search**: Tests with 50+ concepts
- **Performance Testing**: Timed operations with benchmarks
- **Edge Cases**: Special characters, long descriptions, error handling

### 4. Integration Scenarios (`test_integration_scenarios.py`)

- **Game Session**: Complete gameplay simulation with world evolution
- **Knowledge Base**: Academic research scenario with discovery simulation
- **Content Recommendation**: Recommendation system using concept relationships

## Expected Behavior

### API Call Patterns

These tests make real API calls:
- **OpenAI Embeddings**: Every concept creation/update generates embedding calls
- **Similarity Searches**: Each query triggers embedding generation and similarity computation
- **Vector Storage**: Local FAISS or remote Pinecone for embedding storage

### Performance Expectations

The tests include reasonable performance expectations:
- **50 concepts**: Should create in under 60 seconds
- **25 relationships**: Should create in under 30 seconds  
- **10 similarity searches**: Should complete in under 30 seconds

### Data Cleanup

All tests clean up after themselves:
- **Graph Reset**: Each test starts with an empty graph
- **File Cleanup**: Temporary files are removed
- **Vector Cleanup**: Embeddings are cleared between tests

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   AssertionError: API keys not configured
   ```
   - Set your OpenAI API key in environment variables or config.py

2. **Pinecone Errors** (Remote tests only)
   ```
   Skip: API keys not configured for remote testing
   ```
   - Set PINECONE_API_KEY and PINECONE_INDEX_NAME for remote tests

3. **Rate Limiting**
   ```
   OpenAI API rate limit exceeded
   ```
   - Tests include delays, but you may need to increase them for lower-tier API accounts

4. **Timeout Errors**
   ```
   Timeout waiting for API response
   ```
   - Check your internet connection and API service status

### Debug Mode

Run tests with more verbose output:
```bash
python -m pytest tests/system_tests/ -v -s --tb=long
```

### Skip Slow Tests

If you want faster feedback:
```bash
python -m pytest tests/system_tests/ -m "not slow and not performance" -v
```

## API Usage and Costs

These tests make real API calls and will incur costs:

### OpenAI Costs (Estimated)
- **Embeddings**: ~$0.0001 per 1K tokens
- **Typical test run**: 100-500 embedding calls
- **Estimated cost**: $0.01-$0.05 per full test run

### Pinecone Costs (Remote tests only)
- **Starter plan**: Free tier available
- **Test usage**: Minimal, within free tier limits

## Contributing

When adding new system tests:

1. **Use real APIs**: Don't mock API calls in system tests
2. **Clean up**: Always clean up test data
3. **Use fixtures**: Share common setup through pytest fixtures
4. **Add markers**: Mark slow or expensive tests appropriately
5. **Document scenarios**: Explain what real-world scenario the test simulates

## Integration with CI/CD

For CI/CD integration:

```yaml
# Example GitHub Actions configuration
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
  
script:
  # Run only fast system tests in CI
  - python -m pytest tests/system_tests/ -m "not slow and not performance" -v
```