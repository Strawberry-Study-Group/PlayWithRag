# PlayWithRag Test Suite

This directory contains a comprehensive test suite for the PlayWithRag system, organized into three categories for different testing purposes.

## 📁 Test Organization

```
tests/
├── unit_tests/          # Fast, isolated tests with mocks
├── system_tests/        # Integration tests with real APIs  
├── manual_tests/        # Interactive Jupyter notebooks
├── test_config.py       # API keys (git ignored)
├── test_config.template.py  # Template for configuration
└── README.md           # This file
```

## 🧪 Test Categories

### **Unit Tests** (`tests/unit_tests/`)
**Purpose:** Fast, isolated testing with mocked dependencies
- ❌ **No API keys required**
- ⚡ **Fast execution** (< 1 second per test)
- 🔒 **Isolated** - each test runs independently
- 🎭 **Mocked** - external dependencies are mocked

**Files:**
- `test_concept_graph.py` - ConceptGraph service tests
- `test_emb_store.py` - Embedding service tests  
- `test_file_store.py` - File storage tests
- `test_graph_store.py` - Graph storage tests
- `test_concept_operations.py` - Concept operations tests
- `legacy_*.py` - Legacy test files (renamed)

**Run with:**
```bash
python -m pytest tests/unit_tests/ -v
```

### **System Tests** (`tests/system_tests/`)
**Purpose:** Integration testing with real implementations and APIs
- ✅ **API keys required** (OpenAI, optionally Pinecone)
- 🌐 **Real API calls** - tests actual integrations
- 💾 **Real storage** - creates temporary directories
- 🧹 **Auto-cleanup** - temporary files removed after tests

**Files:**
- `test_concept_graph_system.py` - End-to-end ConceptGraph testing
- `test_agent_system.py` - Agent system integration
- `test_local_emb_store.py` - Local embedding store testing
- `test_basic_operations.py` - Basic system operations
- `test_advanced_operations.py` - Advanced system features
- `test_game_world_scenario.py` - Game world scenarios
- `test_integration_scenarios.py` - Integration scenarios

**Run with:**
```bash
# Single test file
PYTHONPATH=/home/eddy/projects/PlayWithRag python tests/system_tests/test_concept_graph_system.py

# All system tests  
python -m pytest tests/system_tests/ -v
```

### **Manual Tests** (`tests/manual_tests/`)
**Purpose:** Interactive exploration and visual validation
- 🎮 **Interactive** - requires manual interaction
- 👁️ **Visual** - outputs images, visualizations
- 📚 **Educational** - demonstrates features
- 🛠️ **Development** - useful for debugging

**Files:**
- `agent_with_rendering_test.ipynb` - Agent + image generation
- `concept_graph_system_test.ipynb` - Interactive concept graph exploration
- `render_test.ipynb` - Image rendering tests

**Run with:**
```bash
jupyter notebook tests/manual_tests/
```

## ⚙️ Setup Instructions

### 1. Configure API Keys

Copy the template and add your keys:

```bash
cp tests/test_config.template.py tests/test_config.py
```

Edit `tests/test_config.py`:
```python
# Required for system tests
OPENAI_API_KEY = "sk-your-actual-openai-key-here"

# Optional - for remote vector storage
PINECONE_API_KEY = "your-pinecone-key-here"
PINECONE_INDEX_NAME = "test-index"

# Optional - for Anthropic LLM features  
ANTHROPIC_API_KEY = "your-anthropic-key-here"
```

⚠️ **Security Note:** `test_config.py` is git-ignored to prevent API key leaks.

### 2. Install Dependencies

```bash
pip install pytest jupyter faiss-cpu openai anthropic pinecone-client numpy scikit-image matplotlib requests replicate
```

## 🚀 Quick Start Commands

```bash
# Fast development testing (no API keys needed)
python -m pytest tests/unit_tests/ -v

# Full integration testing (requires OpenAI API key)
python -m pytest tests/system_tests/ -v

# Interactive exploration
jupyter notebook tests/manual_tests/

# Run specific test
python -m pytest tests/unit_tests/test_concept_graph.py::TestConceptGraphService::test_add_concept -v

# Run with unified folder demo
python examples/unified_folder_demo.py
```

## 📊 Test Matrix

| Test Type | Speed | API Keys | Use Case |
|-----------|-------|----------|----------|
| **Unit** | ⚡ Fast | ❌ None | Development, CI/CD |
| **System** | 🐌 Slow | ✅ Required | Integration validation |
| **Manual** | 🎮 Interactive | ✅ Required | Exploration, debugging |

## 🛠️ Development Workflow

**During Development:**
```bash
# Quick feedback loop
python -m pytest tests/unit_tests/ -v
```

**Before Commit:**
```bash  
# Full validation
python -m pytest tests/unit_tests/ tests/system_tests/ -v
```

**For Demos/Exploration:**
```bash
# Interactive notebooks
jupyter notebook tests/manual_tests/
```

## 🔧 Configuration Options

### Storage Backend
```python
# In test_config.py
USE_LOCAL_STORAGE = True   # FAISS (local)
USE_LOCAL_STORAGE = False  # Pinecone (remote)
```

### Test Data Location
```python
# In test_config.py  
TEST_DATA_DIR = "custom_test_data"
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'test_config'` | `cp test_config.template.py test_config.py` |
| `"OpenAI API key not provided"` | Add real API key to `test_config.py` |
| Tests are slow | Use unit tests for development, system tests for validation |
| Import errors in moved files | Check `sys.path.append()` paths are correct |
| Temporary files not cleaned | Check `temp_*` directories, tests should auto-cleanup |

## 📈 Test Coverage

- ✅ **Unit Tests:** ~95% code coverage
- ✅ **System Tests:** End-to-end workflows  
- ✅ **Manual Tests:** Visual validation
- ✅ **Legacy Support:** Backward compatibility

The test suite ensures reliable development and deployment of the PlayWithRag system across all components.