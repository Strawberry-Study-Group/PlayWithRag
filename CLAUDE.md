# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered game engine with Retrieval-Augmented Generation (RAG) capabilities. The system creates interactive games using LLMs for narrative generation, Image generation models for scene visualization, and memory cores for persistent game state.

## Core Architecture

### Agent-Based System
- `agent/agent.py` - Central orchestrator coordinating all subsystems
- `agent/action_node.py` - Abstract framework for game actions
- `agent/llm.py` - Multi-provider LLM abstraction (OpenAI, Anthropic)
- `agent/shorterm_mem.py` - Session state management with LRU caching

### Memory Core System
- **Long-term memory**: Knowledge graphs stored in memory cores persist game state
- **Short-term memory**: Session cache for recent interactions  
- **Embedding store**: `memory/emb_store.py` handles both Pinecone (cloud) and FAISS (local) vector storage
- **Graph persistence**: JSON-based storage in `memory/graph_store.py`
- **File storage**: `memory/file_store.py` abstracted file operations with interface-based design

### Multi-Modal Rendering
- `render/render.py` - Image generation supporting OpenAI DALL-E, Replicate, and DeepInfra
- Automatic scene visualization based on game state

## Development Commands

**Run the game interactively:**
```bash
jupyter notebook tests/agent_with_rendering_test.ipynb
```

**Execute unit tests:**
```bash
python -m pytest tests/unit_tests/
# or run individual test files:
python -m pytest tests/unit_tests/test_concept_graph.py
python -m pytest tests/unit_tests/test_emb_store.py
```

**Execute system tests:**
```bash
python -m pytest tests/system_tests/
# or run individual test suites:
python -m pytest tests/system_tests/test_concept_graph_system.py
python -m pytest tests/system_tests/test_memory_core_scenario.py
```

## Configuration

1. Copy `agent/config_local.json` and configure:
   - OpenAI API key for GPT-4o-mini and DALL-E 3
   - Pinecone API key and environment (optional - can use local FAISS)
   - Anthropic API key (optional)

2. The system auto-detects available providers and falls back gracefully

## Key Dependencies

Install required packages:
```bash
pip install openai anthropic pinecone-client numpy faiss-cpu scikit-image matplotlib requests replicate
```

## Memory Core Structure

- **Concepts**: NPCs, locations, items, player state (nodes)
- **Relations**: has, owns, uses, knows, located_at (edges)  
- **Persistence**: Each memory core is a self-contained folder with graph JSON files, embedding indices, and images
- **Search**: Vector embeddings enable semantic retrieval of relevant context

## Memory Core Usage

### Creating a Memory Core
```python
from memory.memory import ConceptGraphFactory

# Define memory core configuration
memory_core_config = {
    "embedding": {
        "provider": "local",  # or "remote" for Pinecone
        "api_key": "your_openai_api_key",
        "model": "text-embedding-3-small",
        "dim": 1536
    },
    "files": {
        "graph_file": "graph.json",
        "index_file": "emb_index.json"
    }
}

# Create memory core instance
memory_core = ConceptGraphFactory.create_from_memory_core(
    memory_core_path="/path/to/your/memory_core_folder",
    memory_core_config=memory_core_config
)
```

## Testing Strategy

- **Unit tests**: Component isolation in `tests/unit_tests/`
- **System tests**: End-to-end testing in `tests/system_tests/`
- **Memory core scenarios**: Full game session testing in `tests/system_tests/test_memory_core_scenario.py`
- **Interactive testing**: Jupyter notebooks for full system validation

## File Organization

- `memory/` - Memory core system (graphs, embeddings, file storage)
- `agent/prompt_templates/` - AI prompt engineering for reasoning and retrieval  
- `tests/unit_tests/` - Component isolation tests
- `tests/system_tests/` - End-to-end integration tests
- Images auto-saved to memory core directories during gameplay
- Each memory core is completely self-contained and portable

## Provider Abstractions

The system supports multiple AI providers with automatic fallback:
- **LLM**: OpenAI GPT models, Anthropic Claude
- **Image**: OpenAI DALL-E 3, Replicate, DeepInfra  
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: Pinecone (cloud) or FAISS (local)

## Current Architectural Issues & Improvement Plan

### Identified Problems

1. **Tight Coupling**: Agent class is a "God Object" managing all subsystems directly
2. **Missing Interfaces**: No abstractions for LLM providers, memory stores, or renderers
3. **State Management**: Mixed file I/O with business logic, inconsistent error handling
4. **Configuration**: Monolithic config with hard-coded paths and plain-text secrets
5. **Testing**: Only integration tests, no unit test isolation possible

### Recommended Design Patterns

#### 1. Dependency Injection & Interfaces
```python
# Define clear contracts
class ILLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str: pass

class IMemoryStore(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[Concept]: pass

class IRenderer(ABC):
    @abstractmethod
    def generate_image(self, description: str) -> str: pass

# Inject dependencies into Agent
class Agent:
    def __init__(self, llm: ILLMProvider, memory: IMemoryStore, renderer: IRenderer):
        self.llm = llm
        self.memory = memory
        self.renderer = renderer
```

#### 2. Repository Pattern for Data Access
```python
class ConceptRepository(ABC):
    @abstractmethod
    def save_concept(self, concept: Concept) -> None: pass
    @abstractmethod
    def find_by_id(self, concept_id: str) -> Optional[Concept]: pass
    @abstractmethod
    def search_similar(self, query: str, limit: int) -> List[Concept]: pass

class MemoryCoreRepository(ABC):
    @abstractmethod
    def load_memory_core(self, memory_core_path: str) -> MemoryCore: pass
    @abstractmethod
    def save_memory_core(self, memory_core: MemoryCore) -> None: pass
```

#### 3. Event-Driven Architecture
```python
class GameEvent:
    def __init__(self, event_type: str, data: dict): pass

class EventBus:
    def publish(self, event: GameEvent) -> None: pass
    def subscribe(self, event_type: str, handler: Callable): pass

# Enable loose coupling between components
event_bus.subscribe("player_action", agent.handle_player_action)
event_bus.subscribe("memory_core_updated", memory.invalidate_cache)
```

#### 4. Configuration Management
```python
# Environment-based configuration
class Config:
    @classmethod
    def from_env(cls, env: str = "development") -> "Config": pass
    
    def get_llm_config(self) -> LLMConfig: pass
    def get_storage_config(self) -> StorageConfig: pass

# Secret management
class SecretManager:
    def get_api_key(self, provider: str) -> str: pass
```

#### 5. Command/Query Separation (CQRS)
```python
# Commands change state
class CreateConceptCommand:
    def __init__(self, concept_data: dict): pass

class CommandHandler:
    def handle(self, command: CreateConceptCommand) -> None: pass

# Queries read state
class FindSimilarConceptsQuery:
    def __init__(self, query: str, limit: int): pass

class QueryHandler:
    def handle(self, query: FindSimilarConceptsQuery) -> List[Concept]: pass
```

### Development Guidelines

#### For New Features
1. **Define interfaces first** before implementing concrete classes
2. **Use dependency injection** - pass dependencies through constructors
3. **Separate concerns** - business logic, data access, presentation
4. **Event-driven communication** between loosely coupled components
5. **Repository pattern** for all data access operations

#### For UI Integration
1. **Separate Agent from I/O** - Agent should not handle console/UI directly
2. **Use events** for Agent → UI communication (game state changes)
3. **Command pattern** for UI → Agent communication (player actions)
4. **State management** with immutable updates for UI reactivity

#### For Memory Core Editor
1. **Transaction support** - atomic operations for memory core modifications
2. **Validation layer** - schema validation before persisting changes
3. **Undo/Redo** - command pattern with reversible operations
4. **Real-time sync** - event-driven updates between editor and game engine

#### For Testing
1. **Mock all external dependencies** (LLM providers, file system, APIs)
2. **Test each component in isolation** using dependency injection
3. **Use builders** for creating test data and configurations
4. **Property-based testing** for graph operations and memory consistency

#### Error Handling Strategy
1. **Domain-specific exceptions** instead of generic errors
2. **Retry policies** as configuration, not hard-coded
3. **Circuit breaker pattern** for external service calls
4. **Graceful degradation** when AI providers are unavailable

### Migration Path
1. **Phase 1**: Extract interfaces and implement dependency injection
2. **Phase 2**: Implement repository pattern for data access
3. **Phase 3**: Add event-driven architecture for component communication
4. **Phase 4**: Separate UI concerns and add command/query handlers
5. **Phase 5**: Add transaction support and validation for graph editing