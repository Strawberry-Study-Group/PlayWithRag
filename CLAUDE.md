# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I am working on refactoring the code base. This is an AI-powered game engine with Retrieval-Augmented Generation (RAG) capabilities. The system creates interactive games using LLMs for narrative generation, Image genration models for scene visualization, and knowledge graphs for persistent world state.

## Core Architecture

### Agent-Based System
- `agent/agent.py` - Central orchestrator coordinating all subsystems
- `agent/action_node.py` - Abstract framework for game actions
- `agent/llm.py` - Multi-provider LLM abstraction (OpenAI, Anthropic)
- `agent/shorterm_mem.py` - Session state management with LRU caching

### RAG Memory System
- **Long-term memory**: Knowledge graph in `concept_graph/` persists world state
- **Short-term memory**: Session cache for recent interactions  
- **Embedding store**: `concept_graph/emb_store.py` handles both Pinecone (cloud) and FAISS (local) vector storage
- **Graph persistence**: JSON-based storage in `concept_graph/graph_store.py`
- **File storage**: `concept_graph/file_store.py` abstracted file operations with interface-based design

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
python -m pytest tests/
# or run individual test files:
python tests/concept_graph_test.py
python tests/emb_store_test.py
```

**Test specific components:**
```bash
# Test rendering system
python tests/render_test.py

# Test file operations  
python tests/file_store_test.py
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

## Knowledge Graph Structure

- **Concepts**: NPCs, locations, items, player state (nodes)
- **Relations**: has, owns, uses, knows, located_at (edges)  
- **Persistence**: Incremental updates to local JSON files
- **Search**: Vector embeddings enable semantic retrieval of relevant context

## Testing Strategy

- **Interactive testing**: Jupyter notebooks in `tests/` for full system validation
- **Unit tests**: Traditional Python test files for component isolation
- **Test worlds**: Sample game data in `tests/concept_store/` and `tests/living_in_the_shadow/`
- **Integration**: `tests/agent_with_rendering_test.ipynb` tests the complete pipeline

## File Organization

- `agent/prompt_templates/` - AI prompt engineering for reasoning and retrieval
- `tests/concept_store/` - Game world persistence layer
- Images auto-saved to appropriate test directories during gameplay
- All game state persists locally unless cloud storage configured

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

class GameWorldRepository(ABC):
    @abstractmethod
    def load_world(self, world_id: str) -> GameWorld: pass
    @abstractmethod
    def save_world(self, world: GameWorld) -> None: pass
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
event_bus.subscribe("world_updated", memory.invalidate_cache)
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

#### For Graph Editor
1. **Transaction support** - atomic operations for graph modifications
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