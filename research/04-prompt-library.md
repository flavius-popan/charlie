# Prompt Library Structure

**Source**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/`

## Overview

The Graphiti-core prompt library uses a **versioned, protocol-based architecture** that separates prompt definitions from their runtime wrappers. This allows multiple prompt versions to coexist and provides a clean abstraction for LLM calls.

## Architecture

### Core Components

#### 1. Message Format (`prompts/models.py`)

```python
class Message(BaseModel):
    role: str          # "system", "user", or "assistant"
    content: str       # The actual prompt text
```

**Usage**: All prompts return `list[Message]` representing a conversation structure.

#### 2. PromptFunction Type (`prompts/models.py`)

```python
PromptFunction = Callable[[dict[str, Any]], list[Message]]
```

**Contract**: Takes a context dictionary, returns a list of Messages. Context keys vary by prompt type.

#### 3. PromptVersion Protocol (`prompts/models.py`)

```python
class PromptVersion(Protocol):
    def __call__(self, context: dict[str, Any]) -> list[Message]: ...
```

**Purpose**: Type hint for version-specific prompt functions.

---

## Versioning System

### Structure (`prompts/lib.py`)

The library uses a **three-layer wrapper system**:

```
PromptLibraryWrapper
  └─> PromptTypeWrapper (per prompt type: extract_nodes, dedupe_edges, etc.)
       └─> VersionWrapper (per version: extract_message, extract_json, etc.)
            └─> PromptFunction (actual prompt generator)
```

### Layer Responsibilities

#### 1. VersionWrapper
```python
class VersionWrapper:
    def __init__(self, func: PromptFunction):
        self.func = func

    def __call__(self, context: dict[str, Any]) -> list[Message]:
        messages = self.func(context)
        for message in messages:
            message.content += DO_NOT_ESCAPE_UNICODE if message.role == 'system' else ''
        return messages
```

**Responsibilities**:
- Wraps individual prompt functions
- Appends unicode handling instruction to system messages
- Preserves non-ASCII characters (Korean, Japanese, Chinese, etc.)

#### 2. PromptTypeWrapper
```python
class PromptTypeWrapper:
    def __init__(self, versions: dict[str, PromptFunction]):
        for version, func in versions.items():
            setattr(self, version, VersionWrapper(func))
```

**Responsibilities**:
- Groups all versions of a specific prompt type
- Creates a VersionWrapper for each version
- Provides attribute-based access (e.g., `prompt_type.extract_message`)

#### 3. PromptLibraryWrapper
```python
class PromptLibraryWrapper:
    def __init__(self, library: PromptLibraryImpl):
        for prompt_type, versions in library.items():
            setattr(self, prompt_type, PromptTypeWrapper(versions))
```

**Responsibilities**:
- Top-level container for all prompt types
- Creates PromptTypeWrapper for each prompt type
- Provides access like: `prompt_library.extract_nodes.extract_message(context)`

---

## Prompt Types and Versions

### Registered Prompts (`prompts/lib.py`)

```python
PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    'extract_nodes': extract_nodes_versions,      # Entity extraction
    'dedupe_nodes': dedupe_nodes_versions,        # Entity deduplication
    'extract_edges': extract_edges_versions,      # Fact/relationship extraction
    'dedupe_edges': dedupe_edges_versions,        # Fact deduplication
    'invalidate_edges': invalidate_edges_versions,# Fact contradiction detection
    'extract_edge_dates': extract_edge_dates_versions,  # Temporal extraction
    'summarize_nodes': summarize_nodes_versions,  # Entity summarization
    'eval': eval_versions,                        # Evaluation prompts
}
```

### Version Dictionary Pattern

Each prompt type exports a `versions` dictionary:

```python
# Example from prompts/extract_nodes.py
versions: Versions = {
    'extract_message': extract_message,    # For conversational messages
    'extract_json': extract_json,          # For JSON documents
    'extract_text': extract_text,          # For plain text
    'reflexion': reflexion,                # For missed entity detection
    'extract_summary': extract_summary,    # For entity summaries
    'classify_nodes': classify_nodes,      # For entity classification
    'extract_attributes': extract_attributes,  # For entity attributes
}
```

---

## Context Structure Patterns

### Common Context Keys (Across All Prompts)

```python
{
    'episode_content': str,           # Current text/message/JSON being processed
    'previous_episodes': list[dict],  # Historical context messages
    'custom_prompt': str,             # User-defined prompt additions
}
```

### Extract Nodes Context (`prompts/extract_nodes.py`)

```python
{
    'entity_types': str,              # JSON list of entity type definitions
    'episode_content': str,           # The text to extract entities from
    'previous_episodes': list[dict],  # Prior conversation messages
    'source_description': str,        # Description of data source (for JSON)
    'custom_prompt': str,             # Additional user instructions
    'extracted_entities': list,       # For reflexion: already extracted
    'node': dict,                     # For extract_summary/attributes: the entity
}
```

### Extract Edges Context (`prompts/extract_edges.py`)

```python
{
    'edge_types': str,                # JSON list of fact type definitions
    'episode_content': str,           # Current message
    'previous_episodes': list[dict],  # Historical context
    'nodes': list[dict],              # Extracted entities with IDs
    'reference_time': str,            # ISO 8601 timestamp for temporal resolution
    'custom_prompt': str,             # User additions
    'extracted_facts': list,          # For reflexion: already extracted facts
    'fact': dict,                     # For extract_attributes: the edge
}
```

### Dedupe Nodes Context (`prompts/dedupe_nodes.py`)

```python
{
    'extracted_node': dict,           # New entity to check
    'existing_nodes': list[dict],     # Known entities to compare against
    'entity_type_description': str,   # Type description for context
    'episode_content': str,           # Current message
    'previous_episodes': list[dict],  # Historical context
    'extracted_nodes': list[dict],    # For batch: all new entities
    'nodes': list[dict],              # For node_list: entities to dedupe
}
```

### Dedupe Edges Context (`prompts/dedupe_edges.py`)

```python
{
    'extracted_edges': dict,          # New edge to check
    'related_edges': list[dict],      # Similar edges from graph
    'edge_types': str,                # Fact type definitions
    'existing_edges': list[dict],     # For resolve_edge: known edges
    'edge_invalidation_candidates': list[dict],  # Edges that might be contradicted
    'new_edge': dict,                 # For resolve_edge: the new edge
    'edges': list[dict],              # For edge_list: edges to dedupe
}
```

---

## Helper Functions

### `to_prompt_json` (`prompts/prompt_helpers.py`)

```python
def to_prompt_json(data: Any, ensure_ascii: bool = False, indent: int | None = None) -> str:
    """
    Serialize data to JSON for use in prompts.

    Args:
        data: The data to serialize
        ensure_ascii: If True, escape non-ASCII characters. Default: False
        indent: Number of spaces for indentation. Default: None (minified)

    Returns:
        JSON string representation of the data
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
```

**Default behavior**: Preserves non-ASCII characters (e.g., Korean, Japanese) for better LLM understanding.

### `DO_NOT_ESCAPE_UNICODE` Constant

```python
DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'
```

**Usage**: Automatically appended to system messages by `VersionWrapper` to prevent LLMs from escaping unicode in responses.

---

## Integration Points for DSPy

### Current Pattern (String-Based)

```python
# Graphiti-core currently uses string-based LLM calls
messages = prompt_library.extract_nodes.extract_message(context)
# messages -> [Message(role='system', content='...'), Message(role='user', content='...')]

# Then passed to LLM client (e.g., OpenAI, Anthropic)
response = llm_client.generate(messages)
```

### DSPy Integration Strategy

**Challenges**:
1. DSPy Signatures expect typed fields, not free-form message lists
2. Context dictionaries have variable structures
3. Multiple versions per prompt type

**Recommended Approach**:

#### Option 1: Message-Based Signatures
```python
# Convert Message list to single concatenated prompt
class ExtractNodesSignature(dspy.Signature):
    """Extract entities from text"""
    prompt = dspy.InputField(desc="Full prompt with context")
    extracted_entities: ExtractedEntities = dspy.OutputField()

# Build prompt from messages
messages = prompt_library.extract_nodes.extract_message(context)
prompt_text = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
```

#### Option 2: Context-Based Signatures
```python
# Map context keys to DSPy input fields
class ExtractNodesSignature(dspy.Signature):
    """Extract entities from text"""
    entity_types = dspy.InputField(desc="Entity type definitions")
    episode_content = dspy.InputField(desc="Text to extract from")
    previous_episodes = dspy.InputField(desc="Historical context")
    extracted_entities: ExtractedEntities = dspy.OutputField()

# Pass context directly as kwargs
result = dspy.Predict(ExtractNodesSignature)(**context)
```

#### Option 3: Hybrid (Recommended for Charlie)
```python
# Use graphiti-core prompts for instructions, DSPy for structured output
class ExtractNodesSignature(dspy.Signature):
    """Extract entities from conversational messages"""
    context = dspy.InputField(desc="Structured context dict")
    extracted_entities: ExtractedEntities = dspy.OutputField()

# Attach graphiti-core instructions as system message
messages = prompt_library.extract_nodes.extract_message(context)
system_prompt = messages[0].content
dspy.settings.configure(lm=OutlinesLM(system_prompt=system_prompt))
```

---

## Key Takeaways for Custom Pipeline

### Reuse Decisions

**Can Reuse**:
- Prompt text and instructions (excellent extraction logic)
- Context structure patterns (standardized keys)
- Helper functions (`to_prompt_json`, unicode handling)
- Response model schemas (see `05-pydantic-models.md`)

**Must Replace**:
- `PromptLibraryWrapper` system (not compatible with DSPy)
- Message-based call pattern (DSPy uses Signatures)
- Version selection logic (DSPy doesn't support runtime version switching)

**Should Adapt**:
- Extract core prompt logic into DSPy Signatures
- Convert context dictionaries to Signature input fields
- Map response Pydantic models to Signature output fields
- Preserve prompt instructions as Signature docstrings or system messages

### Example: Minimal DSPy Adaptation

```python
# Original graphiti-core
context = {
    'entity_types': json.dumps(entity_types),
    'episode_content': "Alice: I went to Paris",
    'previous_episodes': [],
    'custom_prompt': ""
}
messages = prompt_library.extract_nodes.extract_message(context)

# DSPy equivalent using graphiti-core prompts
class ExtractNodes(dspy.Signature):
    """You are an AI assistant that extracts entity nodes from conversational messages.
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation."""

    entity_types: str = dspy.InputField(desc="JSON list of entity type definitions")
    episode_content: str = dspy.InputField(desc="Current message to extract from")
    previous_episodes: str = dspy.InputField(desc="JSON list of prior messages")
    custom_prompt: str = dspy.InputField(desc="Additional instructions")

    extracted_entities: ExtractedEntities = dspy.OutputField(desc="List of extracted entities")

# Configure with Outlines for constrained generation
dspy.configure(lm=OutlinesLM(), adapter=OutlinesAdapter())
extractor = dspy.Predict(ExtractNodes)
result = extractor(**context)
```

---

## File Locations

- **Core library**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/lib.py`
- **Message types**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/models.py`
- **Helpers**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/prompt_helpers.py`
- **Extract nodes**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/extract_nodes.py`
- **Extract edges**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/extract_edges.py`
- **Dedupe nodes**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/dedupe_nodes.py`
- **Dedupe edges**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/dedupe_edges.py`
- **Snippets**: `.venv/lib/python3.13/site-packages/graphiti_core/prompts/snippets.py`
