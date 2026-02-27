# LLM SmartMem

[![PyPI version](https://img.shields.io/pypi/v/llm-smartmem.svg)](https://pypi.org/project/llm-smartmem/)
[![Python](https://img.shields.io/pypi/pyversions/llm-smartmem.svg)](https://pypi.org/project/llm-smartmem/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/sharanharsoor/llm-smartmem/actions/workflows/test.yml/badge.svg)](https://github.com/sharanharsoor/llm-smartmem/actions/workflows/test.yml)

Smart memory management for LLM conversations - topic-aware compression that just works.

## Features

- **LLM-agnostic** - Works with OpenAI, Gemini, Anthropic, local models, or any LLM
- **Topic-aware compression** - Intelligently compresses based on conversation topics, not just token count
- **Storage-agnostic** - Works with PostgreSQL, MongoDB, or in-memory
- **LangChain/LangGraph compatible** - Works seamlessly with popular frameworks
- **Zero-config start** - Works out of the box with smart defaults
- **Multi-user safe** - Thread isolation for millions of users via `thread_id`
- **Fast** - Target <100ms for context retrieval

## Installation

```bash
pip install llm-smartmem
```

With optional dependencies:

```bash
pip install llm-smartmem[postgres]    # PostgreSQL storage
pip install llm-smartmem[mongo]       # MongoDB storage
pip install llm-smartmem[all]         # Everything
```

## Quick Start

```python
from llmem import Memory

# Create memory (zero config)
memory = Memory()

# Add conversation turns
memory.add("How do I setup my VR headset?", role="user")
memory.add("To setup your VR headset, first...", role="assistant")
memory.add("What games do you recommend?", role="user")
memory.add("I recommend these games...", role="assistant")

# Get optimized context for next LLM call
context = memory.get_context()

# Check health
health = memory.check_health()
print(f"Status: {health.status.value}, Tokens: {health.token_count}")
```

## With Persistent Storage

### PostgreSQL

```python
import asyncpg
from llmem import Memory
from llmem.storage.postgres import PostgresStorage

pool = await asyncpg.create_pool("postgresql://user:pass@localhost/db")
storage = PostgresStorage(pool=pool)
memory = Memory(storage=storage)

# Thread ID for multi-user isolation
memory.add("Hello", role="user", thread_id="user-123")
context = memory.get_context(thread_id="user-123")
```

### MongoDB

```python
from motor.motor_asyncio import AsyncIOMotorClient
from llmem import Memory
from llmem.storage.mongo import MongoStorage

client = AsyncIOMotorClient("mongodb://localhost:27017")
storage = MongoStorage(db=client.mydb)
memory = Memory(storage=storage)
```

## With Any LLM

LLMem is **LLM-agnostic** - it manages conversation memory, you bring your own model:

```python
from llmem import Memory

memory = Memory()

# Add user message
memory.add(user_input, role="user")

# Get optimized context
context = memory.get_context()

# Use with ANY LLM - OpenAI, Gemini, Anthropic, local models, etc.
response = your_llm.generate(context)

# Track response
memory.add(response, role="assistant")
```

### OpenAI Example

```python
from openai import OpenAI
from llmem import Memory

client = OpenAI()
memory = Memory()

memory.add(user_input, role="user")
context = memory.get_context()

response = client.chat.completions.create(
    model="your-model",
    messages=context
)
memory.add(response.choices[0].message.content, role="assistant")
```

### Google Gemini Example

```python
import google.generativeai as genai
from llmem import Memory

genai.configure(api_key="your-key")
model = genai.GenerativeModel("your-model")
memory = Memory()

memory.add(user_input, role="user")
context = memory.get_context()
response = model.generate_content(str(context))
memory.add(response.text, role="assistant")
```

### Anthropic Claude Example

```python
from anthropic import Anthropic
from llmem import Memory

client = Anthropic()
memory = Memory()

memory.add(user_input, role="user")
context = memory.get_context()

response = client.messages.create(
    model="your-model",
    messages=context
)
memory.add(response.content[0].text, role="assistant")
```

### With LangChain (Any Provider)

```python
from langchain_core.messages import HumanMessage, AIMessage
from llmem import Memory

# Use any LangChain-supported LLM
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic

llm = YourLangChainLLM()
memory = Memory()

memory.add(user_input, role="user")
context = memory.get_context()

# Convert to LangChain messages
messages = [HumanMessage(content=m["content"]) if m["role"] == "user" 
            else AIMessage(content=m["content"]) for m in context]

response = llm.invoke(messages)
memory.add(response.content, role="assistant")
```

## Health Monitoring

```python
health = memory.check_health()
print(f"Status: {health.status.value}")        # healthy, warning, critical
print(f"Token usage: {health.token_usage:.1%}")
print(f"Recommendation: {health.recommendation.value}")

stats = memory.get_stats()
print(f"Total turns: {stats['total_turns']}")
print(f"Total tokens: {stats['total_tokens']}")
```

## Callbacks

```python
memory = Memory(
    on_compress=lambda info: print(f"Compressed: {info}"),
    on_health_change=lambda health: print(f"Health: {health.status.value}")
)
```

## Examples

See the [examples/](examples/) folder for complete working demos:

| Example | Description |
|---------|-------------|
| `01_basic_usage.py` | Core functionality - add, get, health, stats |
| `02_callbacks.py` | Compression and health callbacks |
| `03_multi_user.py` | Thread isolation for multi-user apps |
| `04_with_openai.py` | Integration with OpenAI GPT |
| `04_with_gemini.py` | Integration with Google Gemini |
| `05_langchain_integration.py` | LangChain with any LLM provider |
| `06_langgraph_integration.py` | LangGraph agents |
| `07_postgres_storage.py` | PostgreSQL persistent storage |
| `08_mongodb_storage.py` | MongoDB persistent storage |
| `09_e2e_agent_test.py` | End-to-end test with all backends |
| `10_custom_storage.py` | Build your own storage backend |

### Running Examples

```bash
# Clone and setup
git clone https://github.com/sharanharsoor/llmem.git
cd llmem
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Create .env file with your credentials
echo "GOOGLE_API_KEY=your-key" > .env
echo "DATABASE_URL=postgresql://user:pass@localhost/db" >> .env
echo "MONGODB_URL=mongodb://localhost:27017" >> .env

# Run examples
python examples/01_basic_usage.py
python examples/04_with_gemini.py
```

## API Reference

### Memory Class

| Method | Description |
|--------|-------------|
| `add(content, role, thread_id=None)` | Add a conversation turn |
| `get_context(thread_id=None)` | Get optimized context |
| `get_context_for(query, thread_id=None)` | Get context relevant to query |
| `check_health(thread_id=None)` | Get context health metrics |
| `get_stats(thread_id=None)` | Get statistics |
| `compress(thread_id=None)` | Force compression |
| `clear(thread_id=None)` | Clear memory |

### Storage Backends

| Backend | Description |
|---------|-------------|
| `InMemoryStorage` | Default, no persistence |
| `PostgresStorage` | PostgreSQL with asyncpg |
| `MongoStorage` | MongoDB with motor |
| Custom | Implement `StorageBackend` for any database |

## Custom Storage Backend

LLMem supports **any database**. Implement the `StorageBackend` interface:

```python
from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic

class MyCustomStorage(StorageBackend):
    """Your custom storage (Redis, SQLite, DynamoDB, etc.)"""
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        # Save turn to your database
        pass
    
    async def get_turns(self, thread_id: str, limit=None, offset=0) -> list:
        # Retrieve turns from your database
        pass
    
    async def get_turn_count(self, thread_id: str) -> int:
        # Return count of turns
        pass
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        # Update existing turn
        pass
    
    async def delete_turns(self, turn_ids: list, thread_id: str) -> None:
        # Delete specific turns
        pass
    
    async def clear(self, thread_id: str) -> None:
        # Clear all turns for thread
        pass

# Use your custom storage
storage = MyCustomStorage()
memory = Memory(storage=storage)
```

See `examples/10_custom_storage.py` for complete Redis and SQLite reference implementations.

## Configuration

```python
memory = Memory(
    max_tokens=128000,          # Max context tokens
    compression_threshold=0.7,  # Compress at 70% usage
)
```

## License

MIT
