# LLMem Examples

This folder contains example code demonstrating LLMem features.

## Examples

| File | Description |
|------|-------------|
| `01_basic_usage.py` | Core functionality - add, get, health, stats |
| `02_callbacks.py` | Event callbacks for compression and health |
| `03_multi_user.py` | Thread isolation for multi-user scenarios |
| `04_with_openai.py` | Integration with OpenAI API |
| `05_langchain_integration.py` | LangChain memory replacement |
| `06_langgraph_integration.py` | LangGraph checkpointer wrapper |
| `07_postgres_storage.py` | PostgreSQL persistent storage |
| `08_mongodb_storage.py` | MongoDB persistent storage |

## Running Examples

### Basic Examples (no external deps)

```bash
# Run directly
python examples/01_basic_usage.py
python examples/02_callbacks.py
python examples/03_multi_user.py
```

### OpenAI Example

```bash
# Install OpenAI
pip install openai

# Set API key
export OPENAI_API_KEY="your-key"

# Run
python examples/04_with_openai.py
```

### LangChain Example

```bash
# Install dependencies
pip install llmem[langchain] langchain-openai

# Set API key
export OPENAI_API_KEY="your-key"

# Run
python examples/05_langchain_integration.py
```

### LangGraph Example

```bash
# Install dependencies
pip install llmem[langgraph] langchain-openai langgraph

# Set API key
export OPENAI_API_KEY="your-key"

# Run
python examples/06_langgraph_integration.py
```

### PostgreSQL Example

```bash
# Install dependency
pip install llmem[postgres]

# Start PostgreSQL (Docker)
docker run -d --name llmem-postgres \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=llmem \
    -p 5432:5432 \
    postgres:15

# Run
python examples/07_postgres_storage.py
```

### MongoDB Example

```bash
# Install dependency
pip install llmem[mongo]

# Start MongoDB (Docker)
docker run -d --name llmem-mongo \
    -p 27017:27017 \
    mongo:7

# Run
python examples/08_mongodb_storage.py
```

## Need Help?

- Check the main [README](../README.md)
- Open an issue on GitHub
