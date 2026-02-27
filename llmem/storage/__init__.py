"""Storage backends for LLMem."""

from llmem.storage.base import StorageBackend
from llmem.storage.memory import InMemoryStorage

__all__ = ["StorageBackend", "InMemoryStorage"]

# Lazy imports for optional backends
def PostgresStorage(*args, **kwargs):
    """PostgreSQL storage backend. Requires: pip install llmem[postgres]"""
    from llmem.storage.postgres import PostgresStorage as _PostgresStorage
    return _PostgresStorage(*args, **kwargs)

def MongoStorage(*args, **kwargs):
    """MongoDB storage backend. Requires: pip install llmem[mongo]"""
    from llmem.storage.mongo import MongoStorage as _MongoStorage
    return _MongoStorage(*args, **kwargs)
