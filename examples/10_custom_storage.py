"""Custom Storage Backend Example.

This example shows how to implement your own storage backend
to use LLMem with any database (Redis, SQLite, DynamoDB, etc.)

Requirements:
    Implement the StorageBackend abstract class
"""

import asyncio
from typing import List, Optional, Dict
from datetime import datetime, timezone

from llmem import Memory
from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic


class RedisStorage(StorageBackend):
    """Example: Redis storage backend.
    
    This is a reference implementation showing how to build
    a custom storage backend. Adapt this pattern for any database.
    
    For actual Redis usage, install: pip install redis
    """
    
    def __init__(self, redis_client=None, prefix: str = "llmem"):
        """Initialize Redis storage.
        
        Args:
            redis_client: Redis client instance (redis.asyncio.Redis)
            prefix: Key prefix for namespacing
        """
        self.client = redis_client
        self.prefix = prefix
        # For demo without actual Redis, use in-memory dict
        self._mock_data: Dict[str, List[Turn]] = {}
    
    def _key(self, thread_id: str, suffix: str = "turns") -> str:
        """Generate Redis key."""
        return f"{self.prefix}:{thread_id}:{suffix}"
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a conversation turn."""
        # Real Redis: await self.client.rpush(self._key(thread_id), turn.to_json())
        
        # Demo implementation:
        if thread_id not in self._mock_data:
            self._mock_data[thread_id] = []
        self._mock_data[thread_id].append(turn)
    
    async def get_turns(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Turn]:
        """Get turns for a thread."""
        # Real Redis: 
        # data = await self.client.lrange(self._key(thread_id), offset, offset + limit - 1)
        # return [Turn.from_json(d) for d in data]
        
        # Demo implementation:
        turns = self._mock_data.get(thread_id, [])
        if limit:
            return turns[offset:offset + limit]
        return turns[offset:]
    
    async def get_turn_count(self, thread_id: str) -> int:
        """Get total number of turns."""
        # Real Redis: return await self.client.llen(self._key(thread_id))
        
        # Demo implementation:
        return len(self._mock_data.get(thread_id, []))
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update an existing turn."""
        # Real Redis: Find index and use LSET
        
        # Demo implementation:
        turns = self._mock_data.get(thread_id, [])
        for i, t in enumerate(turns):
            if t.id == turn.id:
                turns[i] = turn
                break
    
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete specific turns."""
        # Real Redis: Remove by value or rebuild list
        
        # Demo implementation:
        if thread_id in self._mock_data:
            self._mock_data[thread_id] = [
                t for t in self._mock_data[thread_id] 
                if t.id not in turn_ids
            ]
    
    async def clear(self, thread_id: str) -> None:
        """Clear all turns for a thread."""
        # Real Redis: await self.client.delete(self._key(thread_id))
        
        # Demo implementation:
        self._mock_data.pop(thread_id, None)
    
    # Optional: Implement topic methods for full persistence
    async def save_topic(self, topic: Topic, thread_id: str) -> None:
        """Save a topic (optional)."""
        pass
    
    async def get_topics(self, thread_id: str) -> List[Topic]:
        """Get topics (optional)."""
        return []


class SQLiteStorage(StorageBackend):
    """Example: SQLite storage backend.
    
    Shows how to implement storage with a SQL database.
    For actual usage, use aiosqlite: pip install aiosqlite
    """
    
    def __init__(self, db_path: str = "llmem.db"):
        """Initialize SQLite storage."""
        self.db_path = db_path
        # For demo without actual SQLite:
        self._mock_data: Dict[str, List[Turn]] = {}
    
    async def initialize(self) -> None:
        """Create tables if they don't exist.
        
        Real implementation:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS turns (
                        id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        token_count INTEGER,
                        created_at TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                await db.execute(
                    'CREATE INDEX IF NOT EXISTS idx_thread ON turns(thread_id)'
                )
                await db.commit()
        """
        pass
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a conversation turn."""
        # Real SQL: INSERT INTO turns VALUES (...)
        if thread_id not in self._mock_data:
            self._mock_data[thread_id] = []
        self._mock_data[thread_id].append(turn)
    
    async def get_turns(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Turn]:
        """Get turns for a thread."""
        # Real SQL: SELECT * FROM turns WHERE thread_id = ? ORDER BY created_at
        turns = self._mock_data.get(thread_id, [])
        if limit:
            return turns[offset:offset + limit]
        return turns[offset:]
    
    async def get_turn_count(self, thread_id: str) -> int:
        """Get total number of turns."""
        # Real SQL: SELECT COUNT(*) FROM turns WHERE thread_id = ?
        return len(self._mock_data.get(thread_id, []))
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update an existing turn."""
        # Real SQL: UPDATE turns SET ... WHERE id = ?
        turns = self._mock_data.get(thread_id, [])
        for i, t in enumerate(turns):
            if t.id == turn.id:
                turns[i] = turn
                break
    
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete specific turns."""
        # Real SQL: DELETE FROM turns WHERE id IN (...)
        if thread_id in self._mock_data:
            self._mock_data[thread_id] = [
                t for t in self._mock_data[thread_id] 
                if t.id not in turn_ids
            ]
    
    async def clear(self, thread_id: str) -> None:
        """Clear all turns for a thread."""
        # Real SQL: DELETE FROM turns WHERE thread_id = ?
        self._mock_data.pop(thread_id, None)


async def main():
    """Demonstrate custom storage backends."""
    
    print("=" * 60)
    print("Custom Storage Backend Demo")
    print("=" * 60)
    
    # Test with Redis-style storage
    print("\n[1] Testing RedisStorage")
    print("-" * 40)
    
    redis_storage = RedisStorage(prefix="myapp")
    memory = Memory(storage=redis_storage)
    
    memory.add("Hello, I need help with Python", role="user", thread_id="user-1")
    memory.add("I'd be happy to help with Python!", role="assistant", thread_id="user-1")
    memory.add("How do I read a file?", role="user", thread_id="user-1")
    
    context = memory.get_context(thread_id="user-1")
    print(f"Stored {len(context)} messages")
    for msg in context:
        print(f"  [{msg['role']}]: {msg['content'][:40]}...")
    
    health = memory.check_health(thread_id="user-1")
    print(f"Health: {health.status.value}, Turns: {health.turn_count}")
    
    # Test with SQLite-style storage
    print("\n[2] Testing SQLiteStorage")
    print("-" * 40)
    
    sqlite_storage = SQLiteStorage(db_path="llmem.db")
    memory2 = Memory(storage=sqlite_storage)
    
    memory2.add("What's the weather?", role="user", thread_id="session-abc")
    memory2.add("I can't check weather, but I can help with code!", role="assistant", thread_id="session-abc")
    
    stats = memory2.get_stats(thread_id="session-abc")
    print(f"Stats: {stats['total_turns']} turns, {stats['total_tokens']} tokens")
    
    print("\n" + "=" * 60)
    print("Custom storage backends work!")
    print("=" * 60)
    print("""
To implement your own storage backend:

1. Inherit from StorageBackend
2. Implement the required async methods:
   - save_turn(turn, thread_id)
   - get_turns(thread_id, limit, offset)
   - get_turn_count(thread_id)
   - update_turn(turn, thread_id)
   - delete_turns(turn_ids, thread_id)
   - clear(thread_id)

3. Optionally implement topic methods for full persistence

See examples/10_custom_storage.py for reference implementations.
""")


if __name__ == "__main__":
    asyncio.run(main())
