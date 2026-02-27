"""PostgreSQL storage backend for LLMem.

Requires: pip install llmem[postgres]
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic


logger = logging.getLogger("llmem.storage.postgres")


# SQL for creating tables
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS llmem_turns (
    id VARCHAR(36) PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    topic_id VARCHAR(36),
    is_compressed BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_llmem_turns_thread_id ON llmem_turns(thread_id);
CREATE INDEX IF NOT EXISTS idx_llmem_turns_created_at ON llmem_turns(thread_id, created_at);

CREATE TABLE IF NOT EXISTS llmem_topics (
    id VARCHAR(36) PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    turn_ids JSONB DEFAULT '[]',
    embedding JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llmem_topics_thread_id ON llmem_topics(thread_id);
"""


class PostgresStorage(StorageBackend):
    """PostgreSQL storage backend using asyncpg.
    
    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> storage = PostgresStorage(pool=pool)
        >>> memory = Memory(storage=storage)
    
    The storage will create tables if they don't exist.
    """
    
    def __init__(
        self,
        pool: Any,  # asyncpg.Pool
        table_prefix: str = "llmem_",
        auto_create_tables: bool = True,
    ):
        """Initialize PostgresStorage.
        
        Args:
            pool: asyncpg connection pool
            table_prefix: Prefix for table names (default: "llmem_")
            auto_create_tables: Create tables if they don't exist
        """
        self.pool = pool
        self.table_prefix = table_prefix
        self._tables_created = False
        self._auto_create = auto_create_tables
        
        self._turns_table = f"{table_prefix}turns"
        self._topics_table = f"{table_prefix}topics"
    
    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if self._tables_created or not self._auto_create:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        
        self._tables_created = True
        logger.debug("PostgreSQL tables created/verified")
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a turn to PostgreSQL."""
        await self._ensure_tables()
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._turns_table} 
                (id, thread_id, content, role, created_at, metadata, token_count, topic_id, is_compressed)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    token_count = EXCLUDED.token_count,
                    topic_id = EXCLUDED.topic_id,
                    is_compressed = EXCLUDED.is_compressed
                """,
                turn.id,
                thread_id,
                turn.content,
                turn.role,
                turn.created_at,
                json.dumps(turn.metadata),
                turn.token_count,
                turn.topic_id,
                turn.is_compressed,
            )
        
        logger.debug(f"Saved turn {turn.id} to PostgreSQL")
    
    async def get_turns(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Turn]:
        """Get turns from PostgreSQL."""
        await self._ensure_tables()
        
        query = f"""
            SELECT id, content, role, created_at, metadata, token_count, topic_id, is_compressed
            FROM {self._turns_table}
            WHERE thread_id = $1
            ORDER BY created_at ASC
        """
        
        params = [thread_id]
        
        if limit is not None:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)
        
        if offset > 0:
            query += f" OFFSET ${len(params) + 1}"
            params.append(offset)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        turns = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            turns.append(Turn(
                id=row["id"],
                content=row["content"],
                role=row["role"],
                created_at=row["created_at"],
                metadata=metadata or {},
                token_count=row["token_count"],
                topic_id=row["topic_id"],
                is_compressed=row["is_compressed"],
            ))
        
        return turns
    
    async def get_turn_count(self, thread_id: str) -> int:
        """Get turn count from PostgreSQL."""
        await self._ensure_tables()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self._turns_table} WHERE thread_id = $1",
                thread_id,
            )
        
        return result or 0
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update a turn in PostgreSQL."""
        await self.save_turn(turn, thread_id)
    
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete turns from PostgreSQL."""
        await self._ensure_tables()
        
        if not turn_ids:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._turns_table} WHERE thread_id = $1 AND id = ANY($2)",
                thread_id,
                turn_ids,
            )
        
        logger.debug(f"Deleted {len(turn_ids)} turns from PostgreSQL")
    
    async def clear(self, thread_id: str) -> None:
        """Clear all data for a thread."""
        await self._ensure_tables()
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._turns_table} WHERE thread_id = $1",
                thread_id,
            )
            await conn.execute(
                f"DELETE FROM {self._topics_table} WHERE thread_id = $1",
                thread_id,
            )
        
        logger.debug(f"Cleared thread {thread_id} from PostgreSQL")
    
    async def save_topic(self, topic: Topic, thread_id: str) -> None:
        """Save a topic to PostgreSQL."""
        await self._ensure_tables()
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._topics_table}
                (id, thread_id, name, turn_ids, embedding, created_at, last_accessed)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    turn_ids = EXCLUDED.turn_ids,
                    embedding = EXCLUDED.embedding,
                    last_accessed = EXCLUDED.last_accessed
                """,
                topic.id,
                thread_id,
                topic.name,
                json.dumps(topic.turn_ids),
                json.dumps(topic.embedding) if topic.embedding else None,
                topic.created_at,
                topic.last_accessed,
            )
    
    async def get_topics(self, thread_id: str) -> List[Topic]:
        """Get topics from PostgreSQL."""
        await self._ensure_tables()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, name, turn_ids, embedding, created_at, last_accessed
                FROM {self._topics_table}
                WHERE thread_id = $1
                ORDER BY created_at ASC
                """,
                thread_id,
            )
        
        topics = []
        for row in rows:
            turn_ids = row["turn_ids"]
            if isinstance(turn_ids, str):
                turn_ids = json.loads(turn_ids)
            
            embedding = row["embedding"]
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            
            topics.append(Topic(
                id=row["id"],
                name=row["name"] or "",
                turn_ids=turn_ids or [],
                embedding=embedding,
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
            ))
        
        return topics
    
    async def update_topic(self, topic: Topic, thread_id: str) -> None:
        """Update a topic in PostgreSQL."""
        await self.save_topic(topic, thread_id)
