"""MongoDB storage backend for LLMem.

Requires: pip install llmem[mongo]
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic


logger = logging.getLogger("llmem.storage.mongo")


class MongoStorage(StorageBackend):
    """MongoDB storage backend using motor (async driver).
    
    Example:
        >>> from motor.motor_asyncio import AsyncIOMotorClient
        >>> client = AsyncIOMotorClient("mongodb://localhost:27017")
        >>> db = client.my_database
        >>> storage = MongoStorage(db=db)
        >>> memory = Memory(storage=storage)
    """
    
    def __init__(
        self,
        db: Any,  # motor.motor_asyncio.AsyncIOMotorDatabase
        collection_prefix: str = "llmem_",
    ):
        """Initialize MongoStorage.
        
        Args:
            db: Motor database instance
            collection_prefix: Prefix for collection names (default: "llmem_")
        """
        self.db = db
        self.collection_prefix = collection_prefix
        
        self._turns_collection = db[f"{collection_prefix}turns"]
        self._topics_collection = db[f"{collection_prefix}topics"]
        self._indexes_created = False
    
    async def _ensure_indexes(self) -> None:
        """Create indexes for performance."""
        if self._indexes_created:
            return
        
        await self._turns_collection.create_index([("thread_id", 1)])
        await self._turns_collection.create_index([("thread_id", 1), ("created_at", 1)])
        await self._topics_collection.create_index([("thread_id", 1)])
        
        self._indexes_created = True
        logger.debug("MongoDB indexes created/verified")
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a turn to MongoDB."""
        await self._ensure_indexes()
        
        doc = {
            "_id": turn.id,
            "thread_id": thread_id,
            "content": turn.content,
            "role": turn.role,
            "created_at": turn.created_at,
            "metadata": turn.metadata,
            "token_count": turn.token_count,
            "topic_id": turn.topic_id,
            "is_compressed": turn.is_compressed,
        }
        
        await self._turns_collection.update_one(
            {"_id": turn.id},
            {"$set": doc},
            upsert=True,
        )
        
        logger.debug(f"Saved turn {turn.id} to MongoDB")
    
    async def get_turns(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Turn]:
        """Get turns from MongoDB."""
        await self._ensure_indexes()
        
        cursor = self._turns_collection.find(
            {"thread_id": thread_id}
        ).sort("created_at", 1)
        
        if offset > 0:
            cursor = cursor.skip(offset)
        
        if limit is not None:
            cursor = cursor.limit(limit)
        
        turns = []
        async for doc in cursor:
            turns.append(Turn(
                id=doc["_id"],
                content=doc["content"],
                role=doc["role"],
                created_at=doc["created_at"],
                metadata=doc.get("metadata", {}),
                token_count=doc.get("token_count"),
                topic_id=doc.get("topic_id"),
                is_compressed=doc.get("is_compressed", False),
            ))
        
        return turns
    
    async def get_turn_count(self, thread_id: str) -> int:
        """Get turn count from MongoDB."""
        await self._ensure_indexes()
        return await self._turns_collection.count_documents({"thread_id": thread_id})
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update a turn in MongoDB."""
        await self.save_turn(turn, thread_id)
    
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete turns from MongoDB."""
        await self._ensure_indexes()
        
        if not turn_ids:
            return
        
        await self._turns_collection.delete_many({
            "_id": {"$in": turn_ids},
            "thread_id": thread_id,
        })
        
        logger.debug(f"Deleted {len(turn_ids)} turns from MongoDB")
    
    async def clear(self, thread_id: str) -> None:
        """Clear all data for a thread."""
        await self._ensure_indexes()
        
        await self._turns_collection.delete_many({"thread_id": thread_id})
        await self._topics_collection.delete_many({"thread_id": thread_id})
        
        logger.debug(f"Cleared thread {thread_id} from MongoDB")
    
    async def save_topic(self, topic: Topic, thread_id: str) -> None:
        """Save a topic to MongoDB."""
        await self._ensure_indexes()
        
        doc = {
            "_id": topic.id,
            "thread_id": thread_id,
            "name": topic.name,
            "turn_ids": topic.turn_ids,
            "embedding": topic.embedding,
            "created_at": topic.created_at,
            "last_accessed": topic.last_accessed,
        }
        
        await self._topics_collection.update_one(
            {"_id": topic.id},
            {"$set": doc},
            upsert=True,
        )
    
    async def get_topics(self, thread_id: str) -> List[Topic]:
        """Get topics from MongoDB."""
        await self._ensure_indexes()
        
        topics = []
        cursor = self._topics_collection.find(
            {"thread_id": thread_id}
        ).sort("created_at", 1)
        
        async for doc in cursor:
            topics.append(Topic(
                id=doc["_id"],
                name=doc.get("name", ""),
                turn_ids=doc.get("turn_ids", []),
                embedding=doc.get("embedding"),
                created_at=doc["created_at"],
                last_accessed=doc["last_accessed"],
            ))
        
        return topics
    
    async def update_topic(self, topic: Topic, thread_id: str) -> None:
        """Update a topic in MongoDB."""
        await self.save_topic(topic, thread_id)
