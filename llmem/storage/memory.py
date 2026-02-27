"""In-memory storage backend for LLMem."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic


class InMemoryStorage(StorageBackend):
    """In-memory storage backend.
    
    Perfect for:
    - Development and testing
    - Single-session applications
    - Demos and examples
    
    Note: Data is lost when the process ends.
    """
    
    def __init__(self):
        self._turns: Dict[str, List[Turn]] = defaultdict(list)
        self._topics: Dict[str, List[Topic]] = defaultdict(list)
    
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a turn to memory."""
        self._turns[thread_id].append(turn)
    
    async def get_turns(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Turn]:
        """Get turns from memory."""
        turns = self._turns[thread_id]
        
        if offset:
            turns = turns[offset:]
        
        if limit is not None:
            turns = turns[:limit]
        
        return turns
    
    async def get_turn_count(self, thread_id: str) -> int:
        """Get turn count."""
        return len(self._turns[thread_id])
    
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update a turn in memory."""
        turns = self._turns[thread_id]
        for i, t in enumerate(turns):
            if t.id == turn.id:
                turns[i] = turn
                return
    
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete turns from memory."""
        turn_ids_set = set(turn_ids)
        self._turns[thread_id] = [
            t for t in self._turns[thread_id] 
            if t.id not in turn_ids_set
        ]
    
    async def clear(self, thread_id: str) -> None:
        """Clear all turns for a thread."""
        self._turns[thread_id] = []
        self._topics[thread_id] = []
    
    async def save_topic(self, topic: Topic, thread_id: str) -> None:
        """Save a topic to memory."""
        # Update if exists, else append
        topics = self._topics[thread_id]
        for i, t in enumerate(topics):
            if t.id == topic.id:
                topics[i] = topic
                return
        topics.append(topic)
    
    async def get_topics(self, thread_id: str) -> List[Topic]:
        """Get topics from memory."""
        return self._topics[thread_id]
    
    async def update_topic(self, topic: Topic, thread_id: str) -> None:
        """Update a topic in memory."""
        await self.save_topic(topic, thread_id)
    
    def clear_all(self) -> None:
        """Clear all data (all threads). Useful for testing."""
        self._turns.clear()
        self._topics.clear()
