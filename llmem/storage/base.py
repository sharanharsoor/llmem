"""Abstract base class for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from llmem.types import Turn, Topic


class StorageBackend(ABC):
    """Abstract base class for LLMem storage backends.
    
    Implement this interface to add support for any database.
    All methods should be async-compatible.
    
    Thread Isolation:
        All operations must be scoped to a thread_id.
        This ensures multi-user safety.
    """
    
    @abstractmethod
    async def save_turn(self, turn: Turn, thread_id: str) -> None:
        """Save a conversation turn.
        
        Args:
            turn: The turn to save
            thread_id: Thread identifier for isolation
        """
        pass
    
    @abstractmethod
    async def get_turns(
        self, 
        thread_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Turn]:
        """Get turns for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of turns to return (None = all)
            offset: Number of turns to skip
            
        Returns:
            List of turns, ordered by creation time (oldest first)
        """
        pass
    
    @abstractmethod
    async def get_turn_count(self, thread_id: str) -> int:
        """Get total number of turns in a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Number of turns
        """
        pass
    
    @abstractmethod
    async def update_turn(self, turn: Turn, thread_id: str) -> None:
        """Update an existing turn.
        
        Args:
            turn: The turn with updated data (matched by turn.id)
            thread_id: Thread identifier
        """
        pass
    
    @abstractmethod
    async def delete_turns(self, turn_ids: List[str], thread_id: str) -> None:
        """Delete specific turns.
        
        Args:
            turn_ids: IDs of turns to delete
            thread_id: Thread identifier
        """
        pass
    
    @abstractmethod
    async def clear(self, thread_id: str) -> None:
        """Clear all turns for a thread.
        
        Args:
            thread_id: Thread identifier
        """
        pass
    
    # Topic operations (optional - default implementations provided)
    
    async def save_topic(self, topic: Topic, thread_id: str) -> None:
        """Save a topic. Override for persistent storage."""
        pass
    
    async def get_topics(self, thread_id: str) -> List[Topic]:
        """Get all topics for a thread. Override for persistent storage."""
        return []
    
    async def update_topic(self, topic: Topic, thread_id: str) -> None:
        """Update a topic. Override for persistent storage."""
        pass
