"""Core Memory class for LLMem."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from llmem.health import ContextHealth
from llmem.storage.base import StorageBackend
from llmem.storage.memory import InMemoryStorage
from llmem.types import Turn, Topic
from llmem.utils.tokens import count_tokens, count_message_tokens


logger = logging.getLogger("llmem")


# Default thread ID for single-user/stateless usage
DEFAULT_THREAD_ID = "__default__"


class Memory:
    """Smart memory management for LLM conversations.
    
    Features:
    - Topic-aware compression
    - Token tracking
    - Health monitoring
    - Callbacks for events
    
    Example:
        >>> memory = Memory()
        >>> memory.add("Hello!", role="user")
        >>> memory.add("Hi there!", role="assistant")
        >>> context = memory.get_context()
    """
    
    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_tokens: int = 128000,
        compression_threshold: float = 0.7,
        model: str = "cl100k_base",
        # Callbacks
        on_compress: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_topic_change: Optional[Callable[[Optional[str], str], None]] = None,
        on_health_change: Optional[Callable[[ContextHealth], None]] = None,
    ):
        """Initialize Memory.
        
        Args:
            storage: Storage backend (defaults to InMemoryStorage)
            max_tokens: Maximum context tokens before compression
            compression_threshold: Token usage ratio to trigger compression (0.0-1.0)
            model: Model name for tokenization
            on_compress: Callback when compression occurs
            on_topic_change: Callback when topic changes (old_topic, new_topic)
            on_health_change: Callback when health status changes
        """
        self.storage = storage or InMemoryStorage()
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.model = model
        
        # Callbacks
        self._on_compress = on_compress
        self._on_topic_change = on_topic_change
        self._on_health_change = on_health_change
        
        # Internal state
        self._current_topic: Dict[str, Optional[str]] = {}
        self._token_counts: Dict[str, int] = {}
        self._compressions_count: Dict[str, int] = {}
        self._last_health: Dict[str, Optional[ContextHealth]] = {}
        
        logger.debug(f"Memory initialized: max_tokens={max_tokens}, model={model}")
    
    # =====================
    # Core API Methods
    # =====================
    
    def add(
        self,
        content: str,
        role: str = "user",
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """Add a conversation turn.
        
        Args:
            content: Message content
            role: Role (user, assistant, system, tool)
            thread_id: Thread ID for multi-user isolation
            metadata: Optional metadata to attach
            
        Returns:
            The created Turn object
        """
        return self._run_async(
            self._add_async(content, role, thread_id, metadata)
        )
    
    async def add_async(
        self,
        content: str,
        role: str = "user",
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """Add a conversation turn (async)."""
        return await self._add_async(content, role, thread_id, metadata)
    
    async def _add_async(
        self,
        content: str,
        role: str,
        thread_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Turn:
        """Internal async implementation of add."""
        tid = thread_id or DEFAULT_THREAD_ID
        
        # Count tokens
        token_count = count_tokens(content, self.model)
        
        # Create turn
        turn = Turn(
            content=content,
            role=role,
            token_count=token_count,
            metadata=metadata or {},
        )
        
        # Save to storage
        await self.storage.save_turn(turn, tid)
        
        # Update token count
        self._token_counts[tid] = self._token_counts.get(tid, 0) + token_count
        
        logger.debug(f"Added turn: role={role}, tokens={token_count}, thread={tid}")
        
        # Check health and trigger compression if needed
        await self._check_and_compress(tid)
        
        return turn
    
    def get_context(
        self,
        thread_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get optimized context for LLM.
        
        Args:
            thread_id: Thread ID
            max_tokens: Optional token limit (uses self.max_tokens if not set)
            
        Returns:
            List of message dicts: [{"role": str, "content": str}, ...]
        """
        return self._run_async(
            self._get_context_async(thread_id, max_tokens)
        )
    
    async def get_context_async(
        self,
        thread_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get optimized context (async)."""
        return await self._get_context_async(thread_id, max_tokens)
    
    async def _get_context_async(
        self,
        thread_id: Optional[str],
        max_tokens: Optional[int],
    ) -> List[Dict[str, str]]:
        """Internal async implementation of get_context."""
        tid = thread_id or DEFAULT_THREAD_ID
        limit = max_tokens or self.max_tokens
        
        turns = await self.storage.get_turns(tid)
        
        if not turns:
            return []
        
        # Convert to message format
        messages = [t.to_message_dict() for t in turns]
        
        # Simple truncation from oldest if over limit
        total_tokens = count_message_tokens(messages, self.model)
        
        while total_tokens > limit and len(messages) > 1:
            messages.pop(0)  # Remove oldest
            total_tokens = count_message_tokens(messages, self.model)
        
        logger.debug(f"get_context: {len(messages)} messages, {total_tokens} tokens")
        return messages
    
    def get_context_for(
        self,
        query: str,
        thread_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get context relevant to a specific query.
        
        This enables topic-aware retrieval: returns context
        that is most relevant to the query.
        
        Args:
            query: The query to optimize context for
            thread_id: Thread ID
            max_tokens: Optional token limit
            
        Returns:
            List of relevant message dicts
        """
        return self._run_async(
            self._get_context_for_async(query, thread_id, max_tokens)
        )
    
    async def _get_context_for_async(
        self,
        query: str,
        thread_id: Optional[str],
        max_tokens: Optional[int],
    ) -> List[Dict[str, str]]:
        """Internal async implementation of get_context_for."""
        # For now, use simple recency-based context
        # Topic-aware relevance will be enhanced with embeddings
        return await self._get_context_async(thread_id, max_tokens)
    
    def check_health(
        self,
        thread_id: Optional[str] = None,
    ) -> ContextHealth:
        """Get context health metrics.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            ContextHealth with status and recommendations
        """
        return self._run_async(self._check_health_async(thread_id))
    
    async def _check_health_async(
        self,
        thread_id: Optional[str],
    ) -> ContextHealth:
        """Internal async implementation of check_health."""
        tid = thread_id or DEFAULT_THREAD_ID
        
        turns = await self.storage.get_turns(tid)
        topics = await self.storage.get_topics(tid)
        
        token_count = sum(t.token_count or 0 for t in turns)
        
        health = ContextHealth.calculate(
            token_count=token_count,
            max_tokens=self.max_tokens,
            turn_count=len(turns),
            topic_coherence=1.0,  # Will be calculated with embeddings
            topics_detected=len(topics),
            compressions_performed=self._compressions_count.get(tid, 0),
        )
        
        # Callback if health changed
        last = self._last_health.get(tid)
        if last is None or last.status != health.status:
            self._last_health[tid] = health
            if self._on_health_change:
                self._on_health_change(health)
        
        return health
    
    def get_stats(
        self,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the conversation.
        
        Returns dict with:
        - total_turns: Number of turns
        - total_tokens: Estimated token count
        - topics: List of topic names
        - compression_count: Number of compressions performed
        """
        return self._run_async(self._get_stats_async(thread_id))
    
    async def _get_stats_async(
        self,
        thread_id: Optional[str],
    ) -> Dict[str, Any]:
        """Internal async implementation of get_stats."""
        tid = thread_id or DEFAULT_THREAD_ID
        
        turns = await self.storage.get_turns(tid)
        topics = await self.storage.get_topics(tid)
        
        return {
            "total_turns": len(turns),
            "total_tokens": sum(t.token_count or 0 for t in turns),
            "topics": [t.name for t in topics if t.name],
            "compression_count": self._compressions_count.get(tid, 0),
        }
    
    def get_topics(
        self,
        thread_id: Optional[str] = None,
    ) -> List[Topic]:
        """Get detected topics.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            List of Topic objects
        """
        return self._run_async(self._get_topics_async(thread_id))
    
    async def _get_topics_async(
        self,
        thread_id: Optional[str],
    ) -> List[Topic]:
        """Internal async implementation of get_topics."""
        tid = thread_id or DEFAULT_THREAD_ID
        return await self.storage.get_topics(tid)
    
    def compress(
        self,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Force compression of conversation history.
        
        Returns:
            Info about compression performed
        """
        return self._run_async(self._compress_async(thread_id))
    
    async def _compress_async(
        self,
        thread_id: Optional[str],
    ) -> Dict[str, Any]:
        """Internal async implementation of compress."""
        tid = thread_id or DEFAULT_THREAD_ID
        
        turns = await self.storage.get_turns(tid)
        
        if len(turns) < 4:
            return {"compressed": False, "reason": "Not enough turns to compress"}
        
        # Simple compression: keep last N turns
        # Full LLM-based summarization will be added with compression strategies
        keep_count = max(4, len(turns) // 2)
        to_remove = turns[:-keep_count]
        
        if to_remove:
            # Decrement token count BEFORE deleting, so health is accurate after compression
            removed_tokens = sum(t.token_count or 0 for t in to_remove)
            await self.storage.delete_turns([t.id for t in to_remove], tid)
            self._token_counts[tid] = max(0, self._token_counts.get(tid, 0) - removed_tokens)

            # Track compression
            self._compressions_count[tid] = self._compressions_count.get(tid, 0) + 1
            
            info = {
                "compressed": True,
                "removed_turns": len(to_remove),
                "remaining_turns": keep_count,
                "removed_tokens": removed_tokens,
            }
            
            logger.info(f"Compressed: removed {len(to_remove)} turns")
            
            if self._on_compress:
                self._on_compress(info)
            
            return info
        
        return {"compressed": False, "reason": "No turns to remove"}
    
    def clear(
        self,
        thread_id: Optional[str] = None,
    ) -> None:
        """Clear all memory for a thread.
        
        Args:
            thread_id: Thread ID
        """
        self._run_async(self._clear_async(thread_id))
    
    async def _clear_async(
        self,
        thread_id: Optional[str],
    ) -> None:
        """Internal async implementation of clear."""
        tid = thread_id or DEFAULT_THREAD_ID
        await self.storage.clear(tid)
        
        # Reset internal state
        self._token_counts.pop(tid, None)
        self._current_topic.pop(tid, None)
        self._compressions_count.pop(tid, None)
        self._last_health.pop(tid, None)
        
        logger.info(f"Cleared memory for thread: {tid}")
    
    # =====================
    # Internal Helpers
    # =====================
    
    async def _check_and_compress(self, thread_id: str) -> None:
        """Check health and auto-compress if needed."""
        health = await self._check_health_async(thread_id)
        
        if health.token_usage >= self.compression_threshold:
            logger.info(f"Token usage {health.token_usage:.1%} >= threshold, compressing")
            await self._compress_async(thread_id)
    
    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
