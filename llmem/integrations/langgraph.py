"""LangGraph integration for LLMem.

Provides memory management that works with LangGraph's checkpointer system.

Requires: pip install llmem[langgraph]

Example:
    >>> from llmem.integrations.langgraph import LangGraphMemory
    >>> from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    >>> from langgraph.prebuilt import create_react_agent
    >>> 
    >>> checkpointer = AsyncPostgresSaver(conn=pool)
    >>> memory = LangGraphMemory(checkpointer=checkpointer)
    >>> 
    >>> agent = create_react_agent(model=model, tools=tools, checkpointer=memory)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, AsyncIterator, Iterator

from llmem import Memory
from llmem.storage.base import StorageBackend


logger = logging.getLogger("llmem.integrations.langgraph")


try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object


def _check_langgraph():
    """Check if LangGraph is available."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is required for this integration. "
            "Install with: pip install llmem[langgraph]"
        )


class LangGraphMemory(BaseCheckpointSaver):
    """LangGraph-compatible checkpointer with LLMem memory management.
    
    This class wraps a LangGraph checkpointer and adds LLMem's
    intelligent memory management capabilities:
    
    - Automatic context compression
    - Topic-aware history management
    - Token usage monitoring
    
    The checkpointer delegates to the underlying LangGraph checkpointer
    while LLMem manages the conversation context optimization.
    
    Example:
        >>> from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        >>> 
        >>> # Wrap your existing checkpointer
        >>> postgres_saver = AsyncPostgresSaver(conn=pool)
        >>> memory = LangGraphMemory(checkpointer=postgres_saver)
        >>> 
        >>> # Use with create_react_agent
        >>> agent = create_react_agent(
        ...     model=model,
        ...     tools=tools,
        ...     checkpointer=memory
        ... )
    """
    
    def __init__(
        self,
        checkpointer: Any,  # BaseCheckpointSaver
        memory: Optional[Memory] = None,
        storage: Optional[StorageBackend] = None,
        auto_compress: bool = True,
        compression_threshold: float = 0.7,
    ):
        """Initialize LangGraphMemory.
        
        Args:
            checkpointer: Underlying LangGraph checkpointer
            memory: Optional existing Memory instance
            storage: Optional storage backend (if not using existing memory)
            auto_compress: Enable automatic compression
            compression_threshold: Token usage threshold for compression
        """
        _check_langgraph()
        
        # Set checkpointer BEFORE calling super().__init__() 
        # because serde property may be accessed during init
        self._checkpointer = checkpointer
        self._memory = memory or Memory(
            storage=storage,
            compression_threshold=compression_threshold,
        )
        self._auto_compress = auto_compress
        
        super().__init__()
        
        logger.debug("LangGraphMemory initialized with wrapped checkpointer")
    
    @property
    def serde(self):
        """Serialization/deserialization from underlying checkpointer."""
        return self._checkpointer.serde
    
    async def aget(
        self,
        config: Dict[str, Any],
    ) -> Optional[CheckpointTuple]:
        """Get checkpoint asynchronously.
        
        Delegates to underlying checkpointer.
        """
        return await self._checkpointer.aget(config)
    
    def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[CheckpointTuple]:
        """Get checkpoint synchronously.
        
        Delegates to underlying checkpointer.
        """
        return self._checkpointer.get(config)
    
    async def aget_tuple(
        self,
        config: Dict[str, Any],
    ) -> Optional[CheckpointTuple]:
        """Get checkpoint tuple asynchronously."""
        return await self._checkpointer.aget_tuple(config)
    
    def get_tuple(
        self,
        config: Dict[str, Any],
    ) -> Optional[CheckpointTuple]:
        """Get checkpoint tuple synchronously."""
        return self._checkpointer.get_tuple(config)
    
    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save checkpoint asynchronously.
        
        Saves to underlying checkpointer and tracks in LLMem.
        """
        # Extract thread_id for LLMem tracking
        thread_id = config.get("configurable", {}).get("thread_id", "__default__")
        
        # Save to underlying checkpointer
        result = await self._checkpointer.aput(config, checkpoint, metadata, new_versions)
        
        # Track messages in LLMem for compression
        await self._track_messages(checkpoint, thread_id)
        
        return result
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save checkpoint synchronously."""
        thread_id = config.get("configurable", {}).get("thread_id", "__default__")
        
        result = self._checkpointer.put(config, checkpoint, metadata, new_versions)
        
        # Track synchronously
        import asyncio
        asyncio.run(self._track_messages(checkpoint, thread_id))
        
        return result
    
    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: List[tuple],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Put writes asynchronously."""
        await self._checkpointer.aput_writes(config, writes, task_id, task_path)
    
    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[tuple],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Put writes synchronously."""
        self._checkpointer.put_writes(config, writes, task_id, task_path)
    
    async def alist(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints asynchronously."""
        async for item in self._checkpointer.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item
    
    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints synchronously."""
        yield from self._checkpointer.list(
            config, filter=filter, before=before, limit=limit
        )
    
    async def _track_messages(
        self,
        checkpoint: Checkpoint,
        thread_id: str,
    ) -> None:
        """Track messages in LLMem for memory management.
        
        Extracts messages from checkpoint state and tracks them
        for compression and topic detection.
        """
        # Extract messages from checkpoint channel values
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        
        if not messages:
            return
        
        # Get the last message (most recent)
        last_msg = messages[-1] if messages else None
        
        if last_msg:
            # LangGraph messages have type and content
            content = getattr(last_msg, "content", str(last_msg))
            msg_type = type(last_msg).__name__.lower()
            
            # Map to role
            if "human" in msg_type:
                role = "user"
            elif "ai" in msg_type:
                role = "assistant"
            elif "system" in msg_type:
                role = "system"
            elif "tool" in msg_type:
                role = "tool"
            else:
                role = "assistant"
            
            # Add to LLMem (if not already tracked)
            await self._memory.add_async(
                content=content,
                role=role,
                thread_id=thread_id,
            )
        
        logger.debug(f"Tracked {len(messages)} messages for thread {thread_id}")
    
    # LLMem features
    
    def check_health(self, thread_id: str = "__default__"):
        """Get health metrics for a thread."""
        return self._memory.check_health(thread_id=thread_id)
    
    def get_stats(self, thread_id: str = "__default__"):
        """Get memory statistics for a thread."""
        return self._memory.get_stats(thread_id=thread_id)
    
    def compress(self, thread_id: str = "__default__"):
        """Force compression for a thread."""
        return self._memory.compress(thread_id=thread_id)
    
    def get_optimized_context(
        self,
        thread_id: str = "__default__",
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Get optimized conversation context.
        
        Use this to get a compressed version of the conversation
        that fits within token limits while preserving important context.
        """
        return self._memory.get_context(
            thread_id=thread_id,
            max_tokens=max_tokens,
        )
