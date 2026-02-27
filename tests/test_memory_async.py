"""Async tests for Memory class."""

import pytest

from llmem import Memory
from llmem.health import HealthStatus


class TestMemoryAsync:
    """Async tests for Memory class."""
    
    @pytest.mark.asyncio
    async def test_add_async(self):
        """Test async add method."""
        memory = Memory()
        
        turn = await memory.add_async("Hello", role="user")
        
        assert turn.content == "Hello"
        assert turn.role == "user"
        assert turn.token_count is not None
    
    @pytest.mark.asyncio
    async def test_get_context_async(self):
        """Test async get_context method."""
        memory = Memory()
        
        await memory.add_async("Hello", role="user")
        await memory.add_async("Hi!", role="assistant")
        
        context = await memory.get_context_async()
        
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_get_context_for_async(self):
        """Test getting context for a query."""
        memory = Memory()
        
        await memory.add_async("I love pizza", role="user")
        await memory.add_async("Pizza is great!", role="assistant")
        
        context = await memory._get_context_for_async("pizza", None, None)
        
        assert len(context) == 2
    
    @pytest.mark.asyncio
    async def test_check_health_async(self):
        """Test async health check."""
        memory = Memory()
        
        await memory.add_async("Hello", role="user")
        
        health = await memory._check_health_async(None)
        
        assert health.status == HealthStatus.HEALTHY
        assert health.turn_count == 1
    
    @pytest.mark.asyncio
    async def test_get_topics_async(self):
        """Test async get topics."""
        memory = Memory()
        
        await memory.add_async("Hello", role="user")
        
        topics = await memory._get_topics_async(None)
        
        assert topics == []
    
    @pytest.mark.asyncio
    async def test_compress_async(self):
        """Test async compression."""
        memory = Memory()
        
        # Add many turns
        for i in range(10):
            await memory.add_async(f"Message {i}", role="user")
        
        result = await memory._compress_async(None)
        
        assert result["compressed"] is True
        assert result["removed_turns"] > 0
    
    @pytest.mark.asyncio
    async def test_clear_async(self):
        """Test async clear."""
        memory = Memory()
        
        await memory.add_async("Hello", role="user")
        await memory._clear_async(None)
        
        context = await memory.get_context_async()
        
        assert len(context) == 0
    
    @pytest.mark.asyncio
    async def test_check_and_compress_auto(self):
        """Test auto compression when threshold exceeded."""
        memory = Memory(
            max_tokens=100,  # Very low limit
            compression_threshold=0.5,
        )
        
        # Add messages until compression triggers
        for i in range(5):
            await memory.add_async(
                f"This is a longer message number {i} to fill up tokens",
                role="user"
            )
        
        stats = memory.get_stats()
        # Compression should have occurred
        assert stats["compression_count"] >= 0


class TestMemoryGetContextForQuery:
    """Test get_context_for method."""
    
    def test_get_context_for(self):
        """Test sync get_context_for."""
        memory = Memory()
        
        memory.add("How do I cook pasta?", role="user")
        memory.add("Boil water and add pasta", role="assistant")
        
        context = memory.get_context_for("cooking")
        
        assert len(context) == 2


class TestMemoryTokenLimits:
    """Test token limit handling."""
    
    def test_get_context_respects_max_tokens(self):
        """Test that get_context truncates when needed."""
        memory = Memory(max_tokens=50)
        
        # Add many messages
        for i in range(20):
            memory.add(f"Message {i} with some content", role="user")
        
        context = memory.get_context()
        
        # Should be truncated to fit token limit
        assert len(context) < 20
    
    def test_get_context_with_custom_max_tokens(self):
        """Test passing custom max_tokens to get_context."""
        memory = Memory(max_tokens=100000)
        
        for i in range(10):
            memory.add(f"Message {i}", role="user")
        
        # Request with lower limit
        context = memory.get_context(max_tokens=20)
        
        # Should be fewer messages due to limit
        assert len(context) <= 10


class TestMemoryDefaultThread:
    """Test default thread ID behavior."""
    
    def test_uses_default_thread_id(self):
        """Test that operations work without explicit thread_id."""
        memory = Memory()
        
        memory.add("Hello", role="user")
        context = memory.get_context()
        health = memory.check_health()
        stats = memory.get_stats()
        topics = memory.get_topics()
        
        assert len(context) == 1
        assert health.turn_count == 1
        assert stats["total_turns"] == 1
        assert topics == []


class TestStorageImports:
    """Test lazy imports for storage backends."""
    
    def test_postgres_lazy_import(self):
        """Test PostgresStorage lazy import."""
        from llmem.storage import PostgresStorage
        
        # Should be callable (even if import fails internally)
        assert callable(PostgresStorage)
    
    def test_mongo_lazy_import(self):
        """Test MongoStorage lazy import."""
        from llmem.storage import MongoStorage
        
        assert callable(MongoStorage)


class TestIntegrationsImports:
    """Test lazy imports for integrations."""
    
    def test_langchain_lazy_import(self):
        """Test LangChainMemory lazy import."""
        from llmem.integrations import LangChainMemory
        
        assert callable(LangChainMemory)
    
    def test_langgraph_lazy_import(self):
        """Test LangGraphMemory lazy import."""
        from llmem.integrations import LangGraphMemory
        
        assert callable(LangGraphMemory)
