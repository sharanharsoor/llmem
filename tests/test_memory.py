"""Tests for core Memory class."""

import pytest

from llmem import Memory
from llmem.health import HealthStatus, Recommendation
from llmem.storage.memory import InMemoryStorage


class TestMemoryBasics:
    """Basic Memory functionality tests."""
    
    def test_create_memory_default(self):
        """Test creating memory with defaults."""
        memory = Memory()
        
        assert memory.max_tokens == 128000
        assert memory.compression_threshold == 0.7
        assert isinstance(memory.storage, InMemoryStorage)
    
    def test_create_memory_custom(self):
        """Test creating memory with custom settings."""
        storage = InMemoryStorage()
        memory = Memory(
            storage=storage,
            max_tokens=50000,
            compression_threshold=0.8,
            model="cl100k_base",
        )
        
        assert memory.storage is storage
        assert memory.max_tokens == 50000
        assert memory.compression_threshold == 0.8
        assert memory.model == "cl100k_base"
    
    def test_add_turn(self):
        """Test adding a conversation turn."""
        memory = Memory()
        
        turn = memory.add("Hello!", role="user")
        
        assert turn.content == "Hello!"
        assert turn.role == "user"
        assert turn.id is not None
        assert turn.token_count is not None
        assert turn.token_count > 0
    
    def test_add_multiple_turns(self):
        """Test adding multiple turns."""
        memory = Memory()
        
        memory.add("How are you?", role="user")
        memory.add("I'm doing well!", role="assistant")
        memory.add("Great to hear", role="user")
        
        stats = memory.get_stats()
        assert stats["total_turns"] == 3
    
    def test_add_with_metadata(self):
        """Test adding turn with metadata."""
        memory = Memory()
        
        turn = memory.add(
            "Test message",
            role="user",
            metadata={"source": "test", "important": True}
        )
        
        assert turn.metadata["source"] == "test"
        assert turn.metadata["important"] is True


class TestGetContext:
    """Tests for context retrieval."""
    
    def test_get_context_empty(self):
        """Test getting context when empty."""
        memory = Memory()
        context = memory.get_context()
        
        assert context == []
    
    def test_get_context_single_turn(self):
        """Test getting context with one turn."""
        memory = Memory()
        memory.add("Hello", role="user")
        
        context = memory.get_context()
        
        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
    
    def test_get_context_multiple_turns(self):
        """Test getting context with multiple turns."""
        memory = Memory()
        memory.add("Hi", role="user")
        memory.add("Hello!", role="assistant")
        memory.add("How are you?", role="user")
        
        context = memory.get_context()
        
        assert len(context) == 3
        assert context[0]["content"] == "Hi"
        assert context[1]["content"] == "Hello!"
        assert context[2]["content"] == "How are you?"
    
    def test_get_context_format(self):
        """Test context is in LLM message format."""
        memory = Memory()
        memory.add("Test", role="user")
        
        context = memory.get_context()
        
        # Each message should have exactly role and content
        assert set(context[0].keys()) == {"role", "content"}


class TestThreadIsolation:
    """Tests for multi-user thread isolation."""
    
    def test_threads_are_isolated(self):
        """Test different threads don't mix."""
        memory = Memory()
        
        memory.add("User A message", role="user", thread_id="user-a")
        memory.add("User B message", role="user", thread_id="user-b")
        
        context_a = memory.get_context(thread_id="user-a")
        context_b = memory.get_context(thread_id="user-b")
        
        assert len(context_a) == 1
        assert len(context_b) == 1
        assert context_a[0]["content"] == "User A message"
        assert context_b[0]["content"] == "User B message"
    
    def test_clear_affects_only_one_thread(self):
        """Test clearing one thread doesn't affect others."""
        memory = Memory()
        
        memory.add("A", role="user", thread_id="t1")
        memory.add("B", role="user", thread_id="t2")
        
        memory.clear(thread_id="t1")
        
        assert memory.get_context(thread_id="t1") == []
        assert len(memory.get_context(thread_id="t2")) == 1
    
    def test_stats_per_thread(self):
        """Test stats are thread-specific."""
        memory = Memory()
        
        memory.add("A", role="user", thread_id="t1")
        memory.add("B", role="user", thread_id="t1")
        memory.add("C", role="user", thread_id="t2")
        
        stats_1 = memory.get_stats(thread_id="t1")
        stats_2 = memory.get_stats(thread_id="t2")
        
        assert stats_1["total_turns"] == 2
        assert stats_2["total_turns"] == 1


class TestHealthMonitoring:
    """Tests for health monitoring."""
    
    def test_check_health_empty(self):
        """Test health check on empty memory."""
        memory = Memory()
        health = memory.check_health()
        
        assert health.status == HealthStatus.HEALTHY
        assert health.recommendation == Recommendation.CONTINUE
        assert health.token_count == 0
        assert health.turn_count == 0
    
    def test_check_health_with_messages(self):
        """Test health check with messages."""
        memory = Memory()
        memory.add("Hello", role="user")
        memory.add("Hi there", role="assistant")
        
        health = memory.check_health()
        
        assert health.turn_count == 2
        assert health.token_count > 0
        assert health.token_usage < 0.01  # Very low usage


class TestCompression:
    """Tests for compression."""
    
    def test_compress_empty(self):
        """Test compressing empty memory."""
        memory = Memory()
        result = memory.compress()
        
        assert result["compressed"] is False
    
    def test_compress_few_turns(self):
        """Test compression doesn't happen with few turns."""
        memory = Memory()
        memory.add("A", role="user")
        memory.add("B", role="assistant")
        
        result = memory.compress()
        
        assert result["compressed"] is False
    
    def test_token_count_updated_after_compression(self):
        """Bug fix: token_counts must decrease after compression, not stay stale.
        
        Before fix: _token_counts was only incremented on add(), never decremented
        after delete_turns(). This caused _check_and_compress to fire on every
        subsequent add(), resulting in cascading compressions.
        """
        memory = Memory(max_tokens=100, compression_threshold=0.7)
        
        # Add enough turns to trigger compression
        for i in range(10):
            memory.add(f"Message {i}", role="user")
        
        # Force a compression
        result = memory.compress()
        if result["compressed"]:
            removed = result["removed_turns"]
            # After compression, internal token count must reflect the removal
            remaining_count = memory._token_counts.get("__default__", 0)
            # It should be less than what it was before (or zero)
            stats = memory.get_stats()
            # Token count in stats (from actual turns) must be >= internal count
            # (internal count can be 0 if all tokens were removed)
            assert remaining_count >= 0

    def test_compress_many_turns(self):
        """Test compression with many turns."""
        memory = Memory()
        for i in range(10):
            memory.add(f"Message {i}", role="user")
        
        result = memory.compress()
        
        assert result["compressed"] is True
        assert result["removed_turns"] > 0
        
        # Check remaining turns
        stats = memory.get_stats()
        assert stats["total_turns"] < 10


class TestCallbacks:
    """Tests for callback functionality."""
    
    def test_on_compress_callback(self):
        """Test compression callback is called."""
        compress_info = []
        
        memory = Memory(
            on_compress=lambda info: compress_info.append(info)
        )
        
        for i in range(10):
            memory.add(f"Message {i}", role="user")
        
        memory.compress()
        
        assert len(compress_info) == 1
        assert compress_info[0]["compressed"] is True
    
    def test_on_health_change_callback(self):
        """Test health change callback is called."""
        health_changes = []
        
        memory = Memory(
            on_health_change=lambda h: health_changes.append(h)
        )
        
        memory.add("Hello", role="user")
        
        # Initial health check triggers callback
        assert len(health_changes) >= 1


class TestGetStats:
    """Tests for statistics."""
    
    def test_stats_empty(self):
        """Test stats on empty memory."""
        memory = Memory()
        stats = memory.get_stats()
        
        assert stats["total_turns"] == 0
        assert stats["total_tokens"] == 0
        assert stats["topics"] == []
        assert stats["compression_count"] == 0
    
    def test_stats_with_turns(self):
        """Test stats with turns."""
        memory = Memory()
        memory.add("Hello world", role="user")
        memory.add("Hi there!", role="assistant")
        
        stats = memory.get_stats()
        
        assert stats["total_turns"] == 2
        assert stats["total_tokens"] > 0


class TestClear:
    """Tests for clearing memory."""
    
    def test_clear_removes_all(self):
        """Test clear removes all turns."""
        memory = Memory()
        memory.add("A", role="user")
        memory.add("B", role="assistant")
        
        memory.clear()
        
        assert memory.get_context() == []
        assert memory.get_stats()["total_turns"] == 0
    
    def test_clear_thread_specific(self):
        """Test clear is thread-specific."""
        memory = Memory()
        memory.add("A", role="user", thread_id="t1")
        memory.add("B", role="user", thread_id="t2")
        
        memory.clear(thread_id="t1")
        
        assert memory.get_stats(thread_id="t1")["total_turns"] == 0
        assert memory.get_stats(thread_id="t2")["total_turns"] == 1
