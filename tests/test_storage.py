"""Tests for storage backends."""

import pytest

from llmem.storage.memory import InMemoryStorage
from llmem.storage.base import StorageBackend
from llmem.types import Turn, Topic


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""
    
    @pytest.fixture
    def storage(self):
        """Fresh storage instance."""
        return InMemoryStorage()
    
    @pytest.mark.asyncio
    async def test_save_and_get_turn(self, storage):
        """Test saving and retrieving a turn."""
        turn = Turn(content="Hello", role="user")
        
        await storage.save_turn(turn, "thread-1")
        turns = await storage.get_turns("thread-1")
        
        assert len(turns) == 1
        assert turns[0].content == "Hello"
        assert turns[0].role == "user"
    
    @pytest.mark.asyncio
    async def test_multiple_turns(self, storage):
        """Test multiple turns are retrieved in order."""
        await storage.save_turn(Turn(content="First", role="user"), "t1")
        await storage.save_turn(Turn(content="Second", role="assistant"), "t1")
        await storage.save_turn(Turn(content="Third", role="user"), "t1")
        
        turns = await storage.get_turns("t1")
        
        assert len(turns) == 3
        assert turns[0].content == "First"
        assert turns[1].content == "Second"
        assert turns[2].content == "Third"
    
    @pytest.mark.asyncio
    async def test_thread_isolation(self, storage):
        """Test turns are isolated by thread."""
        await storage.save_turn(Turn(content="Thread A", role="user"), "a")
        await storage.save_turn(Turn(content="Thread B", role="user"), "b")
        
        turns_a = await storage.get_turns("a")
        turns_b = await storage.get_turns("b")
        
        assert len(turns_a) == 1
        assert len(turns_b) == 1
        assert turns_a[0].content == "Thread A"
        assert turns_b[0].content == "Thread B"
    
    @pytest.mark.asyncio
    async def test_get_turns_with_limit(self, storage):
        """Test limiting returned turns."""
        for i in range(5):
            await storage.save_turn(Turn(content=f"Turn {i}", role="user"), "t1")
        
        turns = await storage.get_turns("t1", limit=3)
        
        assert len(turns) == 3
        assert turns[0].content == "Turn 0"
    
    @pytest.mark.asyncio
    async def test_get_turns_with_offset(self, storage):
        """Test offset for pagination."""
        for i in range(5):
            await storage.save_turn(Turn(content=f"Turn {i}", role="user"), "t1")
        
        turns = await storage.get_turns("t1", offset=2)
        
        assert len(turns) == 3
        assert turns[0].content == "Turn 2"
    
    @pytest.mark.asyncio
    async def test_get_turns_limit_and_offset(self, storage):
        """Test limit and offset together."""
        for i in range(10):
            await storage.save_turn(Turn(content=f"Turn {i}", role="user"), "t1")
        
        turns = await storage.get_turns("t1", limit=3, offset=2)
        
        assert len(turns) == 3
        assert turns[0].content == "Turn 2"
        assert turns[2].content == "Turn 4"
    
    @pytest.mark.asyncio
    async def test_get_turn_count(self, storage):
        """Test turn count."""
        assert await storage.get_turn_count("t1") == 0
        
        await storage.save_turn(Turn(content="A", role="user"), "t1")
        await storage.save_turn(Turn(content="B", role="user"), "t1")
        
        assert await storage.get_turn_count("t1") == 2
    
    @pytest.mark.asyncio
    async def test_update_turn(self, storage):
        """Test updating a turn."""
        turn = Turn(content="Original", role="user", id="turn-1")
        await storage.save_turn(turn, "t1")
        
        turn.content = "Updated"
        await storage.update_turn(turn, "t1")
        
        turns = await storage.get_turns("t1")
        assert turns[0].content == "Updated"
    
    @pytest.mark.asyncio
    async def test_delete_turns(self, storage):
        """Test deleting specific turns."""
        t1 = Turn(content="Keep", role="user", id="keep")
        t2 = Turn(content="Delete", role="user", id="delete")
        
        await storage.save_turn(t1, "t1")
        await storage.save_turn(t2, "t1")
        
        await storage.delete_turns(["delete"], "t1")
        
        turns = await storage.get_turns("t1")
        assert len(turns) == 1
        assert turns[0].content == "Keep"
    
    @pytest.mark.asyncio
    async def test_clear(self, storage):
        """Test clearing a thread."""
        await storage.save_turn(Turn(content="A", role="user"), "t1")
        await storage.save_turn(Turn(content="B", role="user"), "t1")
        
        await storage.clear("t1")
        
        assert await storage.get_turn_count("t1") == 0
    
    @pytest.mark.asyncio
    async def test_clear_doesnt_affect_other_threads(self, storage):
        """Test clear only affects one thread."""
        await storage.save_turn(Turn(content="A", role="user"), "t1")
        await storage.save_turn(Turn(content="B", role="user"), "t2")
        
        await storage.clear("t1")
        
        assert await storage.get_turn_count("t1") == 0
        assert await storage.get_turn_count("t2") == 1
    
    def test_clear_all(self, storage):
        """Test clearing all threads."""
        import asyncio
        
        async def setup():
            await storage.save_turn(Turn(content="A", role="user"), "t1")
            await storage.save_turn(Turn(content="B", role="user"), "t2")
        
        asyncio.run(setup())
        storage.clear_all()
        
        async def check():
            assert await storage.get_turn_count("t1") == 0
            assert await storage.get_turn_count("t2") == 0
        
        asyncio.run(check())
    
    @pytest.mark.asyncio
    async def test_empty_thread_returns_empty_list(self, storage):
        """Test getting turns from empty thread."""
        turns = await storage.get_turns("nonexistent")
        assert turns == []


class TestTopicStorage:
    """Tests for topic storage in InMemoryStorage."""
    
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()
    
    @pytest.mark.asyncio
    async def test_save_and_get_topic(self, storage):
        """Test saving and retrieving topics."""
        topic = Topic(id="t1", name="VR Setup")
        
        await storage.save_topic(topic, "thread-1")
        topics = await storage.get_topics("thread-1")
        
        assert len(topics) == 1
        assert topics[0].name == "VR Setup"
    
    @pytest.mark.asyncio
    async def test_update_topic(self, storage):
        """Test updating a topic."""
        topic = Topic(id="t1", name="Original")
        await storage.save_topic(topic, "thread-1")
        
        topic.name = "Updated"
        await storage.update_topic(topic, "thread-1")
        
        topics = await storage.get_topics("thread-1")
        assert topics[0].name == "Updated"
    
    @pytest.mark.asyncio
    async def test_topic_isolation(self, storage):
        """Test topics are isolated by thread."""
        await storage.save_topic(Topic(name="A"), "t1")
        await storage.save_topic(Topic(name="B"), "t2")
        
        topics_1 = await storage.get_topics("t1")
        topics_2 = await storage.get_topics("t2")
        
        assert len(topics_1) == 1
        assert len(topics_2) == 1
        assert topics_1[0].name == "A"
        assert topics_2[0].name == "B"
