"""Tests for core types."""

from datetime import datetime, timezone

import pytest

from llmem.types import Turn, Topic, CompressedSegment


class TestTurn:
    """Tests for Turn dataclass."""
    
    def test_create_turn_minimal(self):
        """Test creating a turn with minimal args."""
        turn = Turn(content="Hello", role="user")
        
        assert turn.content == "Hello"
        assert turn.role == "user"
        assert turn.id is not None
        assert turn.created_at is not None
        assert turn.metadata == {}
        assert turn.token_count is None
        assert turn.topic_id is None
        assert turn.is_compressed is False
    
    def test_create_turn_full(self):
        """Test creating a turn with all args."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            content="Hello",
            role="assistant",
            id="test-id",
            created_at=now,
            metadata={"key": "value"},
            token_count=5,
            topic_id="topic-1",
            is_compressed=True,
        )
        
        assert turn.id == "test-id"
        assert turn.created_at == now
        assert turn.metadata == {"key": "value"}
        assert turn.token_count == 5
        assert turn.topic_id == "topic-1"
        assert turn.is_compressed is True
    
    def test_turn_to_dict(self):
        """Test Turn serialization."""
        turn = Turn(content="Test", role="user", id="test-123")
        d = turn.to_dict()
        
        assert d["id"] == "test-123"
        assert d["content"] == "Test"
        assert d["role"] == "user"
        assert "created_at" in d
        assert d["is_compressed"] is False
    
    def test_turn_from_dict(self):
        """Test Turn deserialization."""
        data = {
            "id": "test-456",
            "content": "Hello",
            "role": "assistant",
            "created_at": "2026-01-01T12:00:00",
            "metadata": {"source": "test"},
            "token_count": 10,
        }
        
        turn = Turn.from_dict(data)
        
        assert turn.id == "test-456"
        assert turn.content == "Hello"
        assert turn.role == "assistant"
        assert turn.token_count == 10
        assert turn.metadata["source"] == "test"
    
    def test_turn_from_dict_minimal(self):
        """Test Turn deserialization with minimal data."""
        data = {"content": "Hi", "role": "user"}
        turn = Turn.from_dict(data)
        
        assert turn.content == "Hi"
        assert turn.role == "user"
        assert turn.id is not None
    
    def test_turn_to_message_dict(self):
        """Test conversion to LLM message format."""
        turn = Turn(content="What is AI?", role="user")
        msg = turn.to_message_dict()
        
        assert msg == {"role": "user", "content": "What is AI?"}


class TestTopic:
    """Tests for Topic dataclass."""
    
    def test_create_topic_minimal(self):
        """Test creating a topic with minimal args."""
        topic = Topic()
        
        assert topic.id is not None
        assert topic.name == ""
        assert topic.turn_ids == []
        assert topic.embedding is None
    
    def test_create_topic_full(self):
        """Test creating a topic with all args."""
        topic = Topic(
            id="topic-1",
            name="VR Setup",
            turn_ids=["turn-1", "turn-2"],
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert topic.id == "topic-1"
        assert topic.name == "VR Setup"
        assert topic.turn_ids == ["turn-1", "turn-2"]
        assert topic.turn_count == 2
        assert topic.embedding == [0.1, 0.2, 0.3]
    
    def test_topic_turn_count(self):
        """Test turn count property."""
        topic = Topic(turn_ids=["a", "b", "c"])
        assert topic.turn_count == 3
    
    def test_topic_to_dict(self):
        """Test Topic serialization."""
        topic = Topic(id="t-1", name="Test Topic")
        d = topic.to_dict()
        
        assert d["id"] == "t-1"
        assert d["name"] == "Test Topic"
        assert "created_at" in d
    
    def test_topic_from_dict(self):
        """Test Topic deserialization."""
        data = {
            "id": "t-2",
            "name": "Games",
            "turn_ids": ["turn-x"],
            "created_at": "2026-02-01T10:00:00",
            "last_accessed": "2026-02-01T11:00:00",
        }
        
        topic = Topic.from_dict(data)
        
        assert topic.id == "t-2"
        assert topic.name == "Games"
        assert topic.turn_ids == ["turn-x"]


class TestCompressedSegment:
    """Tests for CompressedSegment."""
    
    def test_create_compressed_segment(self):
        """Test creating a compressed segment."""
        segment = CompressedSegment(
            summary="User asked about VR setup and got instructions.",
            original_turn_ids=["t1", "t2", "t3"],
            token_count=15,
        )
        
        assert "VR setup" in segment.summary
        assert len(segment.original_turn_ids) == 3
        assert segment.token_count == 15
        assert segment.created_at is not None
