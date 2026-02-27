"""Core data types for LLMem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Turn:
    """A single conversation turn (message)."""
    
    content: str
    role: str  # "user", "assistant", "system", "tool"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    topic_id: Optional[str] = None
    is_compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count,
            "topic_id": self.topic_id,
            "is_compressed": self.is_compressed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Turn:
        """Create Turn from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
            
        return cls(
            id=data.get("id", str(uuid4())),
            content=data["content"],
            role=data["role"],
            created_at=created_at,
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
            topic_id=data.get("topic_id"),
            is_compressed=data.get("is_compressed", False),
        )
    
    def to_message_dict(self) -> Dict[str, str]:
        """Convert to LLM message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Topic:
    """A detected conversation topic."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    turn_ids: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def turn_count(self) -> int:
        """Number of turns in this topic."""
        return len(self.turn_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "turn_ids": self.turn_ids,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Topic:
        """Create Topic from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            name=data.get("name", ""),
            turn_ids=data.get("turn_ids", []),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.now(timezone.utc),
        )


@dataclass 
class CompressedSegment:
    """A compressed portion of conversation history."""
    
    summary: str
    original_turn_ids: List[str]
    token_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    topic_id: Optional[str] = None
