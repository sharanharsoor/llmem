"""Context health monitoring for LLMem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class HealthStatus(Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERFLOW = "overflow"


class Recommendation(Enum):
    """Recommended actions based on health."""
    
    CONTINUE = "continue"
    COMPRESS_SOON = "compress_soon"
    COMPRESS_NOW = "compress_now"
    SUGGEST_RESET = "suggest_reset"


@dataclass
class ContextHealth:
    """Health metrics for conversation context."""
    
    token_usage: float  # 0.0 to 1.0 (current / max)
    token_count: int
    max_tokens: int
    turn_count: int
    topic_coherence: float  # 0.0 to 1.0 (how related recent turns are)
    compression_ratio: float  # Current compression level
    status: HealthStatus
    recommendation: Recommendation
    topics_detected: int = 0
    compressions_performed: int = 0
    
    @classmethod
    def calculate(
        cls,
        token_count: int,
        max_tokens: int,
        turn_count: int,
        topic_coherence: float = 1.0,
        compression_ratio: float = 0.0,
        topics_detected: int = 0,
        compressions_performed: int = 0,
    ) -> ContextHealth:
        """Calculate health from metrics."""
        token_usage = token_count / max_tokens if max_tokens > 0 else 0.0
        
        # Determine status
        if token_usage >= 0.95:
            status = HealthStatus.OVERFLOW
            recommendation = Recommendation.SUGGEST_RESET
        elif token_usage >= 0.85:
            status = HealthStatus.CRITICAL
            recommendation = Recommendation.COMPRESS_NOW
        elif token_usage >= 0.70:
            status = HealthStatus.WARNING
            recommendation = Recommendation.COMPRESS_SOON
        else:
            status = HealthStatus.HEALTHY
            recommendation = Recommendation.CONTINUE
        
        return cls(
            token_usage=token_usage,
            token_count=token_count,
            max_tokens=max_tokens,
            turn_count=turn_count,
            topic_coherence=topic_coherence,
            compression_ratio=compression_ratio,
            status=status,
            recommendation=recommendation,
            topics_detected=topics_detected,
            compressions_performed=compressions_performed,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_usage": self.token_usage,
            "token_count": self.token_count,
            "max_tokens": self.max_tokens,
            "turn_count": self.turn_count,
            "topic_coherence": self.topic_coherence,
            "compression_ratio": self.compression_ratio,
            "status": self.status.value,
            "recommendation": self.recommendation.value,
            "topics_detected": self.topics_detected,
            "compressions_performed": self.compressions_performed,
        }
