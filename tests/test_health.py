"""Tests for health monitoring."""

import pytest

from llmem.health import ContextHealth, HealthStatus, Recommendation


class TestContextHealth:
    """Tests for ContextHealth."""
    
    def test_healthy_status(self):
        """Test healthy status when under threshold."""
        health = ContextHealth.calculate(
            token_count=10000,
            max_tokens=100000,
            turn_count=20,
        )
        
        assert health.status == HealthStatus.HEALTHY
        assert health.recommendation == Recommendation.CONTINUE
        assert health.token_usage == 0.1
    
    def test_warning_status(self):
        """Test warning status at 70% usage."""
        health = ContextHealth.calculate(
            token_count=75000,
            max_tokens=100000,
            turn_count=100,
        )
        
        assert health.status == HealthStatus.WARNING
        assert health.recommendation == Recommendation.COMPRESS_SOON
    
    def test_critical_status(self):
        """Test critical status at 85% usage."""
        health = ContextHealth.calculate(
            token_count=90000,
            max_tokens=100000,
            turn_count=150,
        )
        
        assert health.status == HealthStatus.CRITICAL
        assert health.recommendation == Recommendation.COMPRESS_NOW
    
    def test_overflow_status(self):
        """Test overflow status at 95% usage."""
        health = ContextHealth.calculate(
            token_count=98000,
            max_tokens=100000,
            turn_count=200,
        )
        
        assert health.status == HealthStatus.OVERFLOW
        assert health.recommendation == Recommendation.SUGGEST_RESET
    
    def test_zero_max_tokens(self):
        """Test handling of zero max_tokens."""
        health = ContextHealth.calculate(
            token_count=100,
            max_tokens=0,
            turn_count=5,
        )
        
        assert health.token_usage == 0.0
    
    def test_full_metadata(self):
        """Test all fields are populated."""
        health = ContextHealth.calculate(
            token_count=50000,
            max_tokens=100000,
            turn_count=80,
            topic_coherence=0.85,
            compression_ratio=0.3,
            topics_detected=5,
            compressions_performed=2,
        )
        
        assert health.token_count == 50000
        assert health.max_tokens == 100000
        assert health.turn_count == 80
        assert health.topic_coherence == 0.85
        assert health.compression_ratio == 0.3
        assert health.topics_detected == 5
        assert health.compressions_performed == 2
    
    def test_to_dict(self):
        """Test serialization to dict."""
        health = ContextHealth.calculate(
            token_count=1000,
            max_tokens=10000,
            turn_count=10,
        )
        
        d = health.to_dict()
        
        assert d["token_count"] == 1000
        assert d["max_tokens"] == 10000
        assert d["turn_count"] == 10
        assert d["status"] == "healthy"
        assert d["recommendation"] == "continue"


class TestHealthStatus:
    """Tests for HealthStatus enum."""
    
    def test_status_values(self):
        """Test status string values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.OVERFLOW.value == "overflow"


class TestRecommendation:
    """Tests for Recommendation enum."""
    
    def test_recommendation_values(self):
        """Test recommendation string values."""
        assert Recommendation.CONTINUE.value == "continue"
        assert Recommendation.COMPRESS_SOON.value == "compress_soon"
        assert Recommendation.COMPRESS_NOW.value == "compress_now"
        assert Recommendation.SUGGEST_RESET.value == "suggest_reset"
