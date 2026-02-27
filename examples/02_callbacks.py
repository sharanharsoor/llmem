"""Callbacks example for LLMem.

This example demonstrates how to use callbacks for:
- Compression events
- Health changes
- Topic changes (future feature)
"""

from llmem import Memory
from llmem.health import HealthStatus


def on_compress(info: dict):
    """Called when compression occurs."""
    print(f"\n[CALLBACK] Compression performed!")
    print(f"  Removed turns: {info.get('removed_turns', 0)}")
    print(f"  Remaining turns: {info.get('remaining_turns', 0)}")


def on_health_change(health):
    """Called when health status changes."""
    status_emoji = {
        HealthStatus.HEALTHY: "✓",
        HealthStatus.WARNING: "⚠",
        HealthStatus.CRITICAL: "!",
        HealthStatus.OVERFLOW: "✗",
    }
    
    emoji = status_emoji.get(health.status, "?")
    print(f"\n[CALLBACK] Health changed: {emoji} {health.status.value}")
    print(f"  Token usage: {health.token_usage:.1%}")
    print(f"  Recommendation: {health.recommendation.value}")


def on_topic_change(old_topic: str, new_topic: str):
    """Called when topic changes (future feature)."""
    print(f"\n[CALLBACK] Topic changed: {old_topic} → {new_topic}")


def main():
    # Create memory with callbacks
    memory = Memory(
        max_tokens=1000,  # Low limit to trigger compression
        compression_threshold=0.5,  # Compress at 50% usage
        on_compress=on_compress,
        on_health_change=on_health_change,
        on_topic_change=on_topic_change,
    )
    
    print("=== Adding messages (watch for callbacks) ===")
    
    # Add messages until compression triggers
    messages = [
        "What is artificial intelligence?",
        "AI is a field of computer science focused on creating intelligent machines.",
        "How does machine learning work?",
        "Machine learning uses algorithms to learn patterns from data.",
        "What are neural networks?",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Can you explain deep learning?",
        "Deep learning uses multiple layers of neural networks to learn complex patterns.",
        "What is natural language processing?",
        "NLP enables computers to understand and process human language.",
    ]
    
    for i, msg in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        print(f"\nAdding message {i+1}: {msg[:40]}...")
        memory.add(msg, role=role)
    
    print("\n=== Final state ===")
    stats = memory.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Compressions: {stats['compression_count']}")


if __name__ == "__main__":
    main()
