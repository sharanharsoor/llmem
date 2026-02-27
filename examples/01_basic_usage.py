"""Basic LLMem usage example.

This example demonstrates the core functionality of LLMem:
- Adding conversation turns
- Getting optimized context
- Checking health
- Getting statistics
"""

from llmem import Memory


def main():
    # Create memory with default settings (in-memory storage)
    memory = Memory()
    
    # Simulate a conversation
    print("=== Adding conversation turns ===")
    
    memory.add("How do I setup my VR headset?", role="user")
    memory.add(
        "To setup your VR headset, follow these steps: "
        "1. Charge the headset fully. "
        "2. Download the companion app. "
        "3. Create an account. "
        "4. Follow the in-app setup wizard.",
        role="assistant"
    )
    
    memory.add("What games do you recommend?", role="user")
    memory.add(
        "Here are some popular VR games: "
        "Beat Saber for rhythm gaming, "
        "Half-Life: Alyx for adventure, "
        "Superhot VR for action.",
        role="assistant"
    )
    
    memory.add("How do I adjust the display settings?", role="user")
    memory.add(
        "To adjust display settings: "
        "1. Go to Settings > Display. "
        "2. Adjust brightness and contrast. "
        "3. Set your IPD (interpupillary distance). "
        "4. Enable or disable eye tracking.",
        role="assistant"
    )
    
    # Get conversation context
    print("\n=== Getting context ===")
    context = memory.get_context()
    
    print(f"Number of messages: {len(context)}")
    for msg in context:
        role = msg["role"].upper()
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  [{role}]: {content}")
    
    # Check health
    print("\n=== Health check ===")
    health = memory.check_health()
    
    print(f"Status: {health.status.value}")
    print(f"Token usage: {health.token_usage:.1%}")
    print(f"Token count: {health.token_count}")
    print(f"Turn count: {health.turn_count}")
    print(f"Recommendation: {health.recommendation.value}")
    
    # Get statistics
    print("\n=== Statistics ===")
    stats = memory.get_stats()
    
    print(f"Total turns: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Compression count: {stats['compression_count']}")
    
    # Clear memory
    print("\n=== Clearing memory ===")
    memory.clear()
    
    stats_after = memory.get_stats()
    print(f"Turns after clear: {stats_after['total_turns']}")


if __name__ == "__main__":
    main()
