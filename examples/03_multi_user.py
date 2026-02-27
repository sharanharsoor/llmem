"""Multi-user example for LLMem.

This example demonstrates thread isolation for multi-user scenarios:
- Different users have separate memory contexts
- Messages don't mix between threads
- Each thread can be managed independently
"""

from llmem import Memory


def main():
    # Create a single memory instance (shared across users)
    memory = Memory()
    
    # User IDs (in production, these come from authentication)
    user_a = "user-alice-session-123"
    user_b = "user-bob-session-456"
    
    print("=== User A conversation ===")
    
    memory.add(
        "I want to learn Python programming",
        role="user",
        thread_id=user_a
    )
    memory.add(
        "Great choice! Python is beginner-friendly. Start with variables and data types.",
        role="assistant",
        thread_id=user_a
    )
    memory.add(
        "What are variables?",
        role="user",
        thread_id=user_a
    )
    memory.add(
        "Variables are containers for storing data values. Example: name = 'Alice'",
        role="assistant",
        thread_id=user_a
    )
    
    print("=== User B conversation ===")
    
    memory.add(
        "How do I cook pasta?",
        role="user",
        thread_id=user_b
    )
    memory.add(
        "Boil water, add salt, cook pasta for 8-10 minutes, drain and serve!",
        role="assistant",
        thread_id=user_b
    )
    memory.add(
        "What sauce goes well with spaghetti?",
        role="user",
        thread_id=user_b
    )
    memory.add(
        "Classic marinara, creamy alfredo, or simple garlic and olive oil!",
        role="assistant",
        thread_id=user_b
    )
    
    print("\n=== Verify isolation ===")
    
    # Get context for each user
    context_a = memory.get_context(thread_id=user_a)
    context_b = memory.get_context(thread_id=user_b)
    
    print(f"\nUser A context ({len(context_a)} messages):")
    for msg in context_a:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")
    
    print(f"\nUser B context ({len(context_b)} messages):")
    for msg in context_b:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")
    
    # Verify stats are separate
    print("\n=== Stats per user ===")
    
    stats_a = memory.get_stats(thread_id=user_a)
    stats_b = memory.get_stats(thread_id=user_b)
    
    print(f"User A: {stats_a['total_turns']} turns, {stats_a['total_tokens']} tokens")
    print(f"User B: {stats_b['total_turns']} turns, {stats_b['total_tokens']} tokens")
    
    # Clear one user without affecting the other
    print("\n=== Clear User A only ===")
    memory.clear(thread_id=user_a)
    
    stats_a_after = memory.get_stats(thread_id=user_a)
    stats_b_after = memory.get_stats(thread_id=user_b)
    
    print(f"User A after clear: {stats_a_after['total_turns']} turns")
    print(f"User B after clear: {stats_b_after['total_turns']} turns (unchanged)")


if __name__ == "__main__":
    main()
