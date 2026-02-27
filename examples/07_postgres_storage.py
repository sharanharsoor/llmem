"""PostgreSQL storage example.

This example shows how to use LLMem with PostgreSQL for
persistent conversation storage.

Requirements:
    pip install asyncpg python-dotenv
"""

import asyncio
import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    # Get database URL from environment
    db_url = os.environ.get("DATABASE_URL")
    
    if not db_url:
        print("Set DATABASE_URL in .env file")
        print("Example: DATABASE_URL=postgresql://user:pass@localhost:5432/dbname")
        return
    
    # Try to import asyncpg
    try:
        import asyncpg
    except ImportError:
        print("Install asyncpg: pip install asyncpg")
        return
    
    from llmem import Memory
    from llmem.storage.postgres import PostgresStorage
    
    print("=== LLMem with PostgreSQL ===\n")
    
    # Create connection pool
    try:
        pool = await asyncpg.create_pool(db_url)
        print(f"Connected to PostgreSQL")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nCheck your DATABASE_URL in .env file")
        return
    
    try:
        # Create storage and memory
        storage = PostgresStorage(pool=pool)
        memory = Memory(storage=storage)
        
        thread_id = "demo-thread-123"
        
        # Clear any existing data first
        await memory._clear_async(thread_id)
        
        # Add some messages (use async methods)
        print("Adding messages...")
        await memory.add_async("Hello!", role="user", thread_id=thread_id)
        await memory.add_async(
            "Hi! How can I help you today?",
            role="assistant",
            thread_id=thread_id
        )
        await memory.add_async(
            "I need help with my account",
            role="user",
            thread_id=thread_id
        )
        await memory.add_async(
            "I'd be happy to help with your account. What do you need?",
            role="assistant",
            thread_id=thread_id
        )
        
        # Get context (use async)
        print("\nRetrieving context...")
        context = await memory.get_context_async(thread_id=thread_id)
        
        for msg in context:
            print(f"  [{msg['role']}]: {msg['content']}")
        
        # Check stats
        print("\n=== Statistics ===")
        stats = await memory._get_stats_async(thread_id)
        print(f"Turns: {stats['total_turns']}")
        print(f"Tokens: {stats['total_tokens']}")
        
        # Demonstrate persistence
        print("\n=== Testing Persistence ===")
        print("Creating new Memory instance with same storage...")
        
        # New memory instance, same pool
        storage2 = PostgresStorage(pool=pool)
        memory2 = Memory(storage=storage2)
        context2 = await memory2.get_context_async(thread_id=thread_id)
        
        print(f"Retrieved {len(context2)} messages from new instance")
        print("Data persisted successfully!")
        
        # Cleanup
        print("\n=== Cleanup ===")
        await memory._clear_async(thread_id)
        print("Thread cleared")
        
    finally:
        await pool.close()
        print("Connection pool closed")


if __name__ == "__main__":
    asyncio.run(main())
