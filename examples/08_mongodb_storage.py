"""MongoDB storage example.

This example shows how to use LLMem with MongoDB for
persistent conversation storage.

Requirements:
    pip install motor dnspython python-dotenv
"""

import asyncio
import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    # Get MongoDB URL from environment
    mongo_url = os.environ.get("MONGODB_URL")
    
    if not mongo_url:
        print("Set MONGODB_URL in .env file")
        print("Example: MONGODB_URL=mongodb://localhost:27017")
        print("Or Atlas: MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/")
        return
    
    # Try to import motor
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        print("Install motor: pip install motor dnspython")
        return
    
    from llmem import Memory
    from llmem.storage.mongo import MongoStorage
    
    print("=== LLMem with MongoDB ===\n")
    
    # Create MongoDB client
    try:
        client = AsyncIOMotorClient(mongo_url)
        # Test connection
        await client.admin.command('ping')
        print(f"Connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nCheck your MONGODB_URL in .env file")
        return
    
    try:
        # Get database and create storage
        db = client.llmem_demo
        storage = MongoStorage(db=db)
        memory = Memory(storage=storage)
        
        thread_id = "demo-thread-456"
        
        # Clear any existing data
        await memory._clear_async(thread_id)
        
        # Add some messages (use async methods)
        print("Adding messages...")
        await memory.add_async(
            "What programming languages do you know?",
            role="user",
            thread_id=thread_id
        )
        await memory.add_async(
            "I can help with many languages including Python, JavaScript, "
            "TypeScript, Java, Go, Rust, and more!",
            role="assistant",
            thread_id=thread_id
        )
        await memory.add_async(
            "Can you help me with Python?",
            role="user",
            thread_id=thread_id
        )
        await memory.add_async(
            "Absolutely! Python is great for beginners and professionals alike. "
            "What would you like to learn?",
            role="assistant",
            thread_id=thread_id
        )
        
        # Get context (use async)
        print("\nRetrieving context...")
        context = await memory.get_context_async(thread_id=thread_id)
        
        for msg in context:
            content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
            print(f"  [{msg['role']}]: {content}")
        
        # Check stats
        print("\n=== Statistics ===")
        stats = await memory._get_stats_async(thread_id)
        print(f"Turns: {stats['total_turns']}")
        print(f"Tokens: {stats['total_tokens']}")
        
        # Health check
        print("\n=== Health Check ===")
        health = await memory._check_health_async(thread_id)
        print(f"Status: {health.status.value}")
        print(f"Token usage: {health.token_usage:.1%}")
        
        # Demonstrate persistence
        print("\n=== Testing Persistence ===")
        print("Creating new Memory instance with same storage...")
        
        # New memory instance, same database
        storage2 = MongoStorage(db=db)
        memory2 = Memory(storage=storage2)
        context2 = await memory2.get_context_async(thread_id=thread_id)
        
        print(f"Retrieved {len(context2)} messages from new instance")
        print("Data persisted successfully!")
        
        # Cleanup
        print("\n=== Cleanup ===")
        await memory._clear_async(thread_id)
        print("Thread cleared")
        
    finally:
        client.close()
        print("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
