"""End-to-end agent test with all storage backends.

This script tests LLMem with a real LangGraph agent across
all supported storage backends to verify everything works correctly.

Requirements:
    pip install langchain-google-genai langgraph asyncpg motor python-dotenv
"""

import os
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def test_with_storage(storage_name: str, storage, api_key: str) -> bool:
    """Test LLMem + LangGraph agent with a specific storage backend."""
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        print("  Install: pip install langchain-google-genai langgraph")
        return False
    
    from llmem import Memory
    
    print(f"\n{'='*50}")
    print(f"Testing with: {storage_name}")
    print('='*50)
    
    # Create LLMem with the specified storage
    llmem = Memory(
        storage=storage,
        max_tokens=50000,
        compression_threshold=0.7,
    )
    
    # Create model
    model = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=api_key,
        temperature=0,
    )
    
    # Simple tools for testing
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"
    
    def get_time(timezone: str) -> str:
        """Get current time in a timezone."""
        return f"Current time in {timezone}: 3:30 PM"
    
    # Create agent
    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=model,
        tools=[get_weather, get_time],
        checkpointer=checkpointer,
    )
    
    thread_id = f"e2e-test-{storage_name.lower().replace(' ', '-')}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Clean up any leftover data from previous runs
    await llmem._clear_async(thread_id=thread_id)
    
    async def chat(user_message: str) -> str:
        """Send message to agent and track with LLMem."""
        await llmem.add_async(user_message, role="user", thread_id=thread_id)
        
        result = await agent.ainvoke(
            {"messages": [("user", user_message)]},
            config=config,
        )
        
        response = result['messages'][-1].content
        # Handle case where response is a list of content blocks
        if isinstance(response, list):
            response = " ".join(str(item) for item in response)
        response = str(response)
        
        await llmem.add_async(response, role="assistant", thread_id=thread_id)
        return response
    
    # Test 1: Basic conversation
    print("\n[Test 1] Basic conversation")
    response = await chat("What's the weather in Tokyo?")
    print(f"  User: What's the weather in Tokyo?")
    print(f"  Agent: {response[:100]}...")
    
    # Test 2: Follow-up (tests context)
    print("\n[Test 2] Follow-up question")
    response = await chat("What about the time there?")
    print(f"  User: What about the time there?")
    print(f"  Agent: {response[:100]}...")
    
    # Test 3: Memory recall
    print("\n[Test 3] Memory recall")
    response = await chat("What city did I ask about first?")
    print(f"  User: What city did I ask about first?")
    print(f"  Agent: {response[:100]}...")
    
    # Verify Tokyo is mentioned (memory working)
    tokyo_recalled = "tokyo" in response.lower()
    print(f"  Memory recall: {'PASS' if tokyo_recalled else 'FAIL'}")
    
    # Test 4: Health check
    print("\n[Test 4] Health monitoring")
    health = await llmem._check_health_async(thread_id=thread_id)
    print(f"  Status: {health.status.value}")
    print(f"  Token usage: {health.token_usage:.1%}")
    print(f"  Turn count: {health.turn_count}")
    health_ok = health.turn_count == 6  # 3 user + 3 assistant
    print(f"  Turn count correct: {'PASS' if health_ok else 'FAIL'}")
    
    # Test 5: Context retrieval
    print("\n[Test 5] Context retrieval")
    context = await llmem._get_context_async(thread_id=thread_id, max_tokens=None)
    print(f"  Context messages: {len(context)}")
    context_ok = len(context) == 6
    print(f"  Context count correct: {'PASS' if context_ok else 'FAIL'}")
    
    # Test 6: Statistics
    print("\n[Test 6] Statistics")
    stats = await llmem._get_stats_async(thread_id=thread_id)
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    
    # Clean up
    await llmem._clear_async(thread_id=thread_id)
    print("\n  Cleaned up test data")
    
    # Overall result
    passed = tokyo_recalled and health_ok and context_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    
    return passed


async def main():
    """Run end-to-end tests with all available storage backends."""
    
    print("=" * 60)
    print("LLMem End-to-End Agent Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\nError: Set GOOGLE_API_KEY in .env file")
        return
    
    results = {}
    
    # Test 1: In-Memory Storage (always available)
    print("\n[Backend 1/3] In-Memory Storage")
    from llmem.storage.memory import InMemoryStorage
    storage = InMemoryStorage()
    results["In-Memory"] = await test_with_storage("In-Memory", storage, api_key)
    
    # Test 2: PostgreSQL Storage (if configured)
    print("\n[Backend 2/3] PostgreSQL Storage")
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import asyncpg
            from llmem.storage.postgres import PostgresStorage
            
            pool = await asyncpg.create_pool(db_url)
            storage = PostgresStorage(pool=pool)
            
            results["PostgreSQL"] = await test_with_storage("PostgreSQL", storage, api_key)
            
            await pool.close()
        except Exception as e:
            print(f"  Error: {e}")
            results["PostgreSQL"] = False
    else:
        print("  Skipped: DATABASE_URL not set")
        results["PostgreSQL"] = None
    
    # Test 3: MongoDB Storage (if configured)
    print("\n[Backend 3/3] MongoDB Storage")
    mongo_url = os.environ.get("MONGODB_URL")
    if mongo_url:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from llmem.storage.mongo import MongoStorage
            
            client = AsyncIOMotorClient(mongo_url)
            db = client.llmem_test
            storage = MongoStorage(db=db)
            
            results["MongoDB"] = await test_with_storage("MongoDB", storage, api_key)
            
            client.close()
        except Exception as e:
            print(f"  Error: {e}")
            results["MongoDB"] = False
    else:
        print("  Skipped: MONGODB_URL not set")
        results["MongoDB"] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for backend, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {backend}: {status}")
    
    # Overall
    tested = [r for r in results.values() if r is not None]
    passed = all(tested) if tested else False
    print(f"\n  Overall: {'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    
    return passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
