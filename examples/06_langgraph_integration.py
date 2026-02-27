"""LangGraph integration example.

This example shows how to use LLMem alongside LangGraph
for intelligent memory management in agentic workflows.

Requirements:
    pip install langchain-google-genai langgraph python-dotenv
"""

import os
import asyncio
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("Set GOOGLE_API_KEY in .env file")
        return
    
    # Try to import dependencies
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        print("Install dependencies: pip install langchain-google-genai langgraph")
        return
    
    # Import LLMem
    from llmem import Memory
    
    print("=== LLMem + LangGraph (with Gemini) ===\n")
    
    # Create LangGraph's checkpointer for agent state
    checkpointer = MemorySaver()
    
    # Create LLMem for conversation memory management
    # This tracks conversation turns separately for compression/optimization
    llmem = Memory(
        max_tokens=50000,
        compression_threshold=0.7,
    )
    
    # Create model
    model = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=api_key,
        temperature=0,
    )
    
    # Simple tool for demonstration
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"The weather in {city} is sunny and 72°F"
    
    agent = create_react_agent(
        model=model,
        tools=[get_weather],
        checkpointer=checkpointer,
    )
    
    thread_id = "user-demo-thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Helper to run agent and track with LLMem
    async def chat(user_message: str) -> str:
        # Track in LLMem
        llmem.add(user_message, role="user", thread_id=thread_id)
        
        # Run agent
        result = await agent.ainvoke(
            {"messages": [("user", user_message)]},
            config=config,
        )
        
        # Get response
        response = result['messages'][-1].content
        
        # Track response in LLMem
        llmem.add(response, role="assistant", thread_id=thread_id)
        
        return response
    
    # Run conversations
    print("User: What's the weather in San Francisco?")
    response1 = await chat("What's the weather in San Francisco?")
    print(f"Agent: {response1}\n")
    
    print("User: How about New York?")
    response2 = await chat("How about New York?")
    print(f"Agent: {response2}\n")
    
    print("User: What cities did I ask about?")
    response3 = await chat("What cities did I ask about?")
    print(f"Agent: {response3}\n")
    
    # Check LLMem health
    print("=== LLMem Health ===")
    health = llmem.check_health(thread_id=thread_id)
    print(f"Status: {health.status.value}")
    print(f"Token usage: {health.token_usage:.1%}")
    print(f"Turn count: {health.turn_count}")
    
    # Get optimized context (LLMem feature)
    print("\n=== LLMem Optimized Context ===")
    context = llmem.get_context(thread_id=thread_id)
    print(f"Context has {len(context)} messages")
    for msg in context:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")
    
    # Statistics
    print("\n=== Statistics ===")
    stats = llmem.get_stats(thread_id=thread_id)
    print(f"Total turns: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
