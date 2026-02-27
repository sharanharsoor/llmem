"""LangChain integration example.

This example shows how to use LLMem with LangChain for
intelligent conversation memory management.

Requirements:
    pip install langchain-google-genai python-dotenv
"""

import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def main():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("Set GOOGLE_API_KEY in .env file")
        return
    
    # Try to import LangChain with Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    except ImportError:
        print("Install dependencies: pip install langchain-google-genai")
        return
    
    # Import LLMem
    from llmem import Memory
    
    print("=== LLMem + LangChain (with Gemini) ===\n")
    
    # Create LLMem memory
    memory = Memory(
        max_tokens=100000,
        compression_threshold=0.7,
    )
    
    # Create LangChain LLM
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=api_key,
        temperature=0.7,
    )
    
    # System message
    system_msg = SystemMessage(content="You are a helpful assistant. Be concise (2-3 sentences).")
    
    def chat(user_input: str) -> str:
        """Send message and get response, using LLMem for memory."""
        # Add user message to LLMem
        memory.add(user_input, role="user")
        
        # Get optimized context from LLMem
        context = memory.get_context()
        
        # Convert to LangChain messages
        messages = [system_msg]
        for msg in context:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Call LLM
        response = llm.invoke(messages)
        assistant_msg = response.content
        
        # Add assistant response to LLMem
        memory.add(assistant_msg, role="assistant")
        
        return assistant_msg
    
    # Have a conversation
    print("Starting conversation...\n")
    
    response1 = chat("Hi! I'm learning about machine learning.")
    print(f"User: Hi! I'm learning about machine learning.")
    print(f"Assistant: {response1}\n")
    
    response2 = chat("What's the difference between ML and AI?")
    print(f"User: What's the difference between ML and AI?")
    print(f"Assistant: {response2}\n")
    
    response3 = chat("Can you summarize what we talked about?")
    print(f"User: Can you summarize what we talked about?")
    print(f"Assistant: {response3}\n")
    
    # Check LLMem health
    print("=== LLMem Health Check ===")
    health = memory.check_health()
    print(f"Status: {health.status.value}")
    print(f"Token usage: {health.token_usage:.1%}")
    print(f"Turn count: {health.turn_count}")
    
    # Get stats
    print("\n=== Statistics ===")
    stats = memory.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    main()
