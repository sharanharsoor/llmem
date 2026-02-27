"""Using LLMem with OpenAI.

This example shows how to integrate LLMem with OpenAI's API
for intelligent conversation management.

Requirements:
    pip install openai python-dotenv
"""

import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from llmem import Memory


def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Set OPENAI_API_KEY in .env file")
        return
    
    # Import OpenAI
    try:
        from openai import OpenAI
    except ImportError:
        print("Install OpenAI: pip install openai")
        return
    
    # Initialize
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    memory = Memory(
        max_tokens=4000,  # Reserve tokens for response
    )
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Be concise in your responses."
    }
    
    print("=== Chat with OpenAI (LLMem manages context) ===")
    print("Type 'quit' to exit, 'health' to check memory health\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "health":
            health = memory.check_health()
            print(f"\nMemory Health:")
            print(f"  Status: {health.status.value}")
            print(f"  Usage: {health.token_usage:.1%}")
            print(f"  Turns: {health.turn_count}")
            print(f"  Recommendation: {health.recommendation.value}\n")
            continue
        
        # Add user message to memory
        memory.add(user_input, role="user")
        
        # Get optimized context from LLMem
        context = memory.get_context()
        
        # Build messages for API call
        messages = [system_prompt] + context
        
        # Call OpenAI
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to memory
            memory.add(assistant_message, role="assistant")
            
            print(f"\nAssistant: {assistant_message}\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")
    
    # Final stats
    print("\n=== Session stats ===")
    stats = memory.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Compressions: {stats['compression_count']}")


if __name__ == "__main__":
    main()
