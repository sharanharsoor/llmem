"""Using LLMem with Google Gemini.

This example shows how to integrate LLMem with Google's Gemini API
for intelligent conversation management.

Requirements:
    pip install google-generativeai python-dotenv
"""

import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from llmem import Memory


def main():
    # Get config from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    
    if not api_key:
        print("Set GOOGLE_API_KEY in .env file")
        return
    
    # Import Gemini
    try:
        import google.generativeai as genai
    except ImportError:
        print("Install Google Generative AI: pip install google-generativeai")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Initialize LLMem
    memory = Memory(
        max_tokens=100000,  # Gemini has large context
        # Token counting uses cl100k_base encoding by default
    )
    
    print("=== LLMem + Google Gemini ===")
    print(f"Model: {model_name}")
    print("Type 'quit' to exit, 'health' to check memory health\n")
    
    # System instruction
    system_prompt = "You are a helpful assistant. Be concise in your responses (2-3 sentences max)."
    
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
        
        if user_input.lower() == "stats":
            stats = memory.get_stats()
            print(f"\nStats: {stats}\n")
            continue
        
        # Add user message to memory
        memory.add(user_input, role="user")
        
        # Get optimized context from LLMem
        context = memory.get_context()
        
        # Build prompt with context
        conversation_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in context
        ])
        
        full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_history}\n\nASSISTANT:"
        
        # Call Gemini
        try:
            response = model.generate_content(full_prompt)
            assistant_message = response.text.strip()
            
            # Add assistant response to memory
            memory.add(assistant_message, role="assistant")
            
            print(f"\nGemini: {assistant_message}\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")
    
    # Final stats
    print("\n=== Session Stats ===")
    stats = memory.get_stats()
    print(f"Total turns: {stats['total_turns']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Compressions: {stats['compression_count']}")
    
    health = memory.check_health()
    print(f"Final health: {health.status.value}")


if __name__ == "__main__":
    main()
