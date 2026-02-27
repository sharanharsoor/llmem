"""Token counting utilities for LLMem."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Union

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


DEFAULT_MODEL = "cl100k_base"


@lru_cache(maxsize=8)
def get_tokenizer(model: str = DEFAULT_MODEL):
    """Get tokenizer for a model.
    
    Args:
        model: Model name or encoding (e.g., "cl100k_base")
        
    Returns:
        Tokenizer instance
        
    Raises:
        ImportError: If tiktoken is not installed
    """
    if not TIKTOKEN_AVAILABLE:
        raise ImportError(
            "tiktoken is required for token counting. "
            "Install with: pip install tiktoken"
        )
    
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(
    text: Union[str, List[str]], 
    model: str = DEFAULT_MODEL
) -> int:
    """Count tokens in text.
    
    Args:
        text: Single string or list of strings
        model: Model name for tokenizer
        
    Returns:
        Total token count
    """
    if not TIKTOKEN_AVAILABLE:
        return _estimate_tokens(text)
    
    tokenizer = get_tokenizer(model)
    
    if isinstance(text, str):
        return len(tokenizer.encode(text))
    
    return sum(len(tokenizer.encode(t)) for t in text)


def _estimate_tokens(text: Union[str, List[str]]) -> int:
    """Fallback token estimation without tiktoken.
    
    Uses ~4 characters per token heuristic.
    """
    if isinstance(text, str):
        return len(text) // 4
    
    return sum(len(t) // 4 for t in text)


def count_message_tokens(
    messages: List[dict],
    model: str = DEFAULT_MODEL
) -> int:
    """Count tokens for a list of messages.
    
    Accounts for message overhead (role, formatting).
    
    Args:
        messages: List of {"role": str, "content": str} dicts
        model: Model name for tokenizer
        
    Returns:
        Total token count including overhead
    """
    # Base overhead per message (role + formatting)
    OVERHEAD_PER_MESSAGE = 4
    
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        total += count_tokens(content, model)
        total += count_tokens(role, model)
        total += OVERHEAD_PER_MESSAGE
    
    # Reply priming
    total += 3
    
    return total
