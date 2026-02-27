"""Tests for token counting utilities."""

import pytest

from llmem.utils.tokens import count_tokens, count_message_tokens, _estimate_tokens


class TestCountTokens:
    """Tests for count_tokens function."""
    
    def test_count_simple_string(self):
        """Test counting tokens in a simple string."""
        count = count_tokens("Hello, world!")
        
        # With tiktoken, this should be around 4 tokens
        assert count > 0
        assert count < 10
    
    def test_count_empty_string(self):
        """Test counting empty string."""
        count = count_tokens("")
        assert count == 0
    
    def test_count_list_of_strings(self):
        """Test counting tokens in multiple strings."""
        count = count_tokens(["Hello", "World"])
        
        # Should be sum of individual counts
        assert count > 0
        assert count == count_tokens("Hello") + count_tokens("World")
    
    def test_count_long_text(self):
        """Test counting longer text."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = count_tokens(text)
        
        # Should scale with text length
        assert count > 50


class TestEstimateTokens:
    """Tests for fallback token estimation."""
    
    def test_estimate_simple(self):
        """Test estimation for simple text."""
        # ~4 chars per token
        estimate = _estimate_tokens("Hello world")  # 11 chars
        
        assert estimate == 11 // 4
    
    def test_estimate_list(self):
        """Test estimation for list of strings."""
        estimate = _estimate_tokens(["Hello", "World"])  # 5 + 5 = 10 chars
        
        assert estimate == 10 // 4


class TestCountMessageTokens:
    """Tests for message token counting."""
    
    def test_count_single_message(self):
        """Test counting single message."""
        messages = [{"role": "user", "content": "Hello"}]
        count = count_message_tokens(messages)
        
        # Content + role + overhead
        assert count > count_tokens("Hello")
    
    def test_count_conversation(self):
        """Test counting a conversation."""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
            {"role": "user", "content": "Can you give examples?"},
        ]
        
        count = count_message_tokens(messages)
        
        # Should be substantial
        assert count > 15
    
    def test_empty_messages(self):
        """Test counting empty message list."""
        count = count_message_tokens([])
        
        # Just the reply priming overhead
        assert count == 3
    
    def test_overhead_included(self):
        """Test that message overhead is included."""
        messages = [{"role": "user", "content": "Hi"}]
        
        content_only = count_tokens("Hi")
        with_overhead = count_message_tokens(messages)
        
        # With overhead should be more than content alone
        assert with_overhead > content_only
