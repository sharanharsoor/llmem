"""LangChain memory integration for LLMem.

Provides a drop-in replacement for deprecated LangChain memory classes.

Requires: pip install llmem[langchain]

Example:
    >>> from llmem.integrations.langchain import LangChainMemory
    >>> from langchain.chains import ConversationChain
    >>> from langchain.llms import OpenAI
    >>> 
    >>> memory = LangChainMemory()
    >>> chain = ConversationChain(llm=OpenAI(), memory=memory)
    >>> chain.predict(input="Hello!")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llmem import Memory
from llmem.storage.base import StorageBackend


logger = logging.getLogger("llmem.integrations.langchain")


try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.memory import BaseMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object
    BaseMemory = object
    BaseMessage = object


def _check_langchain():
    """Check if LangChain is available."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install llmem[langchain]"
        )


class LLMemChatHistory(BaseChatMessageHistory):
    """LLMem-backed chat history for LangChain.
    
    Implements BaseChatMessageHistory using LLMem for storage.
    This enables using LLMem with any LangChain component that
    accepts chat history.
    """
    
    def __init__(
        self,
        memory: Memory,
        thread_id: str = "__default__",
    ):
        """Initialize with LLMem Memory instance.
        
        Args:
            memory: LLMem Memory instance
            thread_id: Thread ID for isolation
        """
        _check_langchain()
        self._memory = memory
        self._thread_id = thread_id
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        
        context = self._memory.get_context(thread_id=self._thread_id)
        
        messages = []
        for msg in context:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        
        return messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"
        
        self._memory.add(
            content=message.content,
            role=role,
            thread_id=self._thread_id,
        )
    
    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self._memory.add(message, role="user", thread_id=self._thread_id)
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self._memory.add(message, role="assistant", thread_id=self._thread_id)
    
    def clear(self) -> None:
        """Clear the chat history."""
        self._memory.clear(thread_id=self._thread_id)


class LangChainMemory(BaseMemory):
    """Drop-in replacement for deprecated LangChain memory.
    
    This class provides compatibility with LangChain's memory interface
    while using LLMem's intelligent memory management under the hood.
    
    Features over standard LangChain memory:
    - Topic-aware compression
    - Token tracking and health monitoring
    - Automatic compression when needed
    
    Example:
        >>> memory = LangChainMemory()
        >>> chain = ConversationChain(llm=llm, memory=memory)
    """
    
    # Memory variable names (LangChain convention)
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    return_messages: bool = False
    
    def __init__(
        self,
        memory: Optional[Memory] = None,
        storage: Optional[StorageBackend] = None,
        thread_id: str = "__default__",
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        **kwargs,
    ):
        """Initialize LangChainMemory.
        
        Args:
            memory: Optional existing Memory instance
            storage: Optional storage backend (creates new Memory)
            thread_id: Thread ID for multi-user isolation
            memory_key: Key name for memory in chain inputs
            input_key: Key name for human input
            output_key: Key name for AI output
            return_messages: Return Message objects instead of string
        """
        _check_langchain()
        
        super().__init__(**kwargs)
        
        self._memory = memory or Memory(storage=storage)
        self._thread_id = thread_id
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        
        # Create chat history wrapper
        self._chat_history = LLMemChatHistory(
            memory=self._memory,
            thread_id=thread_id,
        )
    
    @property
    def memory_variables(self) -> List[str]:
        """Memory variables provided to the chain."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory for use in chain.
        
        Args:
            inputs: Current chain inputs (may contain query for relevance)
            
        Returns:
            Dict with memory_key containing conversation history
        """
        if self.return_messages:
            return {self.memory_key: self._chat_history.messages}
        
        # Return as formatted string
        messages = self._chat_history.messages
        history_str = ""
        
        for msg in messages:
            from langchain_core.messages import AIMessage, HumanMessage
            
            if isinstance(msg, HumanMessage):
                history_str += f"{self.human_prefix}: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"{self.ai_prefix}: {msg.content}\n"
            else:
                history_str += f"{msg.content}\n"
        
        return {self.memory_key: history_str.strip()}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation context.
        
        Called by LangChain chains after each interaction.
        
        Args:
            inputs: User inputs
            outputs: AI outputs
        """
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if input_str:
            self._memory.add(input_str, role="user", thread_id=self._thread_id)
        
        if output_str:
            self._memory.add(output_str, role="assistant", thread_id=self._thread_id)
        
        logger.debug(f"Saved context: input={len(input_str)}chars, output={len(output_str)}chars")
    
    def clear(self) -> None:
        """Clear memory."""
        self._memory.clear(thread_id=self._thread_id)
    
    # Additional LLMem features
    
    def check_health(self):
        """Get health metrics."""
        return self._memory.check_health(thread_id=self._thread_id)
    
    def get_stats(self):
        """Get memory statistics."""
        return self._memory.get_stats(thread_id=self._thread_id)
    
    def compress(self):
        """Force compression."""
        return self._memory.compress(thread_id=self._thread_id)
