"""LLMem integrations with popular frameworks."""

__all__ = []

# Lazy imports to avoid dependency issues
def LangChainMemory(*args, **kwargs):
    """LangChain memory integration. Requires: pip install llmem[langchain]"""
    from llmem.integrations.langchain import LangChainMemory as _LangChainMemory
    return _LangChainMemory(*args, **kwargs)

def LangGraphMemory(*args, **kwargs):
    """LangGraph memory wrapper. Requires: pip install llmem[langgraph]"""
    from llmem.integrations.langgraph import LangGraphMemory as _LangGraphMemory
    return _LangGraphMemory(*args, **kwargs)
