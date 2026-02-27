"""Pytest configuration and fixtures."""

import pytest

from llmem import Memory
from llmem.storage.memory import InMemoryStorage


@pytest.fixture
def memory():
    """Create a fresh Memory instance for testing."""
    return Memory()


@pytest.fixture
def storage():
    """Create a fresh InMemoryStorage instance."""
    return InMemoryStorage()
