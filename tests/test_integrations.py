"""Tests for framework integrations."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llmem import Memory


class TestLangChainIntegration:
    """Tests for LangChain integration."""
    
    @pytest.fixture
    def mock_langchain(self):
        """Mock LangChain imports."""
        with patch.dict("sys.modules", {
            "langchain_core": MagicMock(),
            "langchain_core.chat_history": MagicMock(),
            "langchain_core.messages": MagicMock(),
            "langchain_core.memory": MagicMock(),
        }):
            yield
    
    def test_langchain_not_installed(self):
        """Test proper error when LangChain not installed."""
        from llmem.integrations.langchain import _check_langchain, LANGCHAIN_AVAILABLE
        
        if not LANGCHAIN_AVAILABLE:
            with pytest.raises(ImportError) as exc_info:
                _check_langchain()
            
            assert "LangChain" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)


class TestLangGraphIntegration:
    """Tests for LangGraph integration."""
    
    def test_langgraph_not_installed(self):
        """Test proper error when LangGraph not installed."""
        from llmem.integrations.langgraph import _check_langgraph, LANGGRAPH_AVAILABLE
        
        if not LANGGRAPH_AVAILABLE:
            with pytest.raises(ImportError) as exc_info:
                _check_langgraph()
            
            assert "LangGraph" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)


class TestPostgresStorage:
    """Tests for PostgreSQL storage (unit tests with mocks)."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        conn = AsyncMock()
        
        # Mock acquire context manager
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        conn.execute = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchval = AsyncMock(return_value=0)
        
        return pool
    
    @pytest.mark.asyncio
    async def test_save_turn(self, mock_pool):
        """Test saving a turn to PostgreSQL."""
        from llmem.storage.postgres import PostgresStorage
        from llmem.types import Turn
        
        storage = PostgresStorage(pool=mock_pool, auto_create_tables=True)
        turn = Turn(content="Hello", role="user")
        
        await storage.save_turn(turn, "test-thread")
        
        # Verify execute was called
        mock_pool.acquire.return_value.__aenter__.return_value.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_turns_empty(self, mock_pool):
        """Test getting turns when empty."""
        from llmem.storage.postgres import PostgresStorage
        
        storage = PostgresStorage(pool=mock_pool, auto_create_tables=True)
        
        turns = await storage.get_turns("test-thread")
        
        assert turns == []
    
    @pytest.mark.asyncio
    async def test_clear_thread(self, mock_pool):
        """Test clearing a thread."""
        from llmem.storage.postgres import PostgresStorage
        
        storage = PostgresStorage(pool=mock_pool, auto_create_tables=True)
        
        await storage.clear("test-thread")
        
        # Verify delete was called
        mock_pool.acquire.return_value.__aenter__.return_value.execute.assert_called()


class TestMongoStorage:
    """Tests for MongoDB storage (unit tests with mocks)."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock motor database."""
        db = MagicMock()
        
        # Mock collections
        turns_collection = AsyncMock()
        topics_collection = AsyncMock()
        
        db.__getitem__ = MagicMock(side_effect=lambda name: 
            turns_collection if "turns" in name else topics_collection
        )
        
        # Mock cursor
        cursor = AsyncMock()
        cursor.sort.return_value = cursor
        cursor.skip.return_value = cursor
        cursor.limit.return_value = cursor
        cursor.__aiter__ = lambda self: self
        cursor.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        
        turns_collection.find.return_value = cursor
        turns_collection.count_documents = AsyncMock(return_value=0)
        turns_collection.update_one = AsyncMock()
        turns_collection.delete_many = AsyncMock()
        turns_collection.create_index = AsyncMock()
        
        topics_collection.find.return_value = cursor
        topics_collection.update_one = AsyncMock()
        topics_collection.delete_many = AsyncMock()
        topics_collection.create_index = AsyncMock()
        
        return db
    
    @pytest.mark.asyncio
    async def test_save_turn(self, mock_db):
        """Test saving a turn to MongoDB."""
        from llmem.storage.mongo import MongoStorage
        from llmem.types import Turn
        
        storage = MongoStorage(db=mock_db)
        turn = Turn(content="Hello", role="user")
        
        await storage.save_turn(turn, "test-thread")
        
        # Verify update_one was called (upsert)
        mock_db["llmem_turns"].update_one.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_turn_count(self, mock_db):
        """Test getting turn count."""
        from llmem.storage.mongo import MongoStorage
        
        storage = MongoStorage(db=mock_db)
        
        count = await storage.get_turn_count("test-thread")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_clear_thread(self, mock_db):
        """Test clearing a thread."""
        from llmem.storage.mongo import MongoStorage
        
        storage = MongoStorage(db=mock_db)
        
        await storage.clear("test-thread")
        
        # Verify delete_many was called
        mock_db["llmem_turns"].delete_many.assert_called()
