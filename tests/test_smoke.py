# ───────────────────────── tests/test_smoke.py ─────────────────────────
"""
Smoke tests for ManilaFolder core functionality.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from src.manilafolder.config import Config
from src.manilafolder.db import create_vector_store, is_valid_chroma_db
from src.manilafolder.ingest import prepare_chunks_for_storage, split_documents
from src.manilafolder.logging_utils import setup_logger


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.collection_name == "pdf_texts"
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = Config(chunk_size=500, chunk_overlap=50)
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50

        # Invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Config(chunk_size=0)

        # Invalid chunk_overlap
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            Config(chunk_overlap=-1)

        # chunk_overlap >= chunk_size
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            Config(chunk_size=100, chunk_overlap=100)


class TestLogging:
    """Test logging utilities."""

    def test_setup_logger(self):
        """Test logger setup."""
        config = Config()
        logger = setup_logger(config)

        assert logger.name == "manilafolder"
        assert len(logger.handlers) > 0

        # Test that subsequent calls don't add duplicate handlers
        logger2 = setup_logger(config)
        assert logger is logger2
        assert len(logger.handlers) == len(logger2.handlers)


class TestDocumentProcessing:
    """Test document processing functionality."""

    def test_split_documents(self):
        """Test document splitting."""
        # Create mock documents
        docs = [
            Document(
                page_content="This is a test document. " * 100,  # Long content
                metadata={"source": "test.pdf", "page": 1},
            )
        ]

        config = Config(chunk_size=200, chunk_overlap=50)
        chunks = split_documents(docs, config)

        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(
            len(chunk.page_content) <= 250 for chunk in chunks
        )  # Rough size check

    def test_prepare_chunks_for_storage(self):
        """Test chunk preparation for storage."""
        chunks = [
            Document(
                page_content="First chunk content",
                metadata={"source": "test.pdf", "page": 1},
            ),
            Document(
                page_content="Second chunk content",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]

        texts, metadatas = prepare_chunks_for_storage(chunks)

        assert len(texts) == 2
        assert len(metadatas) == 2
        assert texts[0] == "First chunk content"
        assert texts[1] == "Second chunk content"
        assert all("source" in meta for meta in metadatas)
        assert all("page" in meta for meta in metadatas)
        assert all("chunk_length" in meta for meta in metadatas)


class TestDatabaseOperations:
    """Test vector database operations."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db"
        self.config = Config()

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.manilafolder.db.chromadb.PersistentClient")
    @patch("src.manilafolder.db.SentenceTransformerEmbeddings")
    def test_create_vector_store(self, mock_embeddings, mock_client):
        """Test vector store creation."""
        # Mock ChromaDB client and collection
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        # Mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        try:
            self.setUp()

            # Test creation
            collection = create_vector_store(str(self.db_path), self.config)

            assert collection is mock_collection
            mock_client.assert_called_once()
            mock_client_instance.create_collection.assert_called_once()

        finally:
            self.tearDown()

    def test_is_valid_chroma_db(self):
        """Test ChromaDB validation."""
        try:
            self.setUp()

            # Non-existent path
            assert not is_valid_chroma_db(Path("/nonexistent/path"))

            # Empty directory
            self.db_path.mkdir(parents=True)
            assert not is_valid_chroma_db(self.db_path)

            # Directory with ChromaDB files
            (self.db_path / "chroma.sqlite3").touch()
            assert is_valid_chroma_db(self.db_path)

        finally:
            self.tearDown()


class TestIntegration:
    """Integration tests for core functionality."""

    @patch("src.manilafolder.db.chromadb.PersistentClient")
    @patch("src.manilafolder.db.SentenceTransformerEmbeddings")
    def test_end_to_end_workflow(self, mock_embeddings, mock_client):
        """Test complete workflow from database creation to document storage."""
        # Mock ChromaDB components
        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        mock_collection.count.return_value = 5

        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_integration_db"
            config = Config(chunk_size=100, chunk_overlap=20)

            # Create vector store
            collection = create_vector_store(str(db_path), config)
            assert collection is mock_collection

            # Create test documents
            test_docs = [
                Document(
                    page_content="This is test content for integration testing. " * 10,
                    metadata={"source": "test1.pdf", "page": 1},
                ),
                Document(
                    page_content="More test content for comprehensive testing. " * 10,
                    metadata={"source": "test1.pdf", "page": 2},
                ),
            ]

            # Split documents
            chunks = split_documents(test_docs, config)
            assert len(chunks) > 0

            # Prepare for storage
            texts, metadatas = prepare_chunks_for_storage(chunks)
            assert len(texts) > 0
            assert len(metadatas) == len(texts)

            # Simulate adding to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[f"test_{i}" for i in range(len(texts))],
            )

            # Verify collection was called
            collection.add.assert_called_once()

            # Verify call arguments
            call_args = collection.add.call_args
            assert "documents" in call_args.kwargs
            assert "metadatas" in call_args.kwargs
            assert "ids" in call_args.kwargs

            # Verify document count
            count = collection.count()
            assert count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
