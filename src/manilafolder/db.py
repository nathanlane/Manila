# ───────────────────────── src/manilafolder/db.py ─────────────────────────
"""
ChromaDB vector store operations and management.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .config import Config
from .logging_utils import log_error


class ChromaBackend:
    """ChromaDB implementation of VectorStoreBackend protocol."""

    def __init__(self, config: Config):
        """Initialize ChromaDB backend.

        Args:
            config: Configuration object
        """
        self.config = config
        self._embedding_function = self._get_embedding_function()

    def _get_embedding_function(self) -> Any:
        """Get appropriate embedding function based on configuration.

        Returns:
            ChromaDB embedding function instance
        """
        if self.config.embedding_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")

            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.openai_api_key, model_name=self.config.openai_model
            )
        else:
            # Default to SentenceTransformer
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )

    def create_collection(self, name: str, persist_directory: str) -> Any:
        """Create a new ChromaDB collection.

        Args:
            name: Collection name
            persist_directory: Directory to persist the collection

        Returns:
            ChromaDB collection instance

        Raises:
            RuntimeError: If collection creation fails
        """
        try:
            client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
            collection = client.create_collection(
                name=name, embedding_function=self._embedding_function
            )
            return collection
        except Exception as e:
            log_error(f"Failed to create collection '{name}'", e, self.config)
            raise RuntimeError(f"Collection creation failed: {e}")

    def load_collection(self, name: str, persist_directory: str) -> Any:
        """Load an existing ChromaDB collection.

        Args:
            name: Collection name
            persist_directory: Directory containing the collection

        Returns:
            ChromaDB collection instance

        Raises:
            RuntimeError: If collection loading fails
        """
        try:
            client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection(
                name=name, embedding_function=self._embedding_function
            )
            return collection
        except Exception as e:
            log_error(f"Failed to load collection '{name}'", e, self.config)
            raise RuntimeError(f"Collection loading failed: {e}")

    def add_documents(
        self, collection: Any, documents: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add documents to the ChromaDB collection.

        Args:
            collection: ChromaDB collection instance
            documents: List of document texts
            metadatas: List of metadata dictionaries

        Raises:
            RuntimeError: If document addition fails
        """
        try:
            ids = [f"doc_{i}_{hash(doc) % 10000}" for i, doc in enumerate(documents)]
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            log_error("Failed to add documents to collection", e, self.config)
            raise RuntimeError(f"Document addition failed: {e}")


def create_vector_store(db_path: str, config: Optional[Config] = None) -> Any:
    """Create a new vector store at the specified path.

    Args:
        db_path: Path where the database should be created
        config: Configuration object, uses defaults if None

    Returns:
        ChromaDB collection instance

    Raises:
        RuntimeError: If database creation fails
        FileExistsError: If database already exists
    """
    if config is None:
        config = Config()

    db_path = Path(db_path).resolve()

    # Check if database already exists
    if db_path.exists() and any(db_path.iterdir()):
        raise FileExistsError(f"Database already exists at {db_path}")

    # Create directory if it doesn't exist
    db_path.mkdir(parents=True, exist_ok=True)

    try:
        backend = ChromaBackend(config)
        collection = backend.create_collection(config.collection_name, str(db_path))
        return collection
    except Exception as e:
        log_error(f"Failed to create vector store at {db_path}", e, config)
        raise


def open_vector_store(db_path: str, config: Optional[Config] = None) -> Any:
    """Open an existing vector store.

    Args:
        db_path: Path to the existing database
        config: Configuration object, uses defaults if None

    Returns:
        ChromaDB collection instance

    Raises:
        FileNotFoundError: If database doesn't exist
        RuntimeError: If database opening fails
    """
    if config is None:
        config = Config()

    db_path = Path(db_path).resolve()

    # Check if database exists and contains ChromaDB files
    if not is_valid_chroma_db(db_path):
        raise FileNotFoundError(f"No valid ChromaDB found at {db_path}")

    try:
        backend = ChromaBackend(config)
        collection = backend.load_collection(config.collection_name, str(db_path))
        return collection
    except Exception as e:
        log_error(f"Failed to open vector store at {db_path}", e, config)
        raise


def is_valid_chroma_db(path: Path) -> bool:
    """Check if a directory contains a valid ChromaDB.

    Args:
        path: Path to check

    Returns:
        True if valid ChromaDB found, False otherwise
    """
    if not path.exists() or not path.is_dir():
        return False

    # Look for ChromaDB indicator files
    chroma_indicators = ["chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"]

    return any((path / indicator).exists() for indicator in chroma_indicators)


def get_collection_stats(collection: Any) -> Dict[str, int]:
    """Get statistics about the collection.

    Args:
        collection: ChromaDB collection instance

    Returns:
        Dictionary with collection statistics

    Raises:
        RuntimeError: If stats retrieval fails
    """
    try:
        count = collection.count()

        # Get sample of documents to analyze
        if count > 0:
            sample_size = min(100, count)
            results = collection.get(limit=sample_size)

            # Count unique files
            unique_files = set()
            total_pages = 0

            for metadata in results.get("metadatas", []):
                if metadata and "source" in metadata:
                    unique_files.add(metadata["source"])
                if metadata and "page" in metadata:
                    total_pages += 1

            return {
                "total_chunks": count,
                "unique_files": len(unique_files),
                "total_pages": total_pages,
            }
        else:
            return {"total_chunks": 0, "unique_files": 0, "total_pages": 0}
    except Exception as e:
        log_error("Failed to get collection statistics", e)
        raise RuntimeError(f"Stats retrieval failed: {e}")
