# ───────────────────────── src/manilafolder/__init__.py ─────────────────────────
"""
ManilaFolder: A lightweight macOS GUI for creating, opening, viewing, and extending
ChromaDB vector stores that index PDFs.

A filing cabinet-inspired document organization app that brings the familiar metaphor
of manila folders to digital document management with vector search capabilities.
"""

__version__ = "1.0.0"
__author__ = "ManilaFolder Team"

from .config import Config
from .db import create_vector_store, open_vector_store

# Public API exports
from .ingest import ingest_pdfs, register_loader

__all__ = [
    "ingest_pdfs",
    "register_loader",
    "create_vector_store",
    "open_vector_store",
    "Config",
    "__version__",
]
