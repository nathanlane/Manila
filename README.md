# ManilaFolder

**Quickly spin up and manage ChromaDB vector databases** with an tiny Streamlit interface. Perfect for researchers and developers who need fast, local vector search capabilities to documents.

ManilaFolder streamlines the process of creating and populating ChromaDB vector stores for PDF documents (and more). With just a few clicks, you can have a fully functional vector database with semantic search, document chunking, and optional OCR correction for scanned texts.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/ManilaFolder.git
cd ManilaFolder
make install

# Launch Streamlit interface
make streamlit

# Your vector DB manager is now running at http://localhost:8501
```
Onc
**30 seconds to your first vector database:**
1. Click "Create New Database" in the sidebar
2. Name your database and click Create
3. Drag & drop PDFs into the upload area
4. Start searching!

## ✨ Key Features

### Vector Database Management
- 🚀 **Instant Setup**: Create ChromaDB vector stores in seconds
- 🔍 **Semantic Search**: Query documents with natural language
- 📊 **Real-time Progress**: Monitor document processing and indexing
- 💾 **Persistent Storage**: Local ChromaDB with SQLite backend

### Document Processing
- 📄 **Smart Chunking**: Configurable text splitting for optimal retrieval
- 🔧 **OCR Correction**: Fix errors in scanned documents automatically
- 🎯 **Batch Processing**: Upload multiple PDFs simultaneously
- 📈 **Processing Stats**: Track documents, chunks, and tokens

### Developer-Friendly
- 🌐 **Streamlit Interface**: Modern web UI with hot reload
- 🔌 **Extensible Design**: Add custom loaders and backends
- 📝 **Clean API**: Use as library or standalone app
- 🛡️ **Error Handling**: Comprehensive logging and recovery

## 📋 Requirements

- Python 3.11+
- Works on macOS, Linux, and Windows

## 🛠️ Installation

### Production Setup
```bash
make install          # Install dependencies
make streamlit       # Run web interface
```

### Development Setup
```bash
make dev-install     # Install dev dependencies
make dev            # Full dev setup with QA checks
make streamlit-dev  # Run with auto-reload
```

### Alternative Installation
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📖 Usage Guide

### Creating Your First Vector Database

1. **Launch ManilaFolder**
   ```bash
   make streamlit
   ```

2. **Create Database**
   - Click "Create New Database" in sidebar
   - Enter a name (e.g., "research-papers")
   - Choose location or use default
   - Click "Create Database"

3. **Add Documents**
   - Drag PDFs into the upload area
   - Or click "Browse files" to select
   - Watch the progress bar as documents are processed
   - Each PDF is chunked and embedded automatically

4. **Search Your Documents**
   - Type natural language queries
   - Get semantically relevant results
   - View document context and metadata

### Advanced Configuration

- **Chunk Size**: Adjust for longer/shorter text segments
- **Overlap**: Control context preservation between chunks
- **Embedding Model**: Choose from multiple SentenceTransformers
- **OCR Correction**: Enable for scanned documents

## 🏗️ Architecture

```
ManilaFolder/
├── streamlit_app.py         # Main web application (1,649 lines)
├── src/manilafolder/
│   ├── __init__.py         # Public API exports
│   ├── app.py              # CLI entry point (155 lines)
│   ├── db.py               # ChromaDB operations (243 lines)
│   ├── ingest.py           # Document processing (782 lines)
│   ├── config.py           # Configuration system (398 lines)
│   ├── ocr_correction.py   # OCR error correction (1,329 lines)
│   ├── ocr_patterns.py     # Correction patterns (870 lines)
│   ├── gui.py              # Legacy PySimpleGUI (506 lines)
│   └── logging_utils.py    # Error logging (76 lines)
├── tests/                   # Test suite
├── docs/                    # Documentation
└── Makefile                # Build automation
```

## 📦 Codebase Overview

### Core Modules

**`streamlit_app.py`** (1,649 lines)
Primary web interface with progressive UI, real-time progress tracking, and comprehensive settings management. Features OCR configuration UI with preview capabilities.

**`db.py`** (243 lines)
ChromaDB vector store operations including creation, document storage, retrieval, and search. Implements clean abstraction over ChromaDB API.

**`ingest.py`** (782 lines)
Document processing pipeline with pluggable loader system. Handles PDF text extraction, intelligent chunking, embedding generation, and batch processing with progress callbacks.

**`config.py`** (398 lines)
Centralized configuration management with validation, environment variable support, and VectorStoreBackend protocol for extensibility.

**`ocr_correction.py`** (1,329 lines)
Sophisticated OCR error correction system with:
- 100+ correction patterns covering ~85% of common errors
- Three correction levels (light/moderate/aggressive)
- Integrated spell checking with academic vocabulary protection
- LRU caching for performance optimization
- Detailed statistics tracking

**`ocr_patterns.py`** (870 lines)
Comprehensive pattern library for OCR corrections including character substitutions, ligature fixes, spacing corrections, and context-aware replacements.

**`gui.py`** (506 lines)
Legacy PySimpleGUI desktop interface with RetroMono theme. Functional but superseded by Streamlit interface.

**`logging_utils.py`** (76 lines)
Rotating file logger configuration for error tracking and debugging.

## 🧰 Development

### Commands
```bash
# Quality Assurance
make test           # Run test suite
make lint           # Check code style
make type-check     # Type checking
make qa             # All checks

# Building
make app            # Build standalone executable
make package        # Create distribution package
make clean          # Clean build artifacts
```

### Extending ManilaFolder

**Add New File Types:**
```python
from manilafolder import register_loader
from my_loader import DocxLoader

register_loader('.docx', DocxLoader)
```

**Custom Vector Backend:**
```python
from manilafolder.config import VectorStoreBackend

class PineconeBackend(VectorStoreBackend):
    def create_collection(self, name: str, path: Path):
        # Your implementation
        pass
```

## 🔧 Configuration

Create custom configurations:
```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "embedding_model": "all-MiniLM-L6-v2",
  "collection_name": "research_papers",
  "ocr_correction_enabled": true,
  "ocr_correction_level": "moderate"
}
```

## 🐛 Troubleshooting

- **OCR Issues**: Check correction level and preview patterns
- **Performance**: Monitor cache hit rates in statistics
- **Errors**: Check `manilafolder_error.log` for details
- **ChromaDB**: Verify `chroma.sqlite3` exists in database directory

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run quality checks: `make qa`
4. Submit pull request

---

*ManilaFolder - Advanced document organization with the power of AI* 📁
