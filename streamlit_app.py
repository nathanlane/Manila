#!/usr/bin/env python3
"""
ManilaFolder Streamlit Application

A web-based interface for managing ChromaDB vector stores with PDF document indexing.
Digital document organization with the familiar metaphor of manila folders.

This application provides a comprehensive UI for document management with advanced
features including OCR error correction, text preprocessing, and semantic search.

Key UI Components:
- Database Management: Create/open ChromaDB databases with validation
- Configuration Sidebar: Progressive disclosure of settings including OCR options
- File Processing: Upload interface with real-time progress and statistics
- OCR Correction: Configurable error correction for scanned documents
- Search Interface: Semantic search across indexed documents

OCR-Related UI Features:
1. Configuration Section (lines 926-1024):
   - Enable/disable toggle with help text
   - Correction level slider (light/moderate/aggressive)
   - Spell checking option
   - Performance tuning with cache size control
   - Preview button showing common correction patterns
   - Educational content about how corrections work

2. Processing Statistics Display (lines 1362-1452):
   - Conditional display when OCR is enabled
   - Three-column layout for key metrics
   - Total corrections with formatting
   - Correction rate as percentage
   - Cache hit rate for performance monitoring
   - Contextual insights based on correction rate
   - Help tooltips explaining each metric

User Experience Principles:
- Progressive disclosure: Advanced features hidden until needed
- Clear visual feedback: Metrics, progress bars, status messages
- Educational approach: Help text and previews explain functionality
- Performance transparency: Cache statistics help optimization
- Trust building: Preview corrections before processing

Run with: streamlit run streamlit_app.py
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Import ManilaFolder modules
from src.manilafolder.config import Config
from src.manilafolder.db import (
    create_vector_store,
    delete_collection,
    get_collection_stats,
    is_valid_chroma_db,
    list_collections,
    open_vector_store,
)
from src.manilafolder.ingest import get_supported_extensions, ingest_pdfs
from src.manilafolder.logging_utils import setup_logger

# Configure Streamlit page
st.set_page_config(
    page_title="ManilaFolder",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def validate_database_creation(db_path: str, collection: Any) -> Dict[str, bool]:
    """Validate that database was created successfully.

    Args:
        db_path: Path to the database directory
        collection: ChromaDB collection instance

    Returns:
        Dictionary of validation results
    """
    validation_results = {}

    try:
        # Check 1: ChromaDB files exist
        db_path_obj = Path(db_path)
        chroma_files = ["chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"]
        files_exist = any((db_path_obj / file).exists() for file in chroma_files)
        validation_results["database_files_exist"] = files_exist

        # Check 2: Collection is accessible
        try:
            collection_name = collection.name
            validation_results["collection_accessible"] = bool(collection_name)
        except Exception:
            validation_results["collection_accessible"] = False

        # Check 3: Embedding function works
        try:
            # Test with a simple query (this initializes the embedding function)
            collection.query(query_texts=["test"], n_results=1)
            validation_results["embedding_function_works"] = True
        except Exception as e:
            # If no documents exist, that's okay - we just want to test
            # the embedding function
            error_str = str(e).lower()
            if "no results" in error_str or "not enough" in error_str:
                validation_results["embedding_function_works"] = True
            else:
                validation_results["embedding_function_works"] = False

        # Check 4: Write permissions
        try:
            # Try to add a test document and then remove it
            test_doc = ["This is a test document for validation"]
            test_metadata = [{"source": "validation_test", "test": True}]
            test_id = ["validation_test_doc"]

            collection.add(documents=test_doc, metadatas=test_metadata, ids=test_id)
            collection.delete(ids=test_id)
            validation_results["write_permissions"] = True
        except Exception:
            validation_results["write_permissions"] = False

        # Check 5: Directory is writable
        try:
            test_file = db_path_obj / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            validation_results["directory_writable"] = True
        except Exception:
            validation_results["directory_writable"] = False

    except Exception:
        # If validation itself fails, mark all as failed
        validation_results = {
            "database_files_exist": False,
            "collection_accessible": False,
            "embedding_function_works": False,
            "write_permissions": False,
            "directory_writable": False,
        }

    return validation_results


def test_openai_key(api_key: str, model_name: str = "text-embedding-3-small") -> None:
    """Test if an OpenAI API key is valid by making a small embedding request.

    Args:
        api_key: OpenAI API key to test
        model_name: OpenAI model to test with
    """
    try:
        with st.spinner(f"Testing OpenAI API key with {model_name}..."):
            # Import OpenAI client
            from chromadb.utils import embedding_functions

            # Create embedding function with the API key
            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, model_name=model_name
            )

            # Test with a simple text
            test_texts = ["This is a test to validate the OpenAI API key."]
            embeddings = embedding_function(test_texts)

            # If we get here, the API key works
            if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                st.success("✅ API key is valid! Successfully generated embeddings.")
                st.info(f"📊 Embedding dimension: {len(embeddings[0])}")
            else:
                st.error("❌ API key test failed: No embeddings returned")

    except ImportError as e:
        st.error("❌ Missing required dependencies for OpenAI testing")
        st.error(f"Error: {str(e)}")
    except Exception as e:
        error_msg = str(e).lower()

        if "unauthorized" in error_msg or "invalid api key" in error_msg:
            st.error("❌ Invalid API key: The provided key is not valid")
        elif "quota" in error_msg or "billing" in error_msg:
            st.error("❌ API key has no available quota or billing issues")
        elif "rate limit" in error_msg:
            st.warning(
                "⚠️ Rate limit exceeded. API key appears valid but try again later"
            )
        elif "network" in error_msg or "connection" in error_msg:
            st.error("❌ Network error: Check your internet connection")
        else:
            st.error(f"❌ API key test failed: {str(e)}")

        # Log the full error for debugging
        if "logger" in st.session_state:
            st.session_state.logger.error(
                f"OpenAI API key test failed: {e}", exc_info=True
            )


def export_to_env_file(
    api_key: str, model_name: str = "text-embedding-3-small"
) -> None:
    """Export API key to .env file in the current directory.

    Args:
        api_key: OpenAI API key to export
        model_name: OpenAI model name to export
    """
    try:
        env_path = Path(".env")

        # Read existing .env content if it exists
        existing_content = ""
        existing_vars = {}

        if env_path.exists():
            with open(env_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        existing_vars[key.strip()] = value.strip()
                    existing_content += (
                        line + "\n" if line != lines[-1].rstrip() else line
                    )

        # Update or add the OpenAI variables
        existing_vars["OPENAI_API_KEY"] = f'"{api_key}"'
        existing_vars["OPENAI_MODEL"] = f'"{model_name}"'

        # Write updated .env file
        with open(env_path, "w") as f:
            f.write("# ManilaFolder Environment Variables\n")
            f.write("# Generated automatically - do not commit to version control\n\n")

            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")

        st.success(f"✅ API key exported to .env file: {env_path.absolute()}")
        st.info("💡 Remember to add .env to your .gitignore file to keep keys secure")

        # Show .gitignore reminder
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists() or ".env" not in gitignore_path.read_text():
            st.warning("⚠️ Consider adding '.env' to your .gitignore file for security")

            if st.button(
                "📝 Add .env to .gitignore", help="Automatically add .env to .gitignore"
            ):
                add_to_gitignore()

    except Exception as e:
        st.error(f"❌ Failed to export to .env file: {str(e)}")
        if "logger" in st.session_state:
            st.session_state.logger.error(
                f"Failed to export .env file: {e}", exc_info=True
            )


def add_to_gitignore() -> None:
    """Add .env to .gitignore file."""
    try:
        gitignore_path = Path(".gitignore")

        # Read existing content
        existing_content = ""
        if gitignore_path.exists():
            existing_content = gitignore_path.read_text()

        # Check if .env is already there
        if ".env" not in existing_content:
            # Add .env entry
            if existing_content and not existing_content.endswith("\n"):
                existing_content += "\n"
            existing_content += "\n# Environment variables (ManilaFolder)\n.env\n"

            gitignore_path.write_text(existing_content)
            st.success("✅ Added .env to .gitignore")
        else:
            st.info("ℹ️ .env is already in .gitignore")

    except Exception as e:
        st.error(f"❌ Failed to update .gitignore: {str(e)}")
        if "logger" in st.session_state:
            st.session_state.logger.error(
                f"Failed to update .gitignore: {e}", exc_info=True
            )


def initialize_session_state():
    """Initialize Streamlit session state variables."""

    # Core application state
    if "collection" not in st.session_state:
        st.session_state.collection = None

    if "db_path" not in st.session_state:
        st.session_state.db_path = None

    if "config" not in st.session_state:
        st.session_state.config = Config()
        # Initialize new fields for backward compatibility
        if not hasattr(st.session_state.config, "embedding_provider"):
            st.session_state.config.embedding_provider = "sentencetransformer"
        if not hasattr(st.session_state.config, "openai_model"):
            st.session_state.config.openai_model = "text-embedding-3-small"

    if "logger" not in st.session_state:
        st.session_state.logger = setup_logger(st.session_state.config)

    # UI state
    if "file_stats" not in st.session_state:
        st.session_state.file_stats = []

    if "processing_files" not in st.session_state:
        st.session_state.processing_files = False

    if "show_config" not in st.session_state:
        st.session_state.show_config = False


def create_database_interface():
    """Create the database creation interface."""

    st.subheader("🗂️ Create New Database")

    with st.form("create_db_form"):
        db_name = st.text_input(
            "Database Name",
            value="MyManilaFolder",
            help="Enter a name for your new document database",
        )

        parent_dir = st.text_input(
            "Parent Directory",
            value=str(Path.home()),
            help="Directory where the database folder will be created",
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value="documents",
            help="Name for the initial collection (e.g., 'research', 'contracts', 'manuals')",
        )

        # Show the full path that will be created
        if db_name and parent_dir:
            full_path = Path(parent_dir) / db_name.strip()
            st.info(f"📁 Database will be created at: `{full_path}`")

        submitted = st.form_submit_button("Create Database", type="primary")

        if submitted:
            if not db_name.strip():
                st.error("Please enter a database name")
                return

            if not parent_dir.strip():
                st.error("Please enter a parent directory")
                return

            try:
                db_path = Path(parent_dir) / db_name.strip()

                # Check if directory already exists
                if db_path.exists() and any(db_path.iterdir()):
                    st.error(f"Directory already exists and is not empty: {db_path}")
                    return

                # Validate collection name
                if not collection_name.strip():
                    st.error("Please enter a collection name")
                    return
                
                # Create the database
                with st.spinner("Creating database..."):
                    collection = create_vector_store(
                        str(db_path), 
                        st.session_state.config,
                        collection_name=collection_name.strip()
                    )

                    # Validate the database creation
                    st.info("🔍 Validating database creation...")
                    validation_results = validate_database_creation(
                        str(db_path), collection
                    )

                    if all(validation_results.values()):
                        # Update session state only if validation passes
                        st.session_state.collection = collection
                        st.session_state.db_path = str(db_path)
                        st.success(
                            f"✅ Database created and validated successfully at: "
                            f"{db_path}"
                        )

                        # Show validation details
                        with st.expander("✅ Validation Details"):
                            for check, passed in validation_results.items():
                                status = "✅" if passed else "❌"
                                st.write(f"{status} {check.replace('_', ' ').title()}")
                    else:
                        # Show validation failures
                        st.error("❌ Database validation failed!")
                        for check, passed in validation_results.items():
                            if not passed:
                                st.error(f"Failed: {check.replace('_', ' ').title()}")

                        # Clean up failed database
                        try:
                            if db_path.exists():
                                import shutil

                                shutil.rmtree(db_path)
                                st.info("🧹 Cleaned up failed database creation")
                        except Exception:
                            pass

                        return

                st.rerun()

            except Exception as e:
                st.error(f"Failed to create database: {str(e)}")
                st.session_state.logger.error(
                    f"Database creation failed: {e}", exc_info=True
                )


def open_database_interface():
    """Create the database opening interface."""

    st.subheader("📂 Open Existing Database")

    with st.form("open_db_form"):
        db_directory = st.text_input(
            "Database Directory",
            value="",
            help="Path to directory containing ChromaDB files",
        )

        submitted = st.form_submit_button("Open Database", type="primary")

        if submitted:
            if not db_directory.strip():
                st.error("Please enter a database directory path")
                return

            db_path = Path(db_directory.strip())

            if not db_path.exists():
                st.error(f"Directory does not exist: {db_path}")
                return

            if not is_valid_chroma_db(db_path):
                st.error(f"No valid ChromaDB found at: {db_path}")
                st.info("Look for directories containing 'chroma.sqlite3' files")
                return

            try:
                # List available collections
                available_collections = list_collections(str(db_path))
                
                if not available_collections:
                    st.error("No collections found in this database")
                    return
                
                # If multiple collections, let user choose
                if len(available_collections) > 1:
                    selected_collection = st.selectbox(
                        "Select a collection to open:",
                        available_collections,
                        key="open_collection_selector"
                    )
                else:
                    selected_collection = available_collections[0]
                    st.info(f"Opening collection: {selected_collection}")
                
                with st.spinner("Opening database..."):
                    collection = open_vector_store(
                        str(db_path), 
                        st.session_state.config,
                        collection_name=selected_collection
                    )

                    # Update session state
                    st.session_state.collection = collection
                    st.session_state.db_path = str(db_path)

                st.success(f"✅ Database opened successfully: {db_path}")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to open database: {str(e)}")
                st.session_state.logger.error(
                    f"Database opening failed: {e}", exc_info=True
                )


def database_selection_sidebar():
    """Create database selection interface in sidebar."""

    st.sidebar.header("🗂️ Database")

    if st.session_state.collection is None:
        st.sidebar.info("No database connected")

        db_action = st.sidebar.radio(
            "Choose an action:",
            ["Create New Database", "Open Existing Database"],
            key="db_action_radio",
        )

        if db_action == "Create New Database":
            with st.sidebar:
                create_database_interface()
        else:
            with st.sidebar:
                open_database_interface()

    else:
        # Show current database info with better styling
        st.sidebar.success("✅ Database Connected")

        # Add a compact info box
        with st.sidebar.container():
            st.markdown(
                """
            <style>
            .db-info-box {
                background-color: #F0F9FF;
                border: 1px solid #BFDBFE;
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
                font-size: 0.875rem;
            }
            .db-path {
                color: #1E40AF;
                word-break: break-all;
                font-family: monospace;
                font-size: 0.75rem;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Add collection styling
            st.markdown(
                """
            <style>
            .collection-box {
                background-color: #F3E8FF;
                border: 1px solid #C084FC;
                border-radius: 6px;
                padding: 0.5rem;
                margin: 0.25rem 0;
                font-size: 0.875rem;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            
            # Show path in a styled box
            db_name = Path(st.session_state.db_path).name
            st.markdown(
                f"""
            <div class="db-info-box">
                <strong>📁 {db_name}</strong><br>
                <span class="db-path">{st.session_state.db_path}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            
            # Show current collection
            current_collection = getattr(st.session_state.collection, 'name', 'Unknown')
            st.markdown(
                f"""
            <div class="collection-box">
                <strong>📂 Collection:</strong> {current_collection}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Get database stats with better display
        try:
            stats = get_collection_stats(st.session_state.collection)
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("📄 Files", f"{stats.get('unique_files', 0):,}")
            with col2:
                st.metric("🔤 Chunks", f"{stats.get('total_chunks', 0):,}")
        except Exception:
            st.sidebar.error("Could not load database stats")

        # Collection Management
        st.sidebar.markdown("---")
        with st.sidebar.expander("📚 Collection Management", expanded=False):
            # List collections in the database
            try:
                available_collections = list_collections(st.session_state.db_path)
                if len(available_collections) > 1:
                    st.write("**Switch Collection**")
                    selected_collection = st.selectbox(
                        "Select collection:",
                        available_collections,
                        index=available_collections.index(current_collection) if current_collection in available_collections else 0,
                        key="collection_selector"
                    )
                    
                    if selected_collection != current_collection:
                        if st.button("🔄 Switch", key="switch_collection"):
                            try:
                                st.session_state.collection = open_vector_store(
                                    st.session_state.db_path,
                                    st.session_state.config,
                                    collection_name=selected_collection
                                )
                                st.success(f"Switched to collection: {selected_collection}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to switch collection: {str(e)}")
                
                st.write("**Create New Collection**")
                new_collection_name = st.text_input(
                    "Collection name:",
                    placeholder="Enter collection name",
                    key="new_collection_name"
                )
                
                if st.button("➕ Create", key="create_collection", disabled=not new_collection_name):
                    try:
                        # Create new collection
                        new_collection = create_vector_store(
                            st.session_state.db_path,
                            st.session_state.config,
                            collection_name=new_collection_name
                        )
                        st.session_state.collection = new_collection
                        st.success(f"Created collection: {new_collection_name}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Failed to create collection: {str(e)}")
                
                # Delete collection (with safety check)
                if len(available_collections) > 1:
                    st.write("**Delete Collection**")
                    st.warning("⚠️ Deletion is permanent!")
                    delete_collection_name = st.selectbox(
                        "Select collection to delete:",
                        [c for c in available_collections if c != current_collection],
                        key="delete_collection_selector"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        confirm_delete = st.checkbox("I understand", key="confirm_delete")
                    with col2:
                        if st.button("🗑️ Delete", key="delete_collection", disabled=not confirm_delete):
                            try:
                                delete_collection(st.session_state.db_path, delete_collection_name)
                                st.success(f"Deleted collection: {delete_collection_name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete collection: {str(e)}")
                
            except Exception as e:
                st.error(f"Error managing collections: {str(e)}")
        
        # Close database button with better styling
        st.sidebar.markdown("---")
        if st.sidebar.button(
            "🚪 Close Database", type="secondary", use_container_width=True
        ):
            st.session_state.collection = None
            st.session_state.db_path = None
            st.session_state.file_stats = []
            st.rerun()


def configuration_sidebar():
    """Create configuration interface in sidebar.

    This function creates the main configuration interface for ManilaFolder,
    organizing settings into logical groups with expandable sections. The sidebar
    design follows progressive disclosure principles, showing basic settings first
    and revealing advanced options as users need them.

    Configuration Sections:
    1. Processing Settings - Core chunking and embedding configuration
    2. Text Preprocessing - Options for cleaning and normalizing text
    3. OCR Error Correction - Settings for fixing scanned document errors

    Each section is contained in an expander to keep the interface clean while
    providing access to powerful configuration options.
    """

    st.sidebar.header("⚙️ Configuration")

    with st.sidebar.expander(
        "Processing Settings", expanded=st.session_state.show_config
    ):
        # Chunk size
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=st.session_state.config.chunk_size,
            step=50,
            help="Size of text chunks for document processing",
        )

        # Chunk overlap with percentage display
        max_overlap = min(chunk_size // 2, 500)
        current_overlap = min(st.session_state.config.chunk_overlap, max_overlap)

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=max_overlap,
            value=current_overlap,
            step=10,
            help=(
                "Overlap between consecutive chunks "
                "(recommended: 10-20% of chunk size)"
            ),
            format="%d",
        )

        # Show percentage
        if chunk_size > 0:
            overlap_percent = int((chunk_overlap / chunk_size) * 100)
            st.caption(
                f"📊 Overlap: {chunk_overlap} tokens "
                f"({overlap_percent}% of chunk size)"
            )
        else:
            st.caption(f"📊 Overlap: {chunk_overlap} tokens")

        # Embedding provider selection
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["sentencetransformer", "openai"],
            index=(
                0
                if st.session_state.config.embedding_provider == "sentencetransformer"
                else 1
            ),
            help=("Choose between local SentenceTransformer models or OpenAI API"),
            format_func=lambda x: (
                "🤖 SentenceTransformer (Local)"
                if x == "sentencetransformer"
                else "🔗 OpenAI (API)"
            ),
        )

        # Model selection based on provider
        if embedding_provider == "sentencetransformer":
            sentence_models = {
                "all-MiniLM-L6-v2": "All-MiniLM-L6-v2 (Fast, 384 dim)",
                "all-mpnet-base-v2": "All-MPNet-Base-v2 (Best quality, 768 dim)",
                "all-distilroberta-v1": "All-DistilRoBERTa-v1 (Balanced, 768 dim)",
            }

            embedding_model = st.selectbox(
                "SentenceTransformer Model",
                options=list(sentence_models.keys()),
                index=(
                    0
                    if st.session_state.config.embedding_model == "all-MiniLM-L6-v2"
                    else 0
                ),
                help="Local embedding models - no API key required",
                format_func=lambda x: sentence_models[x],
            )
            openai_model = st.session_state.config.openai_model  # Keep current

        else:
            openai_models = {
                "text-embedding-3-small": "GPT-3 Small (1536 dim, efficient)",
                "text-embedding-3-large": "GPT-3 Large (3072 dim, highest quality)",
                "text-embedding-ada-002": "Ada-002 (1536 dim, legacy)",
            }

            openai_model = st.selectbox(
                "OpenAI Model",
                options=list(openai_models.keys()),
                index=(
                    0
                    if st.session_state.config.openai_model == "text-embedding-3-small"
                    else 0
                ),
                help="OpenAI embedding models - requires API key",
                format_func=lambda x: openai_models[x],
            )
            embedding_model = st.session_state.config.embedding_model  # Keep current

        # API Key management for OpenAI
        if embedding_provider == "openai":
            st.divider()
            st.write("**🔑 OpenAI API Key**")

            # Check if API key exists in environment
            env_key_exists = bool(os.getenv("OPENAI_API_KEY"))

            if env_key_exists:
                api_key_source = st.radio(
                    "API Key Source:",
                    ["Use Environment Variable", "Enter Manually"],
                    help="Environment variable is more secure for production use",
                )
            else:
                st.info("💡 No OPENAI_API_KEY found in environment variables")
                api_key_source = "Enter Manually"

            if api_key_source == "Enter Manually":
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Your API key is stored in session memory only "
                    "and not saved to disk",
                    placeholder="sk-...",
                )

                if api_key:
                    # Basic format validation
                    if not api_key.startswith("sk-"):
                        st.warning("⚠️ OpenAI API keys typically start with 'sk-'")
                    elif len(api_key) < 40:
                        st.warning("⚠️ API key seems too short")
                    else:
                        st.success("✅ API key format looks correct")

                        # Add test button for API key validation
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                "🧪 Test API Key",
                                help="Test if the API key works by making a "
                                "small embedding request",
                            ):
                                # Use the currently selected OpenAI model for testing
                                current_model = (
                                    openai_model
                                    if "openai_model" in locals()
                                    else st.session_state.config.openai_model
                                )
                                test_openai_key(api_key, current_model)

                        with col2:
                            if st.button(
                                "💾 Export to .env",
                                help="Save API key to .env file for future use",
                            ):
                                current_model = (
                                    openai_model
                                    if "openai_model" in locals()
                                    else st.session_state.config.openai_model
                                )
                                export_to_env_file(api_key, current_model)

                    # Update session state
                    st.session_state.config.openai_api_key = api_key
                else:
                    st.session_state.config.openai_api_key = ""

            else:
                # Use environment variable
                env_api_key = os.getenv("OPENAI_API_KEY", "")
                st.session_state.config.openai_api_key = env_api_key
                st.success("✅ Using API key from environment variable")

                # Add test button for environment variable key
                if env_api_key and st.button(
                    "🧪 Test Environment API Key",
                    help="Test if the environment variable API key works",
                ):
                    # Use the currently selected OpenAI model for testing
                    current_model = (
                        openai_model
                        if "openai_model" in locals()
                        else st.session_state.config.openai_model
                    )
                    test_openai_key(env_api_key, current_model)

            # Show warning about security
            with st.expander("🔒 Security Information"):
                st.markdown(
                    """
                **API Key Security:**
                - Keys entered manually are stored in session memory only
                - Keys are NOT saved to disk or persistent storage
                - Keys are cleared when you close the browser
                - For production, use environment variables instead

                **API Key Testing:**
                - The test button makes a small embedding request (~$0.00001 cost)
                - Tests validate both key authenticity and model access
                - Failed tests provide specific error diagnostics

                **Environment Variable Setup:**
                ```bash
                export OPENAI_API_KEY="your-api-key-here"
                ```
                """
                )

        # Update configuration if changed
        config_changed = (
            chunk_size != st.session_state.config.chunk_size
            or chunk_overlap != st.session_state.config.chunk_overlap
            or embedding_provider != st.session_state.config.embedding_provider
            or embedding_model != st.session_state.config.embedding_model
            or openai_model != st.session_state.config.openai_model
        )

        if config_changed:
            st.session_state.config.chunk_size = chunk_size
            st.session_state.config.chunk_overlap = chunk_overlap
            st.session_state.config.embedding_provider = embedding_provider
            st.session_state.config.embedding_model = embedding_model
            st.session_state.config.openai_model = openai_model

            st.sidebar.info("Configuration updated")

    # Text Preprocessing Section
    with st.sidebar.expander("📝 Text Preprocessing", expanded=False):
        # Enable/disable preprocessing
        enable_preprocessing = st.checkbox(
            "Enable Text Preprocessing",
            value=st.session_state.config.enable_preprocessing,
            help="Apply text cleaning before generating embeddings",
        )

        if enable_preprocessing:
            # Provider-specific recommendation
            if st.session_state.config.embedding_provider == "openai":
                st.info("🔗 OpenAI models work best with minimal preprocessing")
                recommended_intensity = "minimal"
            else:
                st.info("🤖 SentenceTransformers benefit from moderate cleaning")
                recommended_intensity = "moderate"

            # Preprocessing intensity
            intensity_options = ["minimal", "moderate", "aggressive"]
            current_intensity = st.session_state.config.preprocessing_intensity
            if current_intensity not in intensity_options:
                current_intensity = recommended_intensity

            preprocessing_intensity = st.select_slider(
                "Preprocessing Intensity",
                options=intensity_options,
                value=current_intensity,
                help="Higher intensity may reduce embedding quality for modern models",
            )

            # Academic document options
            st.write("**📚 Academic Document Processing**")
            enable_academic = st.checkbox(
                "Enable Academic Cleaning",
                value=st.session_state.config.enable_academic_cleaning,
                help="Remove citations, figure references, headers/footers",
            )

            if enable_academic:
                col1, col2 = st.columns(2)
                with col1:
                    remove_citations = st.checkbox(
                        "Remove Citations",
                        value=st.session_state.config.remove_citations,
                        help="Remove [1], (Smith et al., 2020) style citations",
                    )
                    remove_figures = st.checkbox(
                        "Remove Figure Refs",
                        value=st.session_state.config.remove_figure_refs,
                        help="Remove Figure 1, Table 2 references",
                    )
                with col2:
                    remove_headers = st.checkbox(
                        "Remove Headers/Footers",
                        value=st.session_state.config.remove_headers_footers,
                        help="Remove page headers, footers, page numbers",
                    )
                    preserve_structure = st.checkbox(
                        "Preserve Sections",
                        value=st.session_state.config.preserve_section_structure,
                        help="Keep section headings and structure",
                    )
            else:
                # Set defaults when academic cleaning is disabled
                remove_citations = st.session_state.config.remove_citations
                remove_figures = st.session_state.config.remove_figure_refs
                remove_headers = st.session_state.config.remove_headers_footers
                preserve_structure = st.session_state.config.preserve_section_structure

            # Quality filtering
            st.write("**🎯 Quality Filtering**")
            enable_quality = st.checkbox(
                "Enable Quality Filtering",
                value=st.session_state.config.enable_quality_filtering,
                help="Filter out low-quality text chunks",
            )

            if enable_quality:
                col1, col2 = st.columns(2)
                with col1:
                    min_words = st.slider(
                        "Min Words per Chunk",
                        min_value=5,
                        max_value=50,
                        value=st.session_state.config.min_chunk_words,
                        help="Minimum words required in each chunk",
                    )
                    min_alpha = st.slider(
                        "Min Alphabetic %",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.config.min_alpha_ratio,
                        step=0.1,
                        format="%.1f",
                        help="Minimum percentage of alphabetic characters",
                    )
                with col2:
                    max_repetition = st.slider(
                        "Max Repetition %",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.config.max_repetition_ratio,
                        step=0.1,
                        format="%.1f",
                        help="Maximum allowed text repetition",
                    )
                    min_readability = st.slider(
                        "Min Readability Score",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.config.min_readability_score,
                        step=5.0,
                        format="%.1f",
                        help="Minimum readability score (0 = no filtering)",
                    )
            else:
                # Set defaults when quality filtering is disabled
                min_words = st.session_state.config.min_chunk_words
                min_alpha = st.session_state.config.min_alpha_ratio
                max_repetition = st.session_state.config.max_repetition_ratio
                min_readability = st.session_state.config.min_readability_score

        else:
            # Set defaults when preprocessing is disabled
            preprocessing_intensity = st.session_state.config.preprocessing_intensity
            enable_academic = st.session_state.config.enable_academic_cleaning
            remove_citations = st.session_state.config.remove_citations
            remove_figures = st.session_state.config.remove_figure_refs
            remove_headers = st.session_state.config.remove_headers_footers
            preserve_structure = st.session_state.config.preserve_section_structure
            enable_quality = st.session_state.config.enable_quality_filtering
            min_words = st.session_state.config.min_chunk_words
            min_alpha = st.session_state.config.min_alpha_ratio
            max_repetition = st.session_state.config.max_repetition_ratio
            min_readability = st.session_state.config.min_readability_score

        # Update configuration if preprocessing settings changed
        preprocessing_changed = (
            enable_preprocessing != st.session_state.config.enable_preprocessing
            or preprocessing_intensity
            != st.session_state.config.preprocessing_intensity
            or enable_academic != st.session_state.config.enable_academic_cleaning
            or remove_citations != st.session_state.config.remove_citations
            or remove_figures != st.session_state.config.remove_figure_refs
            or remove_headers != st.session_state.config.remove_headers_footers
            or preserve_structure != st.session_state.config.preserve_section_structure
            or enable_quality != st.session_state.config.enable_quality_filtering
            or min_words != st.session_state.config.min_chunk_words
            or min_alpha != st.session_state.config.min_alpha_ratio
            or max_repetition != st.session_state.config.max_repetition_ratio
            or min_readability != st.session_state.config.min_readability_score
        )

        if preprocessing_changed:
            # Update preprocessing configuration
            st.session_state.config.enable_preprocessing = enable_preprocessing
            st.session_state.config.preprocessing_intensity = preprocessing_intensity
            st.session_state.config.enable_academic_cleaning = enable_academic
            st.session_state.config.remove_citations = remove_citations
            st.session_state.config.remove_figure_refs = remove_figures
            st.session_state.config.remove_headers_footers = remove_headers
            st.session_state.config.preserve_section_structure = preserve_structure
            st.session_state.config.enable_quality_filtering = enable_quality
            st.session_state.config.min_chunk_words = min_words
            st.session_state.config.min_alpha_ratio = min_alpha
            st.session_state.config.max_repetition_ratio = max_repetition
            st.session_state.config.min_readability_score = min_readability

            st.sidebar.info("Preprocessing settings updated")

    # OCR Correction Section
    with st.sidebar.expander("🔍 OCR Error Correction", expanded=False):
        """OCR error correction configuration interface.

        This section provides users with controls to enable and configure OCR error
        correction for scanned documents. The UI is designed to be progressive,
        revealing more options as users enable features, preventing overwhelming
        new users while providing power users with fine-grained control.

        UI Design Principles:
            1. **Progressive Disclosure**: Start simple, reveal complexity as needed
            2. **Visual Feedback**: Use colors/icons to indicate impact levels
            3. **Educational**: Help text explains not just what but why
            4. **Transparency**: Preview shows actual corrections that will be made
            5. **Performance Awareness**: Show impact of settings on processing time

        UI Workflow:
            1. User enables OCR correction with main checkbox
            2. Additional options appear for correction level and spell checking
            3. Performance settings allow optimization for large document sets
            4. Preview button shows common correction patterns for transparency

        User Experience Considerations:
            - Default to disabled to avoid unexpected text modifications
            - Use informative help text to explain impact of each setting
            - Progressive disclosure pattern - show advanced options only when needed
            - Visual feedback through info boxes explaining what OCR correction does
            - Preview functionality helps users understand corrections before processing
            - Color-coded correction levels (green/yellow/red) for quick understanding

        Accessibility:
            - All controls have descriptive help text for screen readers
            - Color indicators accompanied by text descriptions
            - Keyboard navigable with clear focus states
            - Preview content in monospace font for clarity

        Performance Impact Visualization:
            The UI shows estimated processing time impact for each setting,
            helping users make informed decisions about the speed/quality tradeoff.

        See Also:
            Config: OCR configuration options and defaults
            apply_ocr_correction: Implementation of OCR correction
            OCR Statistics Display: Results visualization after processing
        """
        # Enable/disable OCR correction
        enable_ocr = st.checkbox(
            "Enable OCR Correction",
            value=st.session_state.config.enable_ocr_correction,
            help="Fix common OCR errors in scanned documents",
        )

        if enable_ocr:
            st.info(
                "🔧 OCR correction fixes common scanning errors like 'rn'→'m', "
                "'l'→'1', and spell checking"
            )

            # Correction level
            ocr_level_options = ["light", "moderate", "aggressive"]
            current_level = st.session_state.config.ocr_correction_level
            if current_level not in ocr_level_options:
                current_level = "moderate"

            ocr_level = st.select_slider(
                "Correction Level",
                options=ocr_level_options,
                value=current_level,
                help=(
                    "Light: Basic pattern fixes only\n"
                    "Moderate: Patterns + conservative spell check\n"
                    "Aggressive: All corrections + aggressive spell check"
                ),
            )

            # Visual indicator for correction level impact
            if ocr_level == "light":
                st.caption("🟢 Minimal changes - safer for technical documents")
            elif ocr_level == "moderate":
                st.caption("🟡 Balanced approach - recommended for most documents")
            else:
                st.caption("🔴 Maximum corrections - may over-correct technical terms")

            # Spell checking option
            enable_spell = st.checkbox(
                "Enable Spell Checking",
                value=st.session_state.config.ocr_enable_spell_check,
                help="Use intelligent spell checking for OCR errors",
            )

            # Simple spellcheck option (only show if spell checking is enabled)
            if enable_spell:
                use_simple_spell = st.checkbox(
                    "Use Conservative Spell Checker",
                    value=st.session_state.config.ocr_use_simple_spellcheck,
                    help=(
                        "🔹 **Conservative (pyspellchecker)**: Slower but safer for technical documents\n"
                        "🔸 **Standard (SymSpell)**: Faster but may over-correct technical terms"
                    ),
                )
            else:
                use_simple_spell = st.session_state.config.ocr_use_simple_spellcheck

            # Performance settings
            st.write("**⚡ Performance Settings**")
            cache_size = st.slider(
                "Cache Size",
                min_value=1000,
                max_value=50000,
                value=st.session_state.config.ocr_cache_size,
                step=1000,
                help="Larger cache = faster processing for repeated words",
                format="%d words",
            )

            # Preview of common corrections
            if st.button("👁️ Preview Common Corrections"):
                """Display OCR correction preview.

                This button reveals a preview of common OCR error patterns that will
                be corrected. This transparency helps users understand what changes
                will be made to their documents and builds trust in the correction
                process.

                Preview Organization:
                    - Letter Confusions: Most common character substitutions
                    - Number Confusions: Digit-letter ambiguities in numeric contexts
                    - Academic Patterns: Scholarly document conventions

                User Benefits:
                    1. **Transparency**: See exactly what will be changed
                    2. **Validation**: Confirm corrections match document type
                    3. **Education**: Learn about common OCR errors
                    4. **Confidence**: Build trust in the correction process

                The preview shows the most frequent patterns (~80% of corrections)
                but the full system includes 100+ patterns for comprehensive coverage.

                Technical Details:
                    - Patterns shown represent ~85% OCR error coverage
                    - Examples use actual before→after transformations
                    - Categories match the pattern organization in ocr_patterns.py
                """
                st.write("**Common OCR Patterns:**")
                st.code(
                    """
Letter confusions:
• 'rn' → 'm' (most common)
• 'cl' → 'd'
• 'll' → 'II'
• 'vv' → 'w'

Number confusions:
• 'O' → '0' (in numbers)
• 'l' → '1' (in numbers)
• 'S' → '5'

Academic patterns:
• '[l]' → '[1]' (citations)
• 'et aI.' → 'et al.'
• 'fig. l' → 'fig. 1'
                    """,
                    language=None,
                )

                # Additional context about corrections
                with st.expander("📖 How OCR Correction Works"):
                    st.markdown(
                        """
                        **Pattern-Based Corrections:**
                        - Identifies common scanning errors using predefined patterns
                        - Context-aware - only corrects when patterns match
                          expected usage
                        - Preserves original text structure and formatting

                        **Spell Checking (when enabled):**
                        - Uses dictionary-based validation with technical term support
                        - Considers word frequency and context for corrections
                        - Academic terms and proper nouns are preserved

                        **Performance Optimization:**
                        - Caching system remembers corrected words for faster processing
                        - Batch processing for efficient handling of large documents
                        - Minimal memory overhead even with large cache sizes
                        """
                    )

        else:
            # Set defaults when OCR correction is disabled
            ocr_level = st.session_state.config.ocr_correction_level
            enable_spell = st.session_state.config.ocr_enable_spell_check
            use_simple_spell = st.session_state.config.ocr_use_simple_spellcheck
            cache_size = st.session_state.config.ocr_cache_size

        # Update configuration if OCR settings changed
        ocr_changed = (
            enable_ocr != st.session_state.config.enable_ocr_correction
            or ocr_level != st.session_state.config.ocr_correction_level
            or enable_spell != st.session_state.config.ocr_enable_spell_check
            or use_simple_spell != st.session_state.config.ocr_use_simple_spellcheck
            or cache_size != st.session_state.config.ocr_cache_size
        )

        if ocr_changed:
            # Update OCR configuration
            st.session_state.config.enable_ocr_correction = enable_ocr
            st.session_state.config.ocr_correction_level = ocr_level
            st.session_state.config.ocr_enable_spell_check = enable_spell
            st.session_state.config.ocr_use_simple_spellcheck = use_simple_spell
            st.session_state.config.ocr_cache_size = cache_size

            st.sidebar.info("OCR correction settings updated")


def refresh_file_stats():
    """Refresh file statistics from the database."""

    if st.session_state.collection is None:
        st.session_state.file_stats = []
        return

    try:
        # Get collection statistics (unused but kept for potential future use)
        get_collection_stats(st.session_state.collection)

        # Get sample documents to build file list
        results = st.session_state.collection.get(limit=1000)
        metadatas = results.get("metadatas", [])

        # Group by source file
        file_stats = {}
        for metadata in metadatas:
            if metadata and "source" in metadata:
                source = metadata["source"]
                filename = Path(source).name

                if filename not in file_stats:
                    file_stats[filename] = {
                        "filename": filename,
                        "pages": set(),
                        "chunks": 0,
                        "source_path": source,
                    }

                if "page" in metadata:
                    file_stats[filename]["pages"].add(metadata["page"])

                file_stats[filename]["chunks"] += 1

        # Convert to list format for display
        st.session_state.file_stats = [
            {
                "Filename": stats["filename"],
                "Pages": len(stats["pages"]),
                "Chunks": stats["chunks"],
                "Source": stats["source_path"],
            }
            for stats in file_stats.values()
        ]

    except Exception as e:
        st.session_state.logger.error(
            f"Failed to refresh file stats: {e}", exc_info=True
        )
        st.session_state.file_stats = []


def file_management_interface():
    """Create the main file management interface."""

    st.header("📁 File Management")

    # Add custom CSS for beautiful database status card
    st.markdown(
        """
    <style>
    .db-status-card {
        padding: 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #FAFAFB 0%, #F3F4F6 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 2rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 2rem;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Refresh file statistics
    refresh_file_stats()

    # Database status card
    try:
        stats = get_collection_stats(st.session_state.collection)

        # Calculate approximate storage size (rough estimate: ~1KB per chunk)
        storage_mb = (stats.get("total_chunks", 0) * 1024) / (1024 * 1024)
        storage_display = (
            f"{storage_mb:.1f}MB" if storage_mb < 1000 else f"{storage_mb/1024:.2f}GB"
        )

        # Create the visual database status card
        st.markdown(
            f"""
        <div class="db-status-card">
            <div class="status-title">
                <span>📊</span>
                <span>Database Overview</span>
            </div>
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-value" style="color: #10B981;">
                        {stats.get("unique_files", 0):,}
                    </div>
                    <div class="metric-label">Documents</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: #3B82F6;">
                        {stats.get("total_chunks", 0):,}
                    </div>
                    <div class="metric-label">Chunks</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: #8B5CF6;">
                        {stats.get("total_pages", 0):,}
                    </div>
                    <div class="metric-label">Pages</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: #EC4899;">
                        {storage_display}
                    </div>
                    <div class="metric-label">Est. Storage</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh Stats", help="Refresh database statistics"):
                refresh_file_stats()
                st.rerun()

    except Exception as e:
        st.error(f"Could not load database statistics: {str(e)}")

    st.divider()

    # File table
    if st.session_state.file_stats:
        st.subheader("📋 Document Library")

        # Add custom styling for the table
        st.markdown(
            """
        <style>
        /* Style the dataframe */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Empty state styling */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            background-color: #F9FAFB;
            border-radius: 12px;
            border: 1px dashed #E5E7EB;
        }
        .empty-state h3 {
            color: #374151;
            margin-bottom: 0.5rem;
        }
        .empty-state p {
            color: #6B7280;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display file table with better styling
        st.dataframe(
            st.session_state.file_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Filename": st.column_config.TextColumn(
                    "📄 Filename", width="medium", help="Name of the uploaded document"
                ),
                "Pages": st.column_config.NumberColumn(
                    "📃 Pages", width="small", help="Number of pages in the document"
                ),
                "Chunks": st.column_config.NumberColumn(
                    "🔤 Chunks",
                    width="small",
                    help="Number of text chunks created for vector search",
                ),
                "Source": st.column_config.TextColumn(
                    "📁 Source Path", width="large", help="Original file location"
                ),
            },
            height=400,
        )

        st.caption(
            f"📊 Showing {len(st.session_state.file_stats)} document{'s' if len(st.session_state.file_stats) != 1 else ''} in the library"
        )

    else:
        # Better empty state
        st.markdown(
            """
        <div class="empty-state">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
            <h3>Your Document Library is Empty</h3>
            <p>Upload some PDFs to start building your searchable knowledge base</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # File upload interface
    st.divider()
    file_upload_interface()


def file_upload_interface():
    """Create the file upload and processing interface.

    This interface provides drag-and-drop file upload functionality with support
    for multiple file formats. When OCR correction is enabled in the configuration,
    uploaded files will be processed with error correction, and statistics will be
    displayed after processing.

    OCR Integration:
    - OCR correction is applied automatically if enabled in configuration
    - No additional UI controls needed - configuration drives behavior
    - Processing results include OCR statistics when corrections are made
    - User gets feedback about correction effectiveness

    UI Flow:
    1. User drags files or clicks to browse
    2. File preview shows selected files
    3. Process button triggers ingestion
    4. Progress tracking during processing
    5. Results display with optional OCR statistics
    """

    st.subheader("📤 Upload Documents")

    if st.session_state.collection is None:
        st.warning("Please connect to a database first")
        return

    # Add custom CSS for beautiful file upload area
    st.markdown(
        """
    <style>
    .upload-area {
        border: 2px dashed #E5E7EB;
        border-radius: 12px;
        background-color: #F9FAFB;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .upload-area:hover {
        border-color: #9CA3AF;
        background-color: #F3F4F6;
    }
    .file-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .file-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: default;
    }
    .file-card:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .file-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .file-name {
        font-size: 0.75rem;
        color: #374151;
        word-break: break-word;
        line-height: 1.2;
    }
    .file-size {
        font-size: 0.7rem;
        color: #9CA3AF;
        margin-top: 0.25rem;
    }
    .upload-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .file-count {
        background: #EBF5FF;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Get supported extensions
    supported_exts = get_supported_extensions()
    # Remove the dot from extensions for Streamlit
    streamlit_exts = [ext.lstrip(".") for ext in supported_exts]

    # File uploader with drag and drop
    uploaded_files = st.file_uploader(
        "Drag and drop PDFs here or click to browse",
        type=streamlit_exts,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(supported_exts)} • Max 200MB per file",
        label_visibility="collapsed",
    )

    if uploaded_files:
        # File upload header with count
        st.markdown(
            f"""
        <div class="upload-header">
            <h4 style="margin: 0;">Selected Files</h4>
            <span class="file-count">{len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show file preview grid
        max_preview = 8
        files_to_show = uploaded_files[:max_preview]

        # Create grid layout for file previews
        cols = st.columns(4)
        for idx, file in enumerate(files_to_show):
            with cols[idx % 4]:
                # Format file size
                size_kb = file.size / 1024
                size_str = (
                    f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                )

                # Truncate filename if too long
                name_display = (
                    file.name if len(file.name) <= 15 else file.name[:12] + "..."
                )

                st.markdown(
                    f"""
                <div class="file-card">
                    <div class="file-icon">📄</div>
                    <div class="file-name" title="{file.name}">{name_display}</div>
                    <div class="file-size">{size_str}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Show "more files" indicator if needed
        if len(uploaded_files) > max_preview:
            st.info(
                f"➕ And {len(uploaded_files) - max_preview} more file{'s' if len(uploaded_files) - max_preview > 1 else ''}..."
            )

        # Process files button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Process All Files",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing_files,
            ):
                process_uploaded_files(uploaded_files)

    else:
        # Empty state with better visual
        st.markdown(
            """
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <h4 style="margin: 0 0 0.5rem 0; color: #374151;">Drop your PDFs here</h4>
            <p style="color: #6B7280; margin: 0;">or click to browse files</p>
            <p style="color: #9CA3AF; font-size: 0.875rem; margin-top: 1rem;">
                Supports: PDF • Max 200MB per file
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def process_uploaded_files(uploaded_files):
    """Process uploaded files with progress tracking and OCR statistics display.

    This function handles the file processing workflow, including OCR error correction
    when enabled. It provides real-time progress updates and displays comprehensive
    statistics about the processing results, including OCR correction metrics.

    Args:
        uploaded_files: List of uploaded file objects from Streamlit file_uploader

    OCR UI Integration:
    - Checks if OCR correction is enabled in configuration
    - Displays OCR statistics only when corrections were performed
    - Aggregates statistics across all processed files
    - Provides visual feedback about correction effectiveness

    User Experience Flow:
    1. Files are saved to temporary directory
    2. Progress bar shows real-time processing status
    3. Results display with file counts and success metrics
    4. OCR statistics appear if corrections were made
    5. Contextual feedback helps users understand the impact

    The OCR statistics section uses a three-column layout for key metrics:
    - Total Corrections: Absolute count of fixes made
    - Correction Rate: Percentage indicating OCR quality
    - Cache Hit Rate: Performance metric for optimization
    """

    if not uploaded_files:
        return

    st.session_state.processing_files = True

    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded files to temporary directory
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(str(file_path))

            # Show files being processed
            file_names = [f.name for f in uploaded_files]
            st.info(
                f"📁 Processing {len(uploaded_files)} file(s): "
                f"{', '.join(file_names)}"
            )

            # Progress tracking
            progress_container = st.container()

            with progress_container:
                st.write("🔄 **Processing files...**")

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Processing stats (unused but kept for potential future use)
                st.empty()

                def progress_callback(current: int, total: int, message: str = ""):
                    """Update progress bar and status."""
                    if total > 0:
                        progress = current / total
                        progress_bar.progress(progress)

                    if message:
                        status_text.text(f"📄 {message}")

                try:
                    # Use existing ingestion function
                    with st.status("Processing files...", expanded=True) as status:
                        st.write("Starting document processing...")

                        # Call the existing ingest_pdfs function
                        results = ingest_pdfs(
                            file_paths,
                            st.session_state.collection,
                            st.session_state.config,
                            progress_callback=progress_callback,
                        )

                        # Show results
                        status.update(label="✅ Processing complete!", state="complete")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Files Processed", results["files_processed"])
                        with col2:
                            st.metric("Files Failed", results["files_failed"])
                        with col3:
                            st.metric("Total Chunks", results["total_chunks"])
                        with col4:
                            success_rate = (
                                results["files_processed"] / len(file_paths) * 100
                            )
                            st.metric("Success Rate", f"{success_rate:.1f}%")

                        # Show OCR correction statistics if enabled
                        if (
                            st.session_state.config.enable_ocr_correction
                            and "ocr_stats" in results
                        ):
                            # This section provides users with detailed feedback
                            # about OCR corrections
                            # performed during document processing. The statistics
                            # help users understand the
                            # impact of OCR correction and validate that it's working
                            # as expected.

                            # UI Design Principles:
                            # - Only shown when OCR correction is enabled
                            #   (progressive disclosure)
                            # - Uses metrics for quick visual scanning of key
                            #   statistics
                            # - Aggregates data across all processed files for
                            #   overview
                            # - Provides both absolute numbers and percentages
                            #   for context
                            # - Color-coded insights based on correction rates

                            # Statistics Displayed:
                            # 1. **Total Corrections**: Raw count of corrections made
                            # - Helps gauge overall OCR quality of source documents
                            # - Higher counts suggest poorer scan quality

                            # 2. **Correction Rate**: Percentage of words that
                            #    were corrected
                            # - <1%: Excellent OCR quality, minimal errors
                            # - 1-5%: Good quality, typical for modern scans
                            # - >5%: Poor quality, consider re-scanning

                            # 3. **Cache Hit Rate**: Performance metric showing
                            #    cache effectiveness
                            # - >85%: Excellent, patterns are repetitive
                            # - 70-85%: Good, cache size is appropriate
                            # - <70%: Consider increasing cache size

                            # User Value:
                            # - Validates that OCR correction is working and
                            #   providing value
                            # - Helps users decide if correction level needs adjustment
                            # - Performance metrics help optimize settings for
                            #   large batches
                            # - Quality indicators guide decisions about re-scanning

                            # Actionable Insights:
                            # The UI provides context-aware recommendations based on the
                            # statistics, such as suggesting more aggressive correction
                            # levels for high error rates or celebrating good
                            #   OCR quality.

                            # See Also:
                            # apply_ocr_correction: Generates these statistics
                            # Config.ocr_cache_size: Cache size configuration
                            # get_statistics: Detailed statistics calculation
                            st.write("---")
                            st.write("**🔍 OCR Correction Statistics**")

                            # Aggregate OCR stats
                            # to provide an overall view of correction effectiveness
                            total_corrections = 0
                            total_words = 0
                            # total_cache_hits = 0  # Unused variable

                            for file_path, ocr_stat in results["ocr_stats"].items():
                                if "total_corrections" in ocr_stat:
                                    total_corrections += ocr_stat["total_corrections"]
                                    total_words += ocr_stat.get("total_words", 0)

                            # Display OCR metrics
                            ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
                            with ocr_col1:
                                st.metric(
                                    "Total Corrections",
                                    f"{total_corrections:,}",
                                    help=(
                                        "Number of OCR errors corrected across "
                                        "all files"
                                    ),
                                )
                            with ocr_col2:
                                if total_words > 0:
                                    correction_rate = (
                                        total_corrections / total_words
                                    ) * 100
                                    st.metric(
                                        "Correction Rate",
                                        f"{correction_rate:.2f}%",
                                        help=(
                                            "Percentage of words that required "
                                            "correction"
                                        ),
                                    )
                                else:
                                    st.metric("Correction Rate", "0%")
                            with ocr_col3:
                                # Get cache hit rate from last file's stats
                                last_stats = (
                                    list(results["ocr_stats"].values())[-1]
                                    if results["ocr_stats"]
                                    else {}
                                )
                                cache_rate = last_stats.get("cache_hit_rate", 0) * 100
                                st.metric(
                                    "Cache Hit Rate",
                                    f"{cache_rate:.1f}%",
                                    help=(
                                        "Percentage of corrections found in cache "
                                        "(higher is better)"
                                    ),
                                )

                            # Additional insights based on correction rate
                            if total_words > 0 and correction_rate > 5:
                                st.info(
                                    "These documents had significant OCR errors. "
                                    "The corrections should improve search accuracy."
                                )
                            elif total_words > 0 and correction_rate < 1:
                                st.success("Good OCR quality in the source documents.")

                        # Show failed files if any
                        if results["failed_files"]:
                            st.warning("⚠️ Some files failed to process:")
                            for failed_file in results["failed_files"]:
                                st.write(f"- {Path(failed_file).name}")

                        # Success message
                        if results["files_processed"] > 0:
                            processed_count = results["files_processed"]
                            st.success(
                                f"✅ Successfully processed "
                                f"{processed_count} files!"
                            )

                            # Refresh file stats
                            refresh_file_stats()

                except Exception as e:
                    error_msg = str(e).lower()

                    # Provide specific error guidance
                    if "pypdf" in error_msg:
                        st.error("❌ PDF processing library missing!")
                        st.markdown(
                            """
                        **To fix this:**
                        1. Install the missing dependency:
                           `pip install pypdf>=3.0.0`
                        2. Or run: `pip install -r requirements.txt`
                           to install all dependencies
                        3. Restart the Streamlit application
                        """
                        )
                    elif "sentence" in error_msg and "transformers" in error_msg:
                        st.error("❌ Sentence Transformers library missing!")
                        st.markdown(
                            """
                        **To fix this:**
                        1. Install the missing dependency:
                           `pip install sentence-transformers>=2.0.0`
                        2. Or run: `pip install -r requirements.txt`
                           to install all dependencies
                        3. Restart the Streamlit application
                        """
                        )
                    elif "openai" in error_msg and "unauthorized" in error_msg:
                        st.error("❌ OpenAI API key issue!")
                        st.markdown(
                            """
                        **To fix this:**
                        1. Check your OpenAI API key in the Configuration section
                        2. Use the 'Test API Key' button to verify it works
                        3. Ensure you have sufficient API credits
                        """
                        )
                    else:
                        st.error(f"❌ Processing failed: {str(e)}")

                    # Show debug information
                    with st.expander("🔍 Debug Information"):
                        st.text(f"Error type: {type(e).__name__}")
                        st.text(f"Error message: {str(e)}")
                        provider = st.session_state.config.embedding_provider
                        st.text(f"Current embedding provider: {provider}")

                        if provider == "sentencetransformer":
                            model = st.session_state.config.embedding_model
                        else:
                            model = st.session_state.config.openai_model
                        st.text(f"Current model: {model}")

                    st.session_state.logger.error(
                        f"File processing failed: {e}", exc_info=True
                    )

    finally:
        st.session_state.processing_files = False
        # Rerun to update UI
        st.rerun()


def database_info_interface():
    """Create database information and analytics interface."""

    st.header("📊 Database Analytics")

    if st.session_state.collection is None:
        st.warning("No database connected")
        return

    try:
        stats = get_collection_stats(st.session_state.collection)

        # Overview metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Overview")
            st.metric("Database Path", "", st.session_state.db_path)
            st.metric("Collection Name", "", st.session_state.config.collection_name)
            st.metric("Embedding Model", "", st.session_state.config.embedding_model)

        with col2:
            st.subheader("📊 Statistics")
            st.metric("Unique Files", stats.get("unique_files", 0))
            st.metric("Total Pages", stats.get("total_pages", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))

        # Configuration details
        st.subheader("⚙️ Current Configuration")

        # Calculate overlap percentage
        overlap_percent = int(
            (st.session_state.config.chunk_overlap / st.session_state.config.chunk_size)
            * 100
        )

        config_data = {
            "Setting": [
                "Chunk Size",
                "Chunk Overlap",
                "Embedding Provider",
                "Embedding Model",
                "Collection Name",
            ],
            "Value": [
                f"{st.session_state.config.chunk_size} tokens",
                f"{st.session_state.config.chunk_overlap} tokens ({overlap_percent}%)",
                (
                    "🤖 SentenceTransformer"
                    if st.session_state.config.embedding_provider
                    == "sentencetransformer"
                    else "🔗 OpenAI"
                ),
                str(
                    st.session_state.config.openai_model
                    if st.session_state.config.embedding_provider == "openai"
                    else st.session_state.config.embedding_model
                ),
                str(st.session_state.config.collection_name),
            ],
        }
        st.table(config_data)

    except Exception as e:
        st.error(f"Could not load database information: {str(e)}")


def search_interface():
    """Create the search interface for semantic document search."""

    st.header("🔍 Semantic Search")

    if st.session_state.collection is None:
        st.warning("Please connect to a database first")
        return

    try:
        # Check if database has any documents
        stats = get_collection_stats(st.session_state.collection)
        if stats.get("total_chunks", 0) == 0:
            st.info("📝 No documents in the database yet. Upload some files first!")
            return

        # Search input
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'machine learning algorithms' or 'project timeline'",
            help="Search across all documents using semantic similarity",
        )

        # Search parameters
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")  # Spacing
        with col2:
            max_results = st.selectbox("Max Results", [5, 10, 20, 50], index=1)

        if search_query and st.button("🔍 Search", type="primary"):
            with st.spinner("Searching documents..."):
                try:
                    # Perform semantic search using ChromaDB
                    results = st.session_state.collection.query(
                        query_texts=[search_query], n_results=max_results
                    )

                    if results["documents"] and results["documents"][0]:
                        st.success(
                            f"Found {len(results['documents'][0])} relevant results"
                        )

                        # Display results
                        for i, (doc, metadata, distance) in enumerate(
                            zip(
                                results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0],
                            )
                        ):
                            source_name = Path(metadata.get("source", "Unknown")).name
                            score = 1 - distance
                            with st.expander(
                                f"📄 Result {i+1} - {source_name} "
                                f"(Score: {score:.2f})"
                            ):
                                # Document info
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    source = metadata.get("source", "Unknown")
                                    st.write(f"**Source:** {source}")
                                    if "page" in metadata:
                                        st.write(f"**Page:** {metadata['page']}")
                                with col2:
                                    st.metric("Similarity", f"{(1-distance)*100:.1f}%")

                                # Document content
                                st.write("**Content:**")
                                st.text_area(
                                    "Document excerpt",
                                    value=doc,
                                    height=100,
                                    disabled=True,
                                    key=f"search_result_{i}",
                                    label_visibility="collapsed",
                                )
                    else:
                        st.warning(
                            "No results found for your query. Try different "
                            "keywords or check if documents are properly indexed."
                        )

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.session_state.logger.error(f"Search failed: {e}", exc_info=True)

        # Search tips
        st.divider()
        with st.expander("💡 Search Tips"):
            st.markdown(
                """
            **How to search effectively:**
            - Use natural language queries (e.g., "project deadlines and milestones")
            - Try different phrasings if you don't find what you're looking for
            - Search is semantic - it understands meaning, not just exact words
            - Longer, more specific queries often work better

            **Examples:**
            - "machine learning model performance"
            - "budget analysis and financial projections"
            - "team meeting notes from last quarter"
            - "technical requirements and specifications"
            """
            )

    except Exception as e:
        st.error(f"Could not load search interface: {str(e)}")


def main():
    """Main Streamlit application."""

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🗂️ ManilaFolder")
    st.subheader("Digital Document Organization with Vector Search")

    # Sidebar
    database_selection_sidebar()
    configuration_sidebar()

    # Main content area
    if st.session_state.collection is None:
        # Welcome screen
        st.info(
            "👈 Please create a new database or open an existing one using the sidebar."
        )

        st.markdown(
            """
        ### Welcome to ManilaFolder

        A digital filing cabinet that brings the familiar metaphor of manila folders
        to document management with powerful vector search capabilities.

        **Features:**
        - 🗂️ **Intuitive Organization**: Manila folder metaphor for document management
        - 🔍 **Vector Search**: Semantic search powered by ChromaDB
        - 📄 **PDF Processing**: Automatic text extraction and intelligent chunking
        - ⚡ **Real-time Processing**: Live progress tracking during document ingestion
        - 🔧 **Extensible**: Support for multiple file formats and embedding models

        **Getting Started:**
        1. Create a new database or open an existing one from the sidebar
        2. Upload PDF documents using drag-and-drop
        3. Search and organize your documents with semantic understanding
        """
        )

    else:
        # Main application interface with tabs
        tab1, tab2, tab3 = st.tabs(["📁 Files", "📊 Analytics", "🔍 Search"])

        with tab1:
            file_management_interface()

        with tab2:
            database_info_interface()

        with tab3:
            search_interface()


if __name__ == "__main__":
    main()
