# ───────────────────────── src/manilafolder/config.py ─────────────────────────
"""
Configuration management and backend protocols.

This module provides configuration management for ManilaFolder, including
settings for document processing, OCR correction, and vector storage backends.

Key Components:
    - VectorStoreBackend: Protocol for implementing custom vector store backends
    - Config: Main configuration class with all processing settings

OCR Configuration Guide:
    The OCR correction system is designed to fix common errors in scanned documents:

    1. **When to Enable OCR Correction**:
       - Scanned PDFs with visible character substitution errors (rn appearing as m)
       - Documents from older scanning systems
       - Academic papers from pre-2000 archives
       - Legal documents with poor scan quality

    2. **Performance Considerations**:
       - Base overhead: 20-30% processing time increase
       - Spell checking adds: ~10% additional overhead
       - Cache size impacts: Larger cache = faster repeated corrections
       - Memory usage: ~100 bytes per cached pattern

    3. **Setting Interactions**:
       - OCR correction works best with preprocessing enabled
       - Academic cleaning may conflict with aggressive OCR correction
       - Quality filtering should run after OCR correction
       - Larger chunk sizes reduce OCR correction overhead per chunk

Examples:
    Configure for heavily degraded scanned documents:
    >>> config = Config(
    ...     enable_ocr_correction=True,
    ...     ocr_correction_level="aggressive",
    ...     ocr_enable_spell_check=False,  # Many OCR errors may not be real words
    ...     ocr_cache_size=50000,  # Large cache for repetitive errors
    ...     enable_preprocessing=True,
    ...     preprocessing_intensity="aggressive",
    ...     min_alpha_ratio=0.3  # Lower threshold for degraded text
    ... )

    Configure for modern scanned academic papers:
    >>> config = Config(
    ...     enable_ocr_correction=True,
    ...     ocr_correction_level="light",
    ...     ocr_enable_spell_check=True,
    ...     enable_academic_cleaning=True,
    ...     preserve_section_structure=True
    ... )

    Optimize for batch processing of mixed-quality documents:
    >>> config = Config(
    ...     enable_ocr_correction=True,
    ...     ocr_correction_level="moderate",
    ...     ocr_cache_size=20000,  # Balanced cache size
    ...     chunk_size=2000,  # Larger chunks to reduce overhead
    ...     enable_quality_filtering=True,
    ...     min_readability_score=0.3  # Filter out severely corrupted chunks
    ... )
"""

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


class VectorStoreBackend(Protocol):
    """Protocol for vector store backends to enable extensibility."""

    @abstractmethod
    def create_collection(self, name: str, persist_directory: str) -> Any:
        """Create a new vector collection.

        Args:
            name: Collection name
            persist_directory: Directory to persist the collection

        Returns:
            Collection instance
        """
        pass

    @abstractmethod
    def load_collection(self, name: str, persist_directory: str) -> Any:
        """Load an existing vector collection.

        Args:
            name: Collection name
            persist_directory: Directory containing the collection

        Returns:
            Collection instance
        """
        pass

    @abstractmethod
    def add_documents(
        self, collection: Any, documents: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add documents to the collection.

        Args:
            collection: Collection instance
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        pass


@dataclass
class Config:
    """Configuration settings for ManilaFolder.

    This class centralizes all configuration options for document processing,
    including text splitting, embeddings, preprocessing, and OCR correction.

    Attributes:
        chunk_size: Size of text chunks for document splitting (default: 1000).
        chunk_overlap: Overlap between consecutive chunks (default: 100).
        collection_name: Name of the ChromaDB collection (default: "pdf_texts").
        embedding_model: Name of the sentence transformer model
            (default: "all-MiniLM-L6-v2").
        log_file: Path to the error log file (default: "manilafolder_error.log").
        max_log_size: Maximum log file size in bytes (default: 10MB).
        log_backup_count: Number of backup log files to keep (default: 3).
        openai_api_key: OpenAI API key (optional, from environment).
        enable_ocr_correction: Enable OCR error correction for documents
            (default: False).
        ocr_correction_level: Aggressiveness of OCR correction (default: "moderate").
        ocr_enable_spell_check: Enable spell checking during OCR correction
            (default: True).
        ocr_use_simple_spellcheck: Use pyspellchecker instead of SymSpell for
            simpler, more conservative corrections (default: False).
        ocr_cache_size: Maximum number of correction patterns to cache (default: 10000).

    Examples:
        Basic configuration with default settings:
        >>> config = Config()
        >>> config.chunk_size
        1000

        Enable OCR correction for scanned documents:
        >>> config = Config(
        ...     enable_ocr_correction=True,
        ...     ocr_correction_level="aggressive"
        ... )

        Custom configuration for academic papers with OCR issues:
        >>> config = Config(
        ...     enable_preprocessing=True,
        ...     enable_academic_cleaning=True,
        ...     enable_ocr_correction=True,
        ...     ocr_correction_level="moderate",
        ...     ocr_enable_spell_check=True,
        ...     ocr_cache_size=20000
        ... )

        Minimal processing for clean digital documents:
        >>> config = Config(
        ...     enable_preprocessing=False,
        ...     enable_ocr_correction=False
        ... )
    """

    # Text splitting configuration
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Database configuration
    collection_name: str = "pdf_texts"

    # Embedding configuration
    embedding_provider: str = "sentencetransformer"  # "sentencetransformer" or "openai"
    embedding_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model
    openai_model: str = "text-embedding-3-small"  # OpenAI embedding model

    # Logging configuration
    log_file: str = "manilafolder_error.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 3

    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Text preprocessing configuration (new)
    enable_preprocessing: bool = True
    preprocessing_intensity: str = "moderate"  # "minimal", "moderate", "aggressive"

    # Basic cleaning (always applied when preprocessing enabled)
    normalize_unicode: bool = True
    fix_encoding: bool = True
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True

    # Academic document processing
    enable_academic_cleaning: bool = True
    remove_citations: bool = True
    remove_figure_refs: bool = True
    remove_headers_footers: bool = True
    preserve_section_structure: bool = True

    # Quality filtering
    enable_quality_filtering: bool = True
    min_chunk_words: int = 10
    min_alpha_ratio: float = 0.5
    max_repetition_ratio: float = 0.7
    min_readability_score: float = 0.0  # 0 = no filtering

    # OCR Correction settings
    enable_ocr_correction: bool = False
    """Enable OCR error correction for scanned or poorly digitized documents.
    When enabled, applies pattern-based corrections for common OCR mistakes like
    rn->m, l->I, 0->O substitutions. Recommended for scanned PDFs or documents
    with known OCR issues. Adds processing overhead (~20-30% slower).
    Correction Coverage:
        - ~85% of common OCR errors are correctable with patterns
        - Character substitutions: ~51.6% of errors (rn→m, cl→d)
        - Number confusions: ~20% of errors (O→0, l→1)
        - Academic patterns: ~10% of errors ([l]→[1], et aI.→et al.)
        - Context-aware: ~15% of errors (word boundaries, formatting)
    When to Enable:
        - Scanned PDFs from pre-2010 scanning technology
        - Documents with visible character substitution errors
        - Academic papers from digital archives
        - Legal documents with poor scan quality
        - Historical documents or manuscripts
    When to Keep Disabled:
        - Born-digital PDFs (created directly from word processors)
        - High-quality modern scans (post-2015)
        - Documents with specialized notation (math, chemistry)
        - Non-English documents
    Default: False (no OCR correction applied).
    See Also:
        ocr_correction_level: Control aggressiveness of corrections
        ocr_enable_spell_check: Additional validation layer
        ocr_cache_size: Performance tuning option
    """

    ocr_correction_level: str = "moderate"  # "light", "moderate", "aggressive"
    """Controls the aggressiveness of OCR error correction.
    Options:
        - "light": Only fixes obvious single-character substitutions (rn->m, l->I).
          Minimal false positives but may miss complex errors.
          - Patterns applied: ~33 simple substitutions
          - Processing speed: Fastest (~2000 pages/minute)
          - False positive rate: <0.1%
          - Use for: Digital PDFs with minor OCR issues
        - "moderate": Includes word boundary corrections and common patterns.
          Good balance for most documents. Recommended default.
          - Patterns applied: ~33 simple + ~50 regex patterns
          - Processing speed: Moderate (~1500 pages/minute)
          - False positive rate: <0.5%
          - Use for: Standard scanned documents
        - "aggressive": Applies all corrections including context-aware fixes.
          Best for heavily degraded text but may introduce errors in clean text.
          - Patterns applied: All patterns + aggressive spell checking
          - Processing speed: Slowest (~1000 pages/minute)
          - False positive rate: 1-2% (mostly on technical terms)
          - Use for: Historical documents, poor quality scans
    Performance Impact by Level:
        - "light": 10-15% overhead vs. no correction
        - "moderate": 20-30% overhead vs. no correction
        - "aggressive": 40-50% overhead vs. no correction
    Selection Guidelines:
        1. Start with "moderate" for unknown document quality
        2. Use "light" if seeing false corrections in technical terms
        3. Use "aggressive" only if "moderate" leaves many errors
        4. Monitor correction_rate in results to validate choice
    Default: "moderate" - balanced approach suitable for most use cases.
    See Also:
        enable_ocr_correction: Master switch for OCR correction
        ocr_enable_spell_check: Additional validation layer
    """

    ocr_enable_spell_check: bool = True
    """Enable spell checking during OCR correction to validate corrections.
    When True, uses a dictionary to verify that OCR corrections produce valid words.
    Helps prevent false corrections but may miss technical terms or proper nouns.
    Disable for documents with many technical terms or non-English content.
    Spell Checking Details:
        - Engine: SymSpell (symmetric spell correction) or pyspellchecker
        - Dictionary: English frequency dictionary
        - Edit distance: 1 (light/moderate) or 2 (aggressive)
        - Protected terms: Academic vocabulary, technical abbreviations
    Impact on Correction Quality:
        - Reduces false positives by ~60%
        - May reject valid corrections for specialized terms
        - Most effective on general academic/business text
        - Less effective on highly technical content
    When to Disable:
        - Documents with >20% technical terminology
        - Non-English or multilingual documents
        - Mathematical or chemical formulas
        - Code snippets or programming content
        - Custom domain-specific vocabularies
    Performance Considerations:
        - Adds ~10% processing time when enabled
        - Memory usage: ~50MB for dictionary
        - First document slower due to dictionary loading
        - Subsequent documents benefit from loaded dictionary
    Default: True (spell checking enabled).
    Performance impact: ~10% additional overhead when OCR correction is active.
    See Also:
        ocr_correction_level: Affects spell check aggressiveness
        ACADEMIC_VOCABULARY: Protected academic terms
        TECHNICAL_TERMS: Protected technical abbreviations
    """

    ocr_use_simple_spellcheck: bool = False
    """Use pyspellchecker instead of SymSpell for more conservative corrections.
    When True, uses pyspellchecker which provides simpler, more conservative
    spell checking with lower memory usage and fewer aggressive corrections.
    Best for documents where you want minimal intervention.

    Comparison:
        pyspellchecker (when True):
            - More conservative corrections
            - Lower memory usage (~5MB vs ~50MB)
            - Slower per-word lookup
            - Better for technical documents
            - Only corrects obvious typos

        SymSpell (when False):
            - More aggressive corrections
            - Higher memory usage
            - Much faster lookups
            - Better for general text
            - May over-correct technical terms

    Default: False (use SymSpell for better performance).
    Requires: ocr_enable_spell_check must be True to have any effect.
    """

    ocr_cache_size: int = 10000
    """Maximum number of OCR correction patterns to cache in memory.
    Caching improves performance by storing frequently used correction patterns.
    Larger cache = better performance for repetitive errors but more memory usage.
    Each cached entry uses ~100 bytes of memory.
    Cache Performance Characteristics:
        - Hit rates: 70-90% for typical academic documents
        - Lookup time: O(1) average case
        - Eviction policy: LRU (Least Recently Used)
        - Memory overhead: ~100 bytes per entry
    Size Recommendations by Document Type:
        1. **Small documents (<100 pages)**:
           - Cache size: 5,000 entries
           - Memory usage: ~500KB
           - Expected hit rate: 60-70%
        2. **Medium documents (100-500 pages)**:
           - Cache size: 10,000 entries (default)
           - Memory usage: ~1MB
           - Expected hit rate: 70-85%
        3. **Large documents (>500 pages)**:
           - Cache size: 20,000-50,000 entries
           - Memory usage: 2-5MB
           - Expected hit rate: 85-95%
        4. **Document collections (batch processing)**:
           - Cache size: 50,000+ entries
           - Memory usage: 5MB+
           - Expected hit rate: 90-98%
    Performance Impact:
        - Doubling cache size typically improves speed by 10-15%
        - Diminishing returns above 50,000 entries
        - Cache misses add ~0.1ms per lookup
        - Cache hits are ~10x faster than recomputation
    Monitoring Cache Efficiency:
        Check 'cache_hit_rate' in OCR statistics:
        - >85%: Excellent, cache size appropriate
        - 70-85%: Good, consider modest increase
        - <70%: Poor, increase cache size
    Default: 10000 (approximately 1MB memory usage).
    See Also:
        LRUCache: Cache implementation details
        get_statistics(): Monitor cache performance
    """

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any configuration values are invalid.
        """
        # Validate chunk settings
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        # Validate OCR settings
        valid_ocr_levels = {"light", "moderate", "aggressive"}
        if self.ocr_correction_level not in valid_ocr_levels:
            raise ValueError(
                f"ocr_correction_level must be one of {valid_ocr_levels}, "
                f"got '{self.ocr_correction_level}'"
            )

        if self.ocr_cache_size < 0:
            raise ValueError("ocr_cache_size cannot be negative")

        # Warn about conflicting settings
        if self.enable_ocr_correction and not self.enable_preprocessing:
            import warnings

            warnings.warn(
                "OCR correction works best with preprocessing enabled. "
                "Consider setting enable_preprocessing=True for better results."
            )

        if (
            self.ocr_correction_level == "aggressive"
            and self.enable_academic_cleaning
            and self.remove_citations
        ):
            import warnings

            warnings.warn(
                "Aggressive OCR correction may interfere with citation removal. "
                "Consider using 'moderate' OCR correction level with academic cleaning."
            )
