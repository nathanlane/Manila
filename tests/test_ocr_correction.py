# ───────────────────────── tests/test_ocr_correction.py ─────────────────────────
"""
Tests for OCR error correction functionality.
"""

import pytest
from langchain.schema import Document

from src.manilafolder.ocr_correction import FastOCRCorrector, LRUCache, correct_document
from src.manilafolder.ocr_patterns import (
    get_all_simple_patterns,
    get_pattern_statistics,
)


class TestLRUCache:
    """Test the LRU cache implementation."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(maxsize=3)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

        # Test LRU eviction
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        """Test that LRU ordering is maintained."""
        cache = LRUCache(maxsize=3)

        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")

        # Access 'a' to make it most recently used
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.set("d", "4")

        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("c") == "3"
        assert cache.get("d") == "4"

    def test_resize(self):
        """Test cache resizing."""
        cache = LRUCache(maxsize=5)

        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # Resize to smaller
        cache.resize(2)

        # Only the 2 most recent should remain
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"


class TestFastOCRCorrector:
    """Test the FastOCRCorrector class."""

    def test_initialization(self):
        """Test corrector initialization."""
        corrector = FastOCRCorrector(
            correction_level="moderate",
            enable_spell_check=False,  # Disable to avoid dictionary issues
        )

        assert corrector.correction_level == "moderate"
        assert corrector.enable_spell_check is False
        assert corrector.word_cache is not None
        assert corrector.phrase_cache is not None

    def test_simple_pattern_corrections(self):
        """Test basic OCR pattern corrections."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )

        # Test common patterns
        test_cases = [
            ("The terrn is wrong", "The term is wrong"),  # rn → m
            ("cloor is open", "door is open"),  # cl → d
            ("vvindow", "window"),  # vv → w
        ]

        for input_text, expected in test_cases:
            corrected, stats = corrector.correct_chunk(input_text)
            assert corrected == expected
            assert stats["corrections_made"] > 0

    def test_academic_pattern_corrections(self):
        """Test academic-specific OCR corrections."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )

        # Academic patterns
        test_cases = [
            ("See reference [l]", "See reference [1]"),
            ("et aI. (2020)", "et al. (2020)"),
            ("voI. 25", "vol. 25"),
        ]

        for input_text, expected in test_cases:
            corrected, stats = corrector.correct_chunk(input_text)
            assert corrected == expected

    def test_regex_pattern_corrections(self):
        """Test regex-based OCR corrections."""
        corrector = FastOCRCorrector(
            correction_level="aggressive",  # Need aggressive for regex patterns
            enable_spell_check=False,
        )

        # Number-letter confusions
        test_cases = [
            ("Page 10O", "Page 100"),  # O → 0 after number
            ("l00 items", "100 items"),  # l → 1 before numbers
            ("Figure l shows", "Figure 1 shows"),  # l → 1 in context
        ]

        for input_text, expected in test_cases:
            corrected, stats = corrector.correct_chunk(input_text)
            assert corrected == expected

    def test_correction_levels(self):
        """Test different correction levels."""
        text = "The terrn has 10O pages"

        # Light correction - no changes expected
        light_corrector = FastOCRCorrector(
            correction_level="light", enable_spell_check=False
        )
        corrected_light, _ = light_corrector.correct_chunk(text)
        assert corrected_light == text  # No corrections at light level

        # Moderate correction - simple patterns only
        moderate_corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )
        corrected_moderate, _ = moderate_corrector.correct_chunk(text)
        assert "term" in corrected_moderate  # rn → m
        assert "10O" in corrected_moderate  # No regex at moderate

        # Aggressive correction - all patterns
        aggressive_corrector = FastOCRCorrector(
            correction_level="aggressive", enable_spell_check=False
        )
        corrected_aggressive, _ = aggressive_corrector.correct_chunk(text)
        assert "term" in corrected_aggressive  # rn → m
        assert "100" in corrected_aggressive  # 10O → 100

    def test_statistics(self):
        """Test correction statistics."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )

        # Process some text
        text = "The terrn is cloor"
        corrector.correct_chunk(text)

        stats = corrector.get_statistics()

        assert stats["words_processed"] > 0
        assert stats["corrections_made"] > 0
        assert "correction_rate" in stats
        assert "cache_hit_rate" in stats

    def test_cache_functionality(self):
        """Test that caching improves performance."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False, cache_size=100
        )

        # Process same text twice
        text = "The sarne text repeated"
        corrector.correct_chunk(text)
        stats1 = corrector.get_statistics()

        corrector.correct_chunk(text)
        stats2 = corrector.get_statistics()

        # Cache hits should increase
        assert stats2["cache_hits"] > stats1["cache_hits"]

    def test_reset_statistics(self):
        """Test statistics reset."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )

        corrector.correct_chunk("Some text")
        stats_before = corrector.get_statistics()
        assert stats_before["words_processed"] > 0

        corrector.reset_statistics()
        stats_after = corrector.get_statistics()
        assert stats_after["words_processed"] == 0
        assert stats_after["corrections_made"] == 0


class TestStreamProcessing:
    """Test streaming text processing."""

    def test_text_stream_generation(self):
        """Test creating text streams from content."""
        # Create a test text
        text = "A" * 1000 + "B" * 1000 + "C" * 1000

        chunks = []

        def text_generator():
            for i in range(0, len(text), 1000):
                yield text[i : i + 1000]

        for chunk in text_generator():
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == "A" * 1000
        assert chunks[1] == "B" * 1000
        assert chunks[2] == "C" * 1000

    def test_stream_correction(self):
        """Test correcting text streams."""
        corrector = FastOCRCorrector(
            correction_level="moderate", enable_spell_check=False
        )

        # Create stream with OCR errors
        def error_stream():
            yield "First part with terrn error. "
            yield "Second part with cloor error. "
            yield "Third part with vvindow error."

        corrected_parts = []
        for corrected, stats in corrector.correct_text_stream(
            error_stream(), chunk_size=100, overlap_size=10
        ):
            corrected_parts.append(corrected)

        full_corrected = "".join(corrected_parts)

        assert "term" in full_corrected
        assert "door" in full_corrected
        assert "window" in full_corrected


class TestDocumentCorrection:
    """Test document-level correction functions."""

    def test_correct_document_small(self):
        """Test correcting small documents."""
        text = "This docurnent has sorne OCR errors"

        corrected, stats = correct_document(
            text, correction_level="moderate", enable_spell_check=False
        )

        assert "document" in corrected  # rn → m
        assert "some" in corrected  # rn → m
        assert stats["corrections_made"] > 0

    def test_correct_document_large(self):
        """Test correcting large documents."""
        # Create a large document
        text = "This has terrn errors. " * 60000  # ~1.2MB

        corrected, stats = correct_document(
            text, correction_level="moderate", enable_spell_check=False
        )

        assert "term" in corrected
        assert stats["words_processed"] > 0
        assert stats["corrections_made"] > 0


class TestOCRPatterns:
    """Test OCR pattern functionality."""

    def test_pattern_loading(self):
        """Test loading OCR patterns."""
        patterns = get_all_simple_patterns()

        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        assert "rn" in patterns
        assert patterns["rn"] == "m"

    def test_pattern_statistics(self):
        """Test pattern statistics."""
        stats = get_pattern_statistics()

        assert "letter_confusions" in stats
        assert "number_letter_confusions" in stats
        assert "academic_patterns" in stats
        assert "regex_patterns" in stats
        assert "total_patterns" in stats

        assert stats["total_patterns"] > 0


class TestIntegration:
    """Integration tests with the document pipeline."""

    def test_document_with_metadata(self):
        """Test correcting documents while preserving metadata."""
        from src.manilafolder.config import Config
        from src.manilafolder.ingest import apply_ocr_correction

        # Create test documents with OCR errors
        docs = [
            Document(
                page_content="This terrn paper discusses cloor design",
                metadata={"source": "test.pdf", "page": 1},
            ),
            Document(
                page_content="Figure l shows the rnain results",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]

        # Create config with OCR enabled
        config = Config()
        config.enable_ocr_correction = True
        config.ocr_correction_level = "moderate"
        config.ocr_enable_spell_check = False

        # Apply corrections
        corrected_docs, stats = apply_ocr_correction(docs, config)

        # Check corrections
        assert len(corrected_docs) == 2
        assert "term" in corrected_docs[0].page_content
        assert "door" in corrected_docs[0].page_content
        assert "main" in corrected_docs[1].page_content

        # Check metadata preservation
        assert corrected_docs[0].metadata["source"] == "test.pdf"
        assert corrected_docs[0].metadata["page"] == 1
        assert corrected_docs[0].metadata["ocr_corrected"] is True
        assert corrected_docs[0].metadata["ocr_corrections_made"] > 0

        # Check statistics
        assert stats["documents_processed"] == 2
        assert stats["total_corrections"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
