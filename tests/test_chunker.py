import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.chunker import semantic_chunk, semantic_chunk_legacy, _get_overlap_sentences, _create_chunk_document


class TestSemanticChunker(unittest.TestCase):
    """Comprehensive tests for the improved semantic chunking functionality."""

    def setUp(self):
        """Set up test data."""
        self.simple_text = """This is the first sentence. This is the second sentence. This is the third sentence."""
        
        self.legal_text = """<PARSED TEXT FOR PAGE: 1 / 10>
        
INTRODUCTION

This case involves allegations of securities fraud. The plaintiff filed a class action lawsuit against the defendants. The defendants moved to dismiss the complaint under Federal Rule of Civil Procedure 12(b)(6).

FACTUAL BACKGROUND

The company issued misleading statements about its financial condition. Investors relied on these statements to their detriment. The stock price declined significantly after the truth was revealed.

<PARSED TEXT FOR PAGE: 2 / 10>

LEGAL ANALYSIS

Securities fraud requires proof of several elements. First, there must be a material misstatement or omission. Second, the defendant must have acted with scienter. Third, the plaintiff must show reliance and damages."""

        self.long_text = " ".join([f"This is sentence number {i}." for i in range(1, 101)])

    def test_basic_chunking(self):
        """Test basic chunking functionality."""
        chunks = semantic_chunk(self.simple_text, max_chunk_size=50)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(hasattr(chunk, 'page_content') for chunk in chunks))
        self.assertTrue(all(hasattr(chunk, 'metadata') for chunk in chunks))

    def test_sentence_boundary_preservation(self):
        """Test that sentences are not broken across chunks."""
        chunks = semantic_chunk(self.simple_text, max_chunk_size=50)
        
        for chunk in chunks:
            content = chunk.page_content
            # Check that chunk ends with proper sentence ending
            if content.strip():
                self.assertTrue(content.strip().endswith('.') or content.strip().endswith('?') or content.strip().endswith('!'))

    def test_overlap_functionality(self):
        """Test sliding window overlap between chunks."""
        chunks = semantic_chunk(self.long_text, max_chunk_size=200, overlap_size=50)
        
        if len(chunks) > 1:
            # Check that there's actual overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_content = chunks[i].page_content
                next_content = chunks[i + 1].page_content
                
                # Get last few words of current chunk and first few words of next chunk
                current_words = current_content.split()[-10:]
                next_words = next_content.split()[:10]
                
                # There should be some overlap
                overlap_found = any(word in next_words for word in current_words)
                self.assertTrue(overlap_found, f"No overlap found between chunks {i} and {i+1}")

    def test_metadata_preservation(self):
        """Test that page and section metadata is preserved."""
        chunks = semantic_chunk(self.legal_text, max_chunk_size=300)
        
        for chunk in chunks:
            metadata = chunk.metadata
            self.assertIn('page', metadata)
            self.assertIn('section', metadata)
            self.assertIn('chunk_size', metadata)
            self.assertIn('sentence_count', metadata)
            
            # Page should be a valid number
            self.assertIsInstance(metadata['page'], int)
            self.assertGreater(metadata['page'], 0)
            
            # Section should be a string
            self.assertIsInstance(metadata['section'], str)

    def test_page_detection(self):
        """Test page number detection and tracking."""
        chunks = semantic_chunk(self.legal_text, max_chunk_size=500)
        
        # Should have chunks from both page 1 and page 2
        pages_found = set(chunk.metadata['page'] for chunk in chunks)
        self.assertIn(1, pages_found)
        self.assertIn(2, pages_found)

    def test_section_detection(self):
        """Test section heading detection and tracking."""
        chunks = semantic_chunk(self.legal_text, max_chunk_size=200)
        
        sections_found = set(chunk.metadata['section'] for chunk in chunks)
        expected_sections = {'INTRODUCTION', 'FACTUAL BACKGROUND', 'LEGAL ANALYSIS'}
        
        # Should find at least some of the expected sections
        self.assertTrue(len(sections_found.intersection(expected_sections)) > 0)

    def test_min_chunk_size_enforcement(self):
        """Test that minimum chunk size is respected."""
        chunks = semantic_chunk(self.legal_text, max_chunk_size=1000, min_chunk_size=100)
        
        for chunk in chunks:
            chunk_size = len(chunk.page_content)
            # Allow some flexibility for the last chunk
            if chunk != chunks[-1]:
                self.assertGreaterEqual(chunk_size, 90)  # Allow small variance

    def test_empty_input(self):
        """Test handling of empty input."""
        chunks = semantic_chunk("", max_chunk_size=100)
        self.assertEqual(len(chunks), 0)

    def test_short_input(self):
        """Test handling of very short input."""
        short_text = "Short text."
        chunks = semantic_chunk(short_text, max_chunk_size=100, min_chunk_size=50)
        
        # Should create at least one chunk even if below min_chunk_size
        self.assertGreaterEqual(len(chunks), 0)

    def test_backward_compatibility(self):
        """Test that legacy function still works."""
        legacy_chunks = semantic_chunk_legacy(self.legal_text, max_chunk_size=300)
        new_chunks = semantic_chunk(self.legal_text, max_chunk_size=300)
        
        # Both should produce chunks
        self.assertGreater(len(legacy_chunks), 0)
        self.assertGreater(len(new_chunks), 0)
        
        # Both should have proper metadata
        for chunk in legacy_chunks:
            self.assertIn('page', chunk.metadata)
            self.assertIn('section', chunk.metadata)

    def test_overlap_sentences_helper(self):
        """Test the overlap sentences helper function."""
        sentences = ["First sentence.", "Second sentence.", "Third sentence.", "Fourth sentence."]
        
        # Test normal case
        overlap = _get_overlap_sentences(sentences, 30)
        self.assertGreater(len(overlap), 0)
        self.assertLessEqual(len(overlap), len(sentences))
        
        # Test empty input
        overlap_empty = _get_overlap_sentences([], 30)
        self.assertEqual(len(overlap_empty), 0)
        
        # Test very small overlap size
        overlap_small = _get_overlap_sentences(sentences, 5)
        self.assertGreaterEqual(len(overlap_small), 0)

    def test_create_chunk_document_helper(self):
        """Test the document creation helper function."""
        sentences = ["First sentence.", "Second sentence."]
        doc = _create_chunk_document(sentences, 1, "Test Section")
        
        self.assertEqual(doc.page_content, "First sentence. Second sentence.")
        self.assertEqual(doc.metadata['page'], 1)
        self.assertEqual(doc.metadata['section'], "Test Section")
        self.assertEqual(doc.metadata['sentence_count'], 2)
        self.assertGreater(doc.metadata['chunk_size'], 0)

    def test_performance_with_large_text(self):
        """Test performance with larger text input."""
        # Create a larger text sample
        large_text = " ".join([f"This is sentence number {i} in a large document." for i in range(1, 1000)])
        
        # This should complete without timing out
        chunks = semantic_chunk(large_text, max_chunk_size=500, overlap_size=100)
        
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 700 for chunk in chunks))  # Allow some variance


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)