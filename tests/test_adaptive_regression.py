# tests/test_adaptive_regression.py

import unittest
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.smart_vectorstore import SmartVectorStore

class TestAdaptiveRegression(unittest.TestCase):
    """
    Regression test to ensure adaptive retrieval works with real documents
    and doesn't break existing functionality.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Find a test PDF
        data_dir = "data"
        cls.test_pdf = None
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.pdf'):
                    cls.test_pdf = os.path.join(data_dir, filename)
                    break
        
        if not cls.test_pdf:
            cls.skipTest("No test PDF found in data directory")
    
    def setUp(self):
        """Set up for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.smart_vs = SmartVectorStore(base_dir=self.temp_dir)
    
    def test_adaptive_retrieval_integration(self):
        """Test adaptive retrieval with real document and embeddings."""
        # Load and chunk document
        print(f"Loading document: {os.path.basename(self.test_pdf)}")
        full_text = load_document(self.test_pdf)
        chunks = semantic_chunk(full_text, max_chunk_size=1000)
        
        self.assertGreater(len(chunks), 0, "Should have chunks")
        
        # Initialize embedder (use BGE small for speed)
        model_path = "models/bge-small-en"
        if os.path.exists(model_path):
            embedder = get_embedder(local_model_path=model_path)
        else:
            self.skipTest("BGE small model not found")
        
        # Create vector store
        vectordb = self.smart_vs.get_or_create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            document_content=full_text,
            embedding_model="bge-small-en",
            chunk_params={"max_chunk_size": 1000, "overlap_size": 200}
        )
        
        # Test different query types
        test_queries = [
            ("What is the plaintiff's name?", "simple"),
            ("Describe the main legal arguments", "medium"),
            ("Compare the plaintiff's claims with the defendant's response", "complex")
        ]
        
        for query, expected_complexity in test_queries:
            print(f"\nTesting query: '{query}' (expected: {expected_complexity})")
            
            # Test adaptive retrieval
            results, metadata = self.smart_vs.query_vectorstore(
                vectordb, query, embedding_model="bge-small-en", adaptive=True
            )
            
            # Verify results
            self.assertIsNotNone(results, "Should return results")
            self.assertGreater(len(results), 0, "Should have at least 1 result")
            self.assertIsNotNone(metadata, "Should return metadata")
            
            # Verify metadata
            self.assertEqual(metadata['query_complexity'], expected_complexity)
            self.assertIn('adaptive_k', metadata)
            self.assertIn('final_results_count', metadata)
            self.assertIn('estimated_tokens', metadata)
            
            # Verify results structure
            for doc, score in results:
                self.assertIsInstance(score, (int, float))
                self.assertTrue(hasattr(doc, 'page_content'))
                self.assertTrue(hasattr(doc, 'metadata'))
            
            print(f"✅ Complexity: {metadata['query_complexity']}, "
                  f"K: {metadata['final_results_count']}, "
                  f"Tokens: {metadata['estimated_tokens']}")
    
    def test_backward_compatibility(self):
        """Test that existing code still works without adaptive retrieval."""
        # Load and chunk document
        full_text = load_document(self.test_pdf)
        chunks = semantic_chunk(full_text, max_chunk_size=500)
        
        # Initialize embedder
        model_path = "models/bge-small-en"
        if os.path.exists(model_path):
            embedder = get_embedder(local_model_path=model_path)
        else:
            self.skipTest("BGE small model not found")
        
        # Create vector store
        vectordb = self.smart_vs.get_or_create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            document_content=full_text,
            embedding_model="bge-small-en",
            chunk_params={"max_chunk_size": 500}
        )
        
        # Test legacy query (adaptive=False)
        results, metadata = self.smart_vs.query_vectorstore(
            vectordb, "What is this case about?", k=5, adaptive=False
        )
        
        # Should return exactly 5 results, no metadata
        self.assertEqual(len(results), 5)
        self.assertIsNone(metadata)
        
        print("✅ Backward compatibility confirmed")
    
    def test_context_window_constraints(self):
        """Test that adaptive retrieval respects context window limits."""
        # Load document
        full_text = load_document(self.test_pdf)
        chunks = semantic_chunk(full_text, max_chunk_size=1000)
        
        # Initialize embedder
        model_path = "models/bge-small-en"
        if os.path.exists(model_path):
            embedder = get_embedder(local_model_path=model_path)
        else:
            self.skipTest("BGE small model not found")
        
        # Create vector store
        vectordb = self.smart_vs.get_or_create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            document_content=full_text,
            embedding_model="bge-small-en",
            chunk_params={"max_chunk_size": 1000}
        )
        
        # Test complex query (should use more chunks)
        complex_query = ("Analyze the legal implications and compare with precedents "
                        "while evaluating the strength of both arguments")
        
        results, metadata = self.smart_vs.query_vectorstore(
            vectordb, complex_query, embedding_model="bge-small-en", adaptive=True
        )
        
        # Should respect context window (max ~7000 tokens, so max ~28 chunks of 1000 chars)
        estimated_tokens = metadata['estimated_tokens']
        self.assertLessEqual(estimated_tokens, 7000, 
                           f"Should respect context window limit: {estimated_tokens} tokens")
        
        print(f"✅ Context window respected: {estimated_tokens} tokens used")

if __name__ == '__main__':
    unittest.main(verbosity=2)