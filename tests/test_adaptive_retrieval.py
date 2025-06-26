# tests/test_adaptive_retrieval.py

import unittest
import sys
import os
from unittest.mock import Mock, patch
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.adaptive_retrieval import AdaptiveRetriever

class TestAdaptiveRetrieval(unittest.TestCase):
    """Comprehensive unit tests for Adaptive Retrieval Parameters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retriever_bge_small = AdaptiveRetriever('bge-small-en')
        self.retriever_bge_base = AdaptiveRetriever('bge-base-en') 
        self.retriever_nomic = AdaptiveRetriever('nomic-embed-text')
        
        # Mock documents with scores
        self.mock_docs_bge = [
            (Document(page_content="Exact match content", metadata={"page": 1}), 0.25),
            (Document(page_content="Very relevant content", metadata={"page": 2}), 0.35),
            (Document(page_content="Somewhat relevant content", metadata={"page": 3}), 0.45),
            (Document(page_content="Marginally relevant content", metadata={"page": 4}), 0.55),
            (Document(page_content="Barely relevant content", metadata={"page": 5}), 0.65)
        ]
        
        self.mock_docs_nomic = [
            (Document(page_content="Exact match content", metadata={"page": 1}), 310.0),
            (Document(page_content="Very relevant content", metadata={"page": 2}), 350.0),
            (Document(page_content="Somewhat relevant content", metadata={"page": 3}), 420.0),
            (Document(page_content="Marginally relevant content", metadata={"page": 4}), 480.0),
            (Document(page_content="Barely relevant content", metadata={"page": 5}), 510.0)
        ]
    
    def test_query_complexity_analysis(self):
        """Test query complexity classification."""
        # Simple queries
        simple_queries = [
            "What is the definition of force majeure?",
            "Who is the plaintiff?",
            "When was the contract signed?",
            "Where did the incident occur?",
            "Is this a breach of contract?"
        ]
        
        for query in simple_queries:
            complexity = self.retriever_bge_small.analyze_query_complexity(query)
            self.assertEqual(complexity, 'simple', f"Query '{query}' should be simple")
        
        # Complex queries
        complex_queries = [
            "Compare the plaintiff's arguments with the defendant's counterarguments",
            "Analyze the implications of this ruling on future cases",
            "How does this case differ from previous precedents?",
            "Explain why the court reached this decision and what factors influenced it",
            "What are the legal and financial consequences? How might this affect stakeholders?"
        ]
        
        for query in complex_queries:
            complexity = self.retriever_bge_small.analyze_query_complexity(query)
            self.assertEqual(complexity, 'complex', f"Query '{query}' should be complex")
        
        # Medium complexity queries
        medium_queries = [
            "Describe the main legal arguments presented",
            "Summarize the evidence and testimony provided",
            "Outline the court's reasoning in the decision"
        ]
        
        for query in medium_queries:
            complexity = self.retriever_bge_small.analyze_query_complexity(query)
            self.assertEqual(complexity, 'medium', f"Query '{query}' should be medium")
    
    def test_adaptive_k_calculation(self):
        """Test adaptive k value calculation."""
        # Simple queries should use smaller k
        k_simple = self.retriever_bge_small.calculate_adaptive_k('simple')
        self.assertEqual(k_simple, 6)
        
        # Medium queries should use default k
        k_medium = self.retriever_bge_small.calculate_adaptive_k('medium')
        self.assertEqual(k_medium, 10)
        
        # Complex queries should use larger k
        k_complex = self.retriever_bge_small.calculate_adaptive_k('complex')
        self.assertEqual(k_complex, 15)
        
        # Test context window constraints
        k_constrained = self.retriever_bge_small.calculate_adaptive_k('complex', max_context_tokens=1000)
        self.assertEqual(k_constrained, 4)  # 1000 / 250 tokens per chunk
    
    def test_model_specific_thresholds(self):
        """Test that different models use appropriate thresholds."""
        # BGE Small thresholds
        threshold_bge_small = self.retriever_bge_small.get_quality_threshold('medium', 'good')
        self.assertAlmostEqual(threshold_bge_small, 0.45, places=2)
        
        # BGE Base thresholds  
        threshold_bge_base = self.retriever_bge_base.get_quality_threshold('medium', 'good')
        self.assertAlmostEqual(threshold_bge_base, 0.50, places=2)
        
        # Nomic thresholds (much higher values)
        threshold_nomic = self.retriever_nomic.get_quality_threshold('medium', 'good')
        self.assertAlmostEqual(threshold_nomic, 460.0, places=1)
    
    def test_quality_filtering(self):
        """Test similarity score filtering."""
        # Test BGE model filtering
        threshold = 0.40
        filtered = self.retriever_bge_small.filter_by_quality(self.mock_docs_bge, threshold)
        
        # Should keep scores <= 0.40 (first 2 documents)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0][1], 0.25)
        self.assertEqual(filtered[1][1], 0.35)
        
        # Test Nomic model filtering  
        threshold = 400.0
        filtered = self.retriever_nomic.filter_by_quality(self.mock_docs_nomic, threshold)
        
        # Should keep scores <= 400.0 (first 2 documents)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0][1], 310.0)
        self.assertEqual(filtered[1][1], 350.0)
    
    def test_complexity_threshold_adjustment(self):
        """Test that thresholds adjust based on query complexity."""
        # Simple queries should have stricter thresholds (lower values for BGE)
        simple_threshold = self.retriever_bge_small.get_quality_threshold('simple', 'good')
        medium_threshold = self.retriever_bge_small.get_quality_threshold('medium', 'good')
        complex_threshold = self.retriever_bge_small.get_quality_threshold('complex', 'good')
        
        # For BGE models (lower = better), simple should be lowest
        self.assertLess(simple_threshold, medium_threshold)
        self.assertLess(medium_threshold, complex_threshold)
    
    @patch('backend.adaptive_retrieval.AdaptiveRetriever')
    def test_adaptive_retrieve_integration(self, mock_retriever_class):
        """Test full adaptive retrieval workflow."""
        # Mock the vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = self.mock_docs_bge
        
        # Create real retriever instance
        retriever = AdaptiveRetriever('bge-small-en')
        
        # Test with simple query
        results, metadata = retriever.adaptive_retrieve(mock_vectorstore, "What is the plaintiff's name?")
        
        # Verify metadata
        self.assertEqual(metadata['query_complexity'], 'simple')
        self.assertEqual(metadata['adaptive_k'], 6)
        self.assertEqual(metadata['embedding_model'], 'bge-small-en')
        self.assertGreater(metadata['estimated_tokens'], 0)
        
        # Verify results are filtered and limited
        self.assertLessEqual(len(results), 6)
        self.assertIsInstance(results[0][0], Document)
        self.assertIsInstance(results[0][1], float)
    
    def test_fallback_strategy(self):
        """Test fallback strategy when high-quality results are insufficient."""
        # Create scenario with mostly poor-quality results
        poor_quality_docs = [
            (Document(page_content="Poor match 1", metadata={"page": 1}), 0.70),
            (Document(page_content="Poor match 2", metadata={"page": 2}), 0.75),
            (Document(page_content="Poor match 3", metadata={"page": 3}), 0.80)
        ]
        
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = poor_quality_docs
        
        retriever = AdaptiveRetriever('bge-small-en')
        results, metadata = retriever.adaptive_retrieve(mock_vectorstore, "What is X?")
        
        # Should still return results due to fallback
        self.assertGreater(len(results), 0)
        self.assertGreater(metadata['final_results_count'], 0)
    
    def test_retrieval_explanation(self):
        """Test retrieval explanation generation."""
        metadata = {
            'query_complexity': 'complex',
            'final_results_count': 12,
            'estimated_tokens': 3000,
            'embedding_model': 'bge-base-en',
            'raw_results_count': 20,
            'filtered_results_count': 15
        }
        
        explanation = self.retriever_bge_base.get_retrieval_explanation(metadata)
        
        # Verify explanation contains key information
        self.assertIn('12 chunks', explanation)
        self.assertIn('complex query', explanation)
        self.assertIn('3000 tokens', explanation)
        self.assertIn('bge-base-en', explanation)
        self.assertIn('5 low-quality', explanation)
    
    def test_context_window_respect(self):
        """Test that retrieval respects context window limits."""
        # Test with very restrictive context limit
        retriever = AdaptiveRetriever('bge-small-en')
        max_k = retriever.calculate_adaptive_k('complex', max_context_tokens=500)
        
        # Should be limited by context window (500 / 250 = 2)
        self.assertEqual(max_k, 2)
        
        # Test with generous context limit
        max_k = retriever.calculate_adaptive_k('simple', max_context_tokens=10000)
        
        # Should use complexity-based k, not context-limited
        self.assertEqual(max_k, 6)
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works."""
        from backend.vectorstore import query_vectorstore
        from backend.smart_vectorstore import SmartVectorStore
        
        # Test legacy vectorstore still works
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = self.mock_docs_bge[:3]
        
        # Legacy call (adaptive=False)
        results = query_vectorstore(mock_vectorstore, "test query", k=3, adaptive=False)
        self.assertEqual(len(results), 3)
        
        # Adaptive call
        results, metadata = query_vectorstore(mock_vectorstore, "test query", adaptive=True)
        self.assertIsNotNone(metadata)
        self.assertIn('query_complexity', metadata)

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)