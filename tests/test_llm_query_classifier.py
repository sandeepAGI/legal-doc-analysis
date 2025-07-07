# tests/test_llm_query_classifier.py

import pytest
from unittest.mock import Mock, patch
import json

from backend.llm_query_classifier import LLMQueryClassifier, QueryClassification

class TestQueryClassification:
    """Test suite for QueryClassification dataclass."""
    
    def test_query_classification_creation(self):
        """Test QueryClassification creation."""
        classification = QueryClassification(
            query_type="COMPARATIVE",
            confidence=0.9,
            processing_strategy="parallel",
            recommended_synthesis="comparative",
            token_allocation_bias="balanced",
            reasoning="Test reasoning",
            cache_key="test123"
        )
        
        assert classification.query_type == "COMPARATIVE"
        assert classification.confidence == 0.9
        assert classification.processing_strategy == "parallel"
        assert classification.recommended_synthesis == "comparative"
        assert classification.token_allocation_bias == "balanced"
        assert classification.reasoning == "Test reasoning"
        assert classification.cache_key == "test123"

class TestLLMQueryClassifier:
    """Test suite for LLMQueryClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create LLMQueryClassifier instance for testing."""
        return LLMQueryClassifier(dedicated_token_budget=300)
    
    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.dedicated_token_budget == 300
        assert classifier.classification_cache == {}
        assert "Query:" in classifier.classification_prompt
        assert "JSON response:" in classifier.classification_prompt
    
    def test_generate_cache_key(self, classifier):
        """Test cache key generation."""
        key1 = classifier._generate_cache_key("What is this about?", 2)
        key2 = classifier._generate_cache_key("What is this about?", 2)
        key3 = classifier._generate_cache_key("What is this about?", 3)
        
        assert key1 == key2  # Same query and doc count
        assert key1 != key3  # Different doc count
        assert len(key1) == 12  # MD5 hash truncated to 12 chars
    
    @patch('backend.llm_query_classifier.synthesize_answer_cached')
    def test_classify_query_success(self, mock_synthesize, classifier):
        """Test successful query classification."""
        # Mock LLM response with valid JSON
        mock_response = '''
        Here is the classification:
        {
            "query_type": "COMPARATIVE",
            "confidence": 0.9,
            "processing_strategy": "parallel",
            "recommended_synthesis": "comparative",
            "token_allocation_bias": "balanced",
            "reasoning": "Query contains comparison keywords"
        }
        That's the analysis.
        '''
        mock_synthesize.return_value = mock_response
        
        classification = classifier.classify_query("Compare document A and B", 2)
        
        assert isinstance(classification, QueryClassification)
        assert classification.query_type == "COMPARATIVE"
        assert classification.confidence == 0.9
        assert classification.processing_strategy == "parallel"
        assert mock_synthesize.called
    
    @patch('backend.llm_query_classifier.synthesize_answer_cached')
    def test_classify_query_with_cache(self, mock_synthesize, classifier):
        """Test query classification with caching."""
        # First call
        mock_response = '{"query_type": "SINGLE_DOCUMENT", "confidence": 0.8, "processing_strategy": "sequential", "recommended_synthesis": "simple", "token_allocation_bias": "balanced", "reasoning": "test"}'
        mock_synthesize.return_value = mock_response
        
        classification1 = classifier.classify_query("What is this about?", 1)
        
        # Second call should use cache
        classification2 = classifier.classify_query("What is this about?", 1)
        
        assert classification1.query_type == classification2.query_type
        assert mock_synthesize.call_count == 1  # Should only be called once
    
    @patch('backend.llm_query_classifier.synthesize_answer_cached')
    def test_classify_query_llm_failure(self, mock_synthesize, classifier):
        """Test query classification when LLM fails."""
        mock_synthesize.side_effect = Exception("LLM failed")
        
        classification = classifier.classify_query("Compare documents", 2)
        
        assert isinstance(classification, QueryClassification)
        assert classification.confidence < 1.0  # Should be fallback classification
        assert "fallback" in classification.reasoning.lower()
    
    def test_parse_classification_response_valid_json(self, classifier):
        """Test parsing valid JSON response."""
        response = '''
        Some text before
        {
            "query_type": "THEMATIC",
            "confidence": 0.85,
            "processing_strategy": "parallel",
            "recommended_synthesis": "analytical",
            "token_allocation_bias": "balanced",
            "reasoning": "Thematic analysis detected"
        }
        Some text after
        '''
        
        data = classifier._parse_classification_response(response, "test query", 3)
        
        assert data['query_type'] == 'THEMATIC'
        assert data['confidence'] == 0.85
        assert data['processing_strategy'] == 'parallel'
    
    def test_parse_classification_response_invalid_json(self, classifier):
        """Test parsing invalid JSON response."""
        response = "This is not JSON at all"
        
        data = classifier._parse_classification_response(response, "test query", 2)
        
        # Should fall back to heuristic classification
        assert 'query_type' in data
        assert 'confidence' in data
        assert data['confidence'] < 1.0  # Fallback should have lower confidence
    
    def test_parse_classification_response_malformed_json(self, classifier):
        """Test parsing malformed JSON response."""
        response = '{"query_type": "INVALID_TYPE", "confidence": "not_a_number"}'
        
        data = classifier._parse_classification_response(response, "test query", 2)
        
        # Should fall back to heuristic classification
        assert 'query_type' in data
        assert isinstance(data['confidence'], float)
    
    def test_validate_classification_valid_data(self, classifier):
        """Test validation of valid classification data."""
        data = {
            "query_type": "COMPARATIVE",
            "confidence": 0.9,
            "processing_strategy": "parallel",
            "recommended_synthesis": "comparative",
            "token_allocation_bias": "balanced",
            "reasoning": "Valid classification"
        }
        
        validated = classifier._validate_classification(data, "test query", 2)
        
        assert validated['query_type'] == 'COMPARATIVE'
        assert validated['confidence'] == 0.9
        assert validated['processing_strategy'] == 'parallel'
    
    def test_validate_classification_invalid_type(self, classifier):
        """Test validation with invalid query type."""
        data = {
            "query_type": "INVALID_TYPE",
            "confidence": 0.9,
            "processing_strategy": "parallel",
            "recommended_synthesis": "comparative",
            "token_allocation_bias": "balanced",
            "reasoning": "Invalid type"
        }
        
        validated = classifier._validate_classification(data, "test query", 2)
        
        # Should fall back to heuristic classification
        assert validated['query_type'] in ['SINGLE_DOCUMENT', 'COMPARATIVE', 'CROSS_DOCUMENT', 'THEMATIC', 'AGGREGATION']
    
    def test_validate_classification_single_document_override(self, classifier):
        """Test validation overrides for single document."""
        data = {
            "query_type": "COMPARATIVE",
            "confidence": 0.9,
            "processing_strategy": "parallel",
            "recommended_synthesis": "comparative",
            "token_allocation_bias": "balanced",
            "reasoning": "Should be overridden"
        }
        
        validated = classifier._validate_classification(data, "test query", 1)  # Single document
        
        assert validated['query_type'] == 'SINGLE_DOCUMENT'
        assert validated['processing_strategy'] == 'sequential'
        assert validated['recommended_synthesis'] == 'simple'
    
    def test_get_fallback_classification_data_single_document(self, classifier):
        """Test fallback classification for single document."""
        data = classifier._get_fallback_classification_data("What is this about?", 1)
        
        assert data['query_type'] == 'SINGLE_DOCUMENT'
        assert data['processing_strategy'] == 'sequential'
        assert data['recommended_synthesis'] == 'simple'
    
    def test_get_fallback_classification_data_comparative(self, classifier):
        """Test fallback classification for comparative query."""
        data = classifier._get_fallback_classification_data("Compare document A vs document B", 2)
        
        assert data['query_type'] == 'COMPARATIVE'
        assert data['processing_strategy'] == 'parallel'
        assert data['recommended_synthesis'] == 'comparative'
    
    def test_get_fallback_classification_data_cross_document(self, classifier):
        """Test fallback classification for cross-document query."""
        data = classifier._get_fallback_classification_data("Find contradictions across documents", 3)
        
        assert data['query_type'] == 'CROSS_DOCUMENT'
        assert data['processing_strategy'] == 'parallel'
        assert data['recommended_synthesis'] == 'analytical'
    
    def test_get_fallback_classification_data_thematic(self, classifier):
        """Test fallback classification for thematic query."""
        data = classifier._get_fallback_classification_data("What are the common themes?", 3)
        
        assert data['query_type'] == 'THEMATIC'
        assert data['processing_strategy'] == 'parallel'
        assert data['recommended_synthesis'] == 'analytical'
    
    def test_get_fallback_classification_data_aggregation(self, classifier):
        """Test fallback classification for aggregation query."""
        data = classifier._get_fallback_classification_data("Summarize all documents", 3)
        
        assert data['query_type'] == 'AGGREGATION'
        assert data['processing_strategy'] == 'parallel'
        assert data['recommended_synthesis'] == 'simple'
    
    def test_get_fallback_classification_data_default(self, classifier):
        """Test fallback classification for unrecognized query."""
        data = classifier._get_fallback_classification_data("Random query text", 3)
        
        assert data['query_type'] == 'CROSS_DOCUMENT'
        assert data['processing_strategy'] == 'parallel'
    
    def test_get_processing_recommendations_single_document(self, classifier):
        """Test processing recommendations for single document."""
        classification = QueryClassification(
            query_type="SINGLE_DOCUMENT",
            confidence=0.9,
            processing_strategy="sequential",
            recommended_synthesis="simple",
            token_allocation_bias="document_heavy",
            reasoning="Single doc",
            cache_key="test"
        )
        
        recommendations = classifier.get_processing_recommendations(classification, 1)
        
        assert recommendations['use_parallel_processing'] is False
        assert recommendations['requires_cross_document_analysis'] is False
        assert recommendations['synthesis_complexity'] == 'simple'
    
    def test_get_processing_recommendations_comparative(self, classifier):
        """Test processing recommendations for comparative query."""
        classification = QueryClassification(
            query_type="COMPARATIVE",
            confidence=0.9,
            processing_strategy="parallel",
            recommended_synthesis="comparative",
            token_allocation_bias="synthesis_heavy",
            reasoning="Comparative",
            cache_key="test"
        )
        
        recommendations = classifier.get_processing_recommendations(classification, 3)
        
        assert recommendations['use_parallel_processing'] is True
        assert recommendations['requires_cross_document_analysis'] is True
        assert recommendations['source_attribution_importance'] == 'high'
        assert recommendations['synthesis_complexity'] == 'comparative'
    
    def test_get_processing_recommendations_cross_document(self, classifier):
        """Test processing recommendations for cross-document query."""
        classification = QueryClassification(
            query_type="CROSS_DOCUMENT",
            confidence=0.85,
            processing_strategy="parallel",
            recommended_synthesis="analytical",
            token_allocation_bias="synthesis_heavy",
            reasoning="Cross-doc",
            cache_key="test"
        )
        
        recommendations = classifier.get_processing_recommendations(classification, 3)
        
        assert recommendations['enable_contradiction_detection'] is True
        assert recommendations['requires_cross_document_analysis'] is True
    
    def test_get_processing_recommendations_thematic(self, classifier):
        """Test processing recommendations for thematic query."""
        classification = QueryClassification(
            query_type="THEMATIC",
            confidence=0.8,
            processing_strategy="parallel",
            recommended_synthesis="analytical",
            token_allocation_bias="balanced",
            reasoning="Thematic",
            cache_key="test"
        )
        
        recommendations = classifier.get_processing_recommendations(classification, 4)
        
        assert recommendations['enable_theme_extraction'] is True
        assert recommendations['requires_cross_document_analysis'] is True
    
    def test_get_processing_recommendations_aggregation(self, classifier):
        """Test processing recommendations for aggregation query."""
        classification = QueryClassification(
            query_type="AGGREGATION",
            confidence=0.75,
            processing_strategy="parallel",
            recommended_synthesis="simple",
            token_allocation_bias="balanced",
            reasoning="Aggregation",
            cache_key="test"
        )
        
        recommendations = classifier.get_processing_recommendations(classification, 2)
        
        assert recommendations['prioritize_document_coverage'] is True
    
    def test_get_expected_output_type(self, classifier):
        """Test expected output type mapping."""
        assert classifier._get_expected_output_type("SINGLE_DOCUMENT") == "direct_answer"
        assert classifier._get_expected_output_type("COMPARATIVE") == "comparison_analysis"
        assert classifier._get_expected_output_type("CROSS_DOCUMENT") == "pattern_analysis"
        assert classifier._get_expected_output_type("THEMATIC") == "theme_summary"
        assert classifier._get_expected_output_type("AGGREGATION") == "comprehensive_summary"
        assert classifier._get_expected_output_type("UNKNOWN") == "general_analysis"
    
    def test_explain_classification(self, classifier):
        """Test classification explanation generation."""
        classification = QueryClassification(
            query_type="COMPARATIVE",
            confidence=0.9,
            processing_strategy="parallel",
            recommended_synthesis="comparative",
            token_allocation_bias="balanced",
            reasoning="Comparative query",
            cache_key="test"
        )
        
        explanation = classifier.explain_classification(classification)
        
        assert "comparing" in explanation.lower()
        assert "90.0%" in explanation  # Confidence percentage
    
    def test_clear_cache(self, classifier):
        """Test cache clearing."""
        # Add something to cache
        classifier.classification_cache["test_key"] = Mock()
        assert len(classifier.classification_cache) == 1
        
        classifier.clear_cache()
        
        assert len(classifier.classification_cache) == 0
    
    def test_get_cache_stats(self, classifier):
        """Test cache statistics."""
        # Add some items to cache
        classifier.classification_cache["key1"] = Mock()
        classifier.classification_cache["key2"] = Mock()
        
        stats = classifier.get_cache_stats()
        
        assert stats['cached_classifications'] == 2
        assert stats['dedicated_token_budget'] == 300

class TestClassifierIntegration:
    """Integration tests for the complete classifier system."""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow."""
        classifier = LLMQueryClassifier()
        
        # Should work without errors
        classification = classifier.classify_query("Compare these documents", 2, use_cache=False)
        
        assert isinstance(classification, QueryClassification)
        assert classification.query_type in ['SINGLE_DOCUMENT', 'COMPARATIVE', 'CROSS_DOCUMENT', 'THEMATIC', 'AGGREGATION']
        assert 0.0 <= classification.confidence <= 1.0
    
    def test_cache_persistence_across_calls(self):
        """Test that cache persists across multiple calls."""
        classifier = LLMQueryClassifier()
        
        # Make same call twice
        query = "What is the main argument?"
        classification1 = classifier.classify_query(query, 1)
        classification2 = classifier.classify_query(query, 1)
        
        # Should get same result from cache
        assert classification1.query_type == classification2.query_type
        assert classification1.cache_key == classification2.cache_key
    
    def test_different_queries_different_classifications(self):
        """Test that different queries get different classifications."""
        classifier = LLMQueryClassifier()
        
        single_doc = classifier.classify_query("What is the main point?", 1)
        comparative = classifier.classify_query("Compare document A and B", 2)
        
        # Should get different classifications
        assert single_doc.query_type != comparative.query_type or single_doc.processing_strategy != comparative.processing_strategy

if __name__ == "__main__":
    pytest.main([__file__, "-v"])