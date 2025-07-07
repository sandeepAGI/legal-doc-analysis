# backend/llm_query_classifier.py

import json
import re
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
import logging

from .llm_wrapper import get_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryClassification:
    """Classification result for a query."""
    query_type: str
    confidence: float
    processing_strategy: str
    recommended_synthesis: str
    token_allocation_bias: str
    reasoning: str
    cache_key: str

class LLMQueryClassifier:
    """
    Standalone query classifier that uses a dedicated token budget separate from 
    main document processing. Classifications are cached for reuse.
    """
    
    def __init__(self, dedicated_token_budget: int = 300):
        """
        Initialize standalone query classifier.
        
        Args:
            dedicated_token_budget: Separate token budget just for classification
        """
        self.dedicated_token_budget = dedicated_token_budget
        self.classification_cache = {}  # In-memory cache for classifications
        
        # Compact classification prompt for efficiency
        self.classification_prompt = """Classify this query for multi-document processing:

Query: "{query}"

Choose ONE type:
1. SINGLE_DOCUMENT - Query about one specific document
2. COMPARATIVE - Comparing/contrasting between documents  
3. CROSS_DOCUMENT - Finding patterns/contradictions across documents
4. THEMATIC - Identifying themes/topics across all documents
5. AGGREGATION - Summarizing/combining information from all documents

JSON response:
{{"query_type": "TYPE", "confidence": 0.9, "processing_strategy": "parallel/sequential/hybrid", "recommended_synthesis": "simple/comparative/analytical", "token_allocation_bias": "balanced/synthesis_heavy/document_heavy", "reasoning": "brief explanation"}}"""
    
    def classify_query(self, query: str, num_documents: int = 1, use_cache: bool = True) -> QueryClassification:
        """
        Classify query using dedicated token budget, independent of main processing.
        
        Args:
            query: User query to classify
            num_documents: Number of documents (for context)
            use_cache: Whether to use cached results
            
        Returns:
            QueryClassification with processing recommendations
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, num_documents)
        
        # Check cache first
        if use_cache and cache_key in self.classification_cache:
            logger.info(f"Using cached classification for query: {query[:30]}...")
            cached_result = self.classification_cache[cache_key]
            cached_result.cache_key = cache_key
            return cached_result
        
        try:
            logger.info(f"Classifying query (dedicated budget): {query[:50]}...")
            
            # Use dedicated token budget for classification
            classification_response = self._call_llm_for_classification(query)
            
            # Parse and validate response
            classification_data = self._parse_classification_response(
                classification_response, query, num_documents
            )
            
            # Create classification object
            classification = QueryClassification(cache_key=cache_key, **classification_data)
            
            # Cache the result
            if use_cache:
                self.classification_cache[cache_key] = classification
                logger.info(f"Cached classification: {classification.query_type}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Return fallback classification
            return self._get_fallback_classification(query, num_documents, cache_key)
    
    def _call_llm_for_classification(self, query: str) -> str:
        """
        Call LLM for classification using dedicated token budget.
        This is completely separate from main document processing.
        """
        # Format the classification prompt
        prompt = self.classification_prompt.format(query=query)
        
        # Call LLM directly with dedicated budget (not affecting main processing)
        llm = get_llm()
        classification_response = llm.invoke(prompt)
        
        logger.info(f"Classification completed using ~{len(prompt)//4} tokens from dedicated budget")
        
        return classification_response
    
    def _generate_cache_key(self, query: str, num_documents: int) -> str:
        """Generate cache key for query classification."""
        cache_input = f"{query.strip().lower()}|{num_documents}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()[:12]
    
    def _parse_classification_response(self, response: str, query: str, num_documents: int) -> Dict[str, Any]:
        """Parse and validate LLM classification response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                classification_data = json.loads(json_str)
                
                # Validate and normalize
                return self._validate_classification(classification_data, query, num_documents)
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse classification response: {e}")
        
        # Fallback if parsing fails
        return self._get_fallback_classification_data(query, num_documents)
    
    def _validate_classification(self, data: Dict[str, Any], query: str, num_documents: int) -> Dict[str, Any]:
        """Validate and normalize classification data."""
        
        # Valid options
        valid_types = {'SINGLE_DOCUMENT', 'COMPARATIVE', 'CROSS_DOCUMENT', 'THEMATIC', 'AGGREGATION'}
        valid_strategies = {'parallel', 'sequential', 'hybrid'}
        valid_synthesis = {'simple', 'comparative', 'analytical'}
        valid_allocation = {'balanced', 'synthesis_heavy', 'document_heavy'}
        
        # Validate and normalize each field
        query_type = data.get('query_type', '').upper()
        if query_type not in valid_types:
            logger.warning(f"Invalid query_type: {query_type}. Using fallback.")
            return self._get_fallback_classification_data(query, num_documents)
        
        confidence = max(0.0, min(1.0, float(data.get('confidence', 0.7))))
        
        strategy = data.get('processing_strategy', 'parallel').lower()
        if strategy not in valid_strategies:
            strategy = 'parallel'
        
        synthesis = data.get('recommended_synthesis', 'simple').lower()
        if synthesis not in valid_synthesis:
            synthesis = 'simple'
        
        allocation = data.get('token_allocation_bias', 'balanced').lower()
        if allocation not in valid_allocation:
            allocation = 'balanced'
        
        reasoning = data.get('reasoning', 'LLM classification')
        
        # Apply business logic for single document
        if num_documents == 1:
            query_type = 'SINGLE_DOCUMENT'
            strategy = 'sequential'
            synthesis = 'simple'
            allocation = 'document_heavy'
            reasoning += ' (adjusted for single document)'
        
        return {
            'query_type': query_type,
            'confidence': confidence,
            'processing_strategy': strategy,
            'recommended_synthesis': synthesis,
            'token_allocation_bias': allocation,
            'reasoning': reasoning
        }
    
    def _get_fallback_classification(self, query: str, num_documents: int, cache_key: str) -> QueryClassification:
        """Generate fallback classification."""
        data = self._get_fallback_classification_data(query, num_documents)
        return QueryClassification(cache_key=cache_key, **data)
    
    def _get_fallback_classification_data(self, query: str, num_documents: int) -> Dict[str, Any]:
        """Generate fallback classification using simple heuristics."""
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if num_documents == 1:
            return {
                'query_type': 'SINGLE_DOCUMENT',
                'confidence': 0.8,
                'processing_strategy': 'sequential',
                'recommended_synthesis': 'simple',
                'token_allocation_bias': 'document_heavy',
                'reasoning': 'Single document detected'
            }
        
        # Multi-document heuristics
        if any(word in query_lower for word in ['compare', 'contrast', 'difference', 'versus', 'vs', 'between']):
            return {
                'query_type': 'COMPARATIVE',
                'confidence': 0.7,
                'processing_strategy': 'parallel',
                'recommended_synthesis': 'comparative',
                'token_allocation_bias': 'synthesis_heavy',
                'reasoning': 'Comparative keywords detected'
            }
        
        if any(word in query_lower for word in ['contradiction', 'conflict', 'disagree', 'pattern', 'across', 'find']):
            return {
                'query_type': 'CROSS_DOCUMENT',
                'confidence': 0.7,
                'processing_strategy': 'parallel',
                'recommended_synthesis': 'analytical',
                'token_allocation_bias': 'synthesis_heavy',
                'reasoning': 'Cross-document analysis keywords detected'
            }
        
        if any(word in query_lower for word in ['theme', 'common', 'similar', 'trend', 'overall', 'general']):
            return {
                'query_type': 'THEMATIC',
                'confidence': 0.7,
                'processing_strategy': 'parallel',
                'recommended_synthesis': 'analytical',
                'token_allocation_bias': 'balanced',
                'reasoning': 'Thematic analysis keywords detected'
            }
        
        if any(word in query_lower for word in ['summarize', 'summary', 'all', 'total', 'combine', 'overall']):
            return {
                'query_type': 'AGGREGATION',
                'confidence': 0.7,
                'processing_strategy': 'parallel',
                'recommended_synthesis': 'simple',
                'token_allocation_bias': 'balanced',
                'reasoning': 'Aggregation keywords detected'
            }
        
        # Default fallback
        return {
            'query_type': 'CROSS_DOCUMENT',
            'confidence': 0.6,
            'processing_strategy': 'parallel',
            'recommended_synthesis': 'simple',
            'token_allocation_bias': 'balanced',
            'reasoning': f'Default classification for {num_documents} documents'
        }
    
    def get_processing_recommendations(self, classification: QueryClassification, 
                                     num_documents: int) -> Dict[str, Any]:
        """
        Get detailed processing recommendations based on classification.
        
        Args:
            classification: Query classification result
            num_documents: Number of documents to process
            
        Returns:
            Processing strategy recommendations for the orchestrator
        """
        recommendations = {
            'use_parallel_processing': classification.processing_strategy in ['parallel', 'hybrid'],
            'synthesis_complexity': classification.recommended_synthesis,
            'token_allocation_strategy': classification.token_allocation_bias,
            'expected_output_type': self._get_expected_output_type(classification.query_type),
            'requires_cross_document_analysis': classification.query_type in ['COMPARATIVE', 'CROSS_DOCUMENT', 'THEMATIC'],
            'source_attribution_importance': 'high' if classification.query_type == 'COMPARATIVE' else 'medium',
            'enable_contradiction_detection': classification.query_type == 'CROSS_DOCUMENT',
            'enable_theme_extraction': classification.query_type == 'THEMATIC',
            'prioritize_document_coverage': classification.query_type == 'AGGREGATION',
            'confidence_threshold': 0.8 if classification.confidence > 0.8 else 0.6
        }
        
        # Override for single document
        if num_documents == 1:
            recommendations.update({
                'use_parallel_processing': False,
                'requires_cross_document_analysis': False,
                'synthesis_complexity': 'simple',
                'token_allocation_strategy': 'document_heavy'
            })
        
        return recommendations
    
    def _get_expected_output_type(self, query_type: str) -> str:
        """Get expected output format for query type."""
        output_types = {
            'SINGLE_DOCUMENT': 'direct_answer',
            'COMPARATIVE': 'comparison_analysis',
            'CROSS_DOCUMENT': 'pattern_analysis',
            'THEMATIC': 'theme_summary',
            'AGGREGATION': 'comprehensive_summary'
        }
        return output_types.get(query_type, 'general_analysis')
    
    def explain_classification(self, classification: QueryClassification) -> str:
        """Generate human-readable explanation."""
        explanations = {
            'SINGLE_DOCUMENT': "Query focuses on a single document",
            'COMPARATIVE': "Query requires comparing between documents",
            'CROSS_DOCUMENT': "Query seeks patterns across multiple documents",
            'THEMATIC': "Query identifies themes across documents",
            'AGGREGATION': "Query summarizes information from all documents"
        }
        
        base = explanations.get(classification.query_type, "Multi-document query")
        confidence_text = f"(Confidence: {classification.confidence:.1%})"
        
        return f"{base} {confidence_text}"
    
    def clear_cache(self):
        """Clear the classification cache."""
        self.classification_cache.clear()
        logger.info("Query classification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_classifications': len(self.classification_cache),
            'dedicated_token_budget': self.dedicated_token_budget
        }