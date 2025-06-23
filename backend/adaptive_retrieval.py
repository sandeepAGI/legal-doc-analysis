# backend/adaptive_retrieval.py

import re
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

class AdaptiveRetriever:
    """
    Adaptive retrieval system that adjusts k and similarity thresholds based on:
    1. Query complexity analysis
    2. Embedding model characteristics
    3. Context window constraints
    """
    
    # Model-specific similarity score thresholds
    SCORE_THRESHOLDS = {
        'bge-small-en': {
            'excellent': 0.30,
            'good': 0.45,
            'fair': 0.55,
            'poor': 0.60
        },
        'bge-base-en': {
            'excellent': 0.35,
            'good': 0.50,
            'fair': 0.58,
            'poor': 0.65
        },
        'nomic-embed-text': {
            'excellent': 320,
            'good': 460,
            'fair': 490,
            'poor': 520
        }
    }
    
    # Context window constraints (tokens)
    MAX_CONTEXT_TOKENS = 7000  # Conservative limit for 8K context
    TOKENS_PER_CHUNK = 250     # Estimated tokens per 1000-char chunk
    
    def __init__(self, embedding_model: str = 'bge-small-en'):
        """
        Initialize adaptive retriever.
        
        Args:
            embedding_model: Name of embedding model being used
        """
        self.embedding_model = embedding_model
        self.thresholds = self.SCORE_THRESHOLDS.get(
            embedding_model, 
            self.SCORE_THRESHOLDS['bge-small-en']  # Default fallback
        )
    
    def analyze_query_complexity(self, query: str) -> str:
        """
        Analyze query complexity to determine retrieval strategy.
        
        Args:
            query: User query string
            
        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        query_lower = query.lower().strip()
        
        # Simple query indicators
        simple_patterns = [
            r'\bwho\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b',
            r'\bdefine\b', r'\bdefinition\b', r'\bmean\b', r'\bmeans\b',
            r'\bis\b.*\?', r'\bare\b.*\?', r'\bdoes\b.*\?', r'\bdid\b.*\?'
        ]
        
        # Complex query indicators
        complex_patterns = [
            r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b', r'\bsimilar\b',
            r'\banalyze\b', r'\banalysis\b', r'\bevaluate\b', r'\bassess\b',
            r'\bexplain why\b', r'\bhow does\b', r'\bwhat are the implications\b',
            r'\bdiscuss\b', r'\bexamine\b', r'\bconsider\b',
            r'\band\b.*\band\b', r'\bor\b.*\bor\b',  # Multiple conditions
            r'\b(?:all|every|each)\b.*(?:instances|cases|examples)\b'
        ]
        
        # Check for simple patterns
        simple_score = sum(1 for pattern in simple_patterns 
                          if re.search(pattern, query_lower))
        
        # Check for complex patterns
        complex_score = sum(1 for pattern in complex_patterns 
                           if re.search(pattern, query_lower))
        
        # Additional complexity factors
        word_count = len(query.split())
        question_marks = query.count('?')
        multiple_questions = question_marks > 1
        
        # Decision logic
        if complex_score > 0 or multiple_questions or word_count > 20:
            return 'complex'
        elif simple_score > 0 and word_count <= 8:
            return 'simple'
        elif word_count <= 15 and simple_score == 0 and complex_score == 0:
            return 'medium'
        else:
            return 'medium'
    
    def calculate_adaptive_k(self, query_complexity: str, max_context_tokens: int = None) -> int:
        """
        Calculate optimal k value based on query complexity and context constraints.
        
        Args:
            query_complexity: 'simple', 'medium', or 'complex'
            max_context_tokens: Maximum tokens available for context
            
        Returns:
            Optimal k value
        """
        if max_context_tokens is None:
            max_context_tokens = self.MAX_CONTEXT_TOKENS
        
        # Base k values by complexity
        base_k_values = {
            'simple': 6,
            'medium': 10,
            'complex': 15
        }
        
        base_k = base_k_values.get(query_complexity, 10)
        
        # Calculate maximum k based on context window
        max_k_from_context = max_context_tokens // self.TOKENS_PER_CHUNK
        max_k_safe = min(max_k_from_context, 20)  # Hard limit
        
        # Return constrained k value
        return min(base_k, max_k_safe)
    
    def get_quality_threshold(self, query_complexity: str, quality_level: str = 'good') -> float:
        """
        Get similarity score threshold based on query complexity and desired quality.
        
        Args:
            query_complexity: 'simple', 'medium', or 'complex'
            quality_level: 'excellent', 'good', 'fair', or 'poor'
            
        Returns:
            Similarity score threshold
        """
        base_threshold = self.thresholds.get(quality_level, self.thresholds['good'])
        
        # Adjust threshold based on query complexity
        if query_complexity == 'simple':
            # Be more strict for simple queries (expect exact matches)
            adjustment = 0.95 if 'bge' in self.embedding_model else 0.95
        elif query_complexity == 'complex':
            # Be more lenient for complex queries (broader search)
            adjustment = 1.15 if 'bge' in self.embedding_model else 1.15
        else:
            adjustment = 1.0
        
        return base_threshold * adjustment
    
    def filter_by_quality(self, results: List[Tuple[Document, float]], 
                         threshold: float) -> List[Tuple[Document, float]]:
        """
        Filter results by similarity score threshold.
        
        Args:
            results: List of (document, score) tuples
            threshold: Similarity score threshold
            
        Returns:
            Filtered results meeting quality threshold
        """
        return [(doc, score) for doc, score in results if score <= threshold]
    
    def adaptive_retrieve(self, vectorstore, query: str, 
                         fallback_enabled: bool = True) -> Tuple[List[Tuple[Document, float]], Dict[str, Any]]:
        """
        Perform adaptive retrieval with dynamic k and quality filtering.
        
        Args:
            vectorstore: Vector store instance
            query: User query string
            fallback_enabled: Whether to use fallback strategy if insufficient results
            
        Returns:
            Tuple of (filtered_results, retrieval_metadata)
        """
        # Analyze query complexity
        complexity = self.analyze_query_complexity(query)
        
        # Calculate adaptive k
        adaptive_k = self.calculate_adaptive_k(complexity)
        
        # Get initial quality threshold
        quality_threshold = self.get_quality_threshold(complexity, 'good')
        
        # Retrieve initial results with higher k for filtering
        search_k = min(adaptive_k * 2, 25)  # Search more, filter down
        raw_results = vectorstore.similarity_search_with_score(query, k=search_k)
        
        # Filter by quality threshold
        filtered_results = self.filter_by_quality(raw_results, quality_threshold)
        
        # Fallback strategy if insufficient high-quality results
        if fallback_enabled and len(filtered_results) < max(3, adaptive_k // 2):
            # Gradually relax threshold
            fallback_threshold = self.get_quality_threshold(complexity, 'fair')
            filtered_results = self.filter_by_quality(raw_results, fallback_threshold)
            
            # If still insufficient, use poor quality threshold
            if len(filtered_results) < 2:
                fallback_threshold = self.get_quality_threshold(complexity, 'poor')
                filtered_results = self.filter_by_quality(raw_results, fallback_threshold)
                
                # Last resort: return top results regardless of quality
                if len(filtered_results) < 1:
                    filtered_results = raw_results[:adaptive_k]
        
        # Limit to target k
        final_results = filtered_results[:adaptive_k]
        
        # Calculate retrieval metadata
        metadata = {
            'query_complexity': complexity,
            'adaptive_k': adaptive_k,
            'search_k': search_k,
            'quality_threshold': quality_threshold,
            'raw_results_count': len(raw_results),
            'filtered_results_count': len(filtered_results),
            'final_results_count': len(final_results),
            'embedding_model': self.embedding_model,
            'estimated_tokens': len(final_results) * self.TOKENS_PER_CHUNK
        }
        
        return final_results, metadata
    
    def get_retrieval_explanation(self, metadata: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of retrieval strategy.
        
        Args:
            metadata: Retrieval metadata from adaptive_retrieve
            
        Returns:
            Explanation string
        """
        complexity = metadata['query_complexity']
        k = metadata['final_results_count']
        tokens = metadata['estimated_tokens']
        
        explanation = f"Retrieved {k} chunks using {complexity} query strategy "
        explanation += f"(~{tokens} tokens, {metadata['embedding_model']} model)"
        
        if metadata['filtered_results_count'] < metadata['raw_results_count']:
            filtered_out = metadata['raw_results_count'] - metadata['filtered_results_count']
            explanation += f". Filtered out {filtered_out} low-quality results."
        
        return explanation