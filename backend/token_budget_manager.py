# backend/token_budget_manager.py

from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from langchain_core.documents import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenBudget:
    """Token budget allocation for a document or synthesis phase."""
    document_id: str
    allocated_tokens: int
    used_tokens: int = 0
    remaining_tokens: int = 0
    
    def __post_init__(self):
        self.remaining_tokens = self.allocated_tokens - self.used_tokens
    
    def use_tokens(self, tokens: int) -> bool:
        """Use tokens from budget. Returns True if successful."""
        if self.remaining_tokens >= tokens:
            self.used_tokens += tokens
            self.remaining_tokens -= tokens
            return True
        return False
    
    def get_utilization(self) -> float:
        """Get budget utilization percentage."""
        if self.allocated_tokens == 0:
            return 0.0
        return (self.used_tokens / self.allocated_tokens) * 100

class TokenBudgetManager:
    """
    Manages token budgets for multi-document processing with Llama 3 tokenization.
    Uses conservative buffer strategy to maintain performance quality.
    """
    
    def __init__(self, max_context_tokens: int = 8192, performance_buffer: int = 1500):
        """
        Initialize token budget manager for Llama 3 8K context.
        
        Args:
            max_context_tokens: Maximum context window size (8192 for Llama 3)
            performance_buffer: Buffer to maintain quality (avoid degradation near limits)
        """
        self.max_context_tokens = max_context_tokens
        self.performance_buffer = performance_buffer
        self.effective_limit = max_context_tokens - performance_buffer  # 6692 tokens
        
        # Conservative allocation strategy
        self.synthesis_buffer = 1200      # Cross-document synthesis
        self.response_buffer = 800        # LLM response generation
        self.available_for_docs = self.effective_limit - self.synthesis_buffer - self.response_buffer  # 4692 tokens
        
        # Initialize Llama 3 tokenizer locally
        try:
            # Use Llama 3 tokenizer - adjust model name as needed
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                local_files_only=True,
                trust_remote_code=True
            )
            logger.info("Loaded Llama 3 tokenizer successfully")
        except Exception as e:
            logger.warning(f"Failed to load Llama 3 tokenizer: {e}")
            try:
                # Fallback to any available Llama tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "huggingface/CodeBERTa-small-v1",  # Fallback tokenizer
                    local_files_only=True
                )
                logger.info("Using fallback tokenizer")
            except Exception as e2:
                logger.error(f"Failed to load any tokenizer: {e2}")
                self.tokenizer = None
        
        # Document budgets
        self.document_budgets: Dict[str, TokenBudget] = {}
        self.synthesis_budget: Optional[TokenBudget] = None
        
        # Token allocation constraints
        self.min_tokens_per_doc = 600   # Minimum viable tokens per document
        self.max_tokens_per_doc = 2000  # Maximum tokens per document (conservative)
        
        logger.info(f"TokenBudgetManager initialized:")
        logger.info(f"  Max context: {self.max_context_tokens}")
        logger.info(f"  Performance buffer: {self.performance_buffer}")
        logger.info(f"  Effective limit: {self.effective_limit}")
        logger.info(f"  Available for docs: {self.available_for_docs}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens in text using Llama 3 tokenizer.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens (estimated)
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}. Using fallback.")
        
        # Fallback estimation for Llama models: ~3.5 characters per token
        return len(text) // 3.5
    
    def estimate_document_complexity(self, document: Document) -> str:
        """
        Estimate document complexity for token allocation.
        
        Args:
            document: Document to analyze
            
        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        content = document.page_content
        
        # Analyze content characteristics
        word_count = len(content.split())
        char_count = len(content)
        
        # Look for complex patterns typical in legal documents
        complex_patterns = [
            'analysis', 'comparison', 'evaluation', 'assessment',
            'however', 'therefore', 'consequently', 'furthermore',
            'table', 'figure', 'exhibit', 'appendix', 'whereas',
            'plaintiff', 'defendant', 'court', 'ruling', 'precedent'
        ]
        
        complexity_score = sum(1 for pattern in complex_patterns if pattern in content.lower())
        
        # Legal document specific complexity indicators
        legal_indicators = content.lower().count('§') + content.lower().count('v.') + content.count('(')
        
        # Classification thresholds adjusted for legal documents
        if word_count > 4000 or complexity_score > 6 or legal_indicators > 50:
            return 'complex'
        elif word_count > 1500 or complexity_score > 3 or legal_indicators > 20:
            return 'medium'
        else:
            return 'simple'
    
    def allocate_document_budgets(self, documents: List[Document]) -> Dict[str, TokenBudget]:
        """
        Allocate token budgets across multiple documents with conservative limits.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary mapping document IDs to TokenBudget objects
        """
        if not documents:
            return {}
        
        num_docs = len(documents)
        
        # Check if we have enough tokens for minimum viable processing
        min_total_needed = num_docs * self.min_tokens_per_doc
        if min_total_needed > self.available_for_docs:
            logger.warning(f"Insufficient tokens for {num_docs} documents.")
            logger.warning(f"Need {min_total_needed}, have {self.available_for_docs}")
            
            # Reduce synthesis buffer to make room, but keep response buffer
            self.synthesis_buffer = max(600, self.synthesis_buffer // 2)
            self.available_for_docs = self.effective_limit - self.synthesis_buffer - self.response_buffer
            
            if min_total_needed > self.available_for_docs:
                logger.error(f"Cannot process {num_docs} documents with available tokens")
                # Process only what we can handle
                max_docs = self.available_for_docs // self.min_tokens_per_doc
                logger.warning(f"Limiting to {max_docs} documents")
                documents = documents[:max_docs]
                num_docs = len(documents)
        
        # Analyze document complexities
        doc_complexities = {}
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc_complexities[doc_id] = {
                'document': doc,
                'complexity': self.estimate_document_complexity(doc),
                'estimated_tokens': self.estimate_tokens(doc.page_content)
            }
        
        # Allocate tokens based on complexity and content size
        complexity_weights = {'simple': 1.0, 'medium': 1.4, 'complex': 1.8}
        total_complexity_weight = 0
        
        for doc_id, info in doc_complexities.items():
            weight = complexity_weights[info['complexity']]
            # Boost allocation for longer documents (but cap the boost)
            if info['estimated_tokens'] > 1500:
                weight *= 1.2
            total_complexity_weight += weight
        
        # Allocate tokens proportionally with conservative limits
        budgets = {}
        total_allocated = 0
        
        for doc_id, info in doc_complexities.items():
            weight = complexity_weights[info['complexity']]
            if info['estimated_tokens'] > 1500:
                weight *= 1.2
            
            # Calculate proportional allocation
            proportion = weight / total_complexity_weight
            allocated = int(self.available_for_docs * proportion)
            
            # Enforce conservative min/max constraints
            allocated = max(self.min_tokens_per_doc, 
                          min(self.max_tokens_per_doc, allocated))
            
            budgets[doc_id] = TokenBudget(
                document_id=doc_id,
                allocated_tokens=allocated
            )
            total_allocated += allocated
        
        # Handle remaining tokens conservatively
        remaining = self.available_for_docs - total_allocated
        if remaining > 0:
            # Distribute remaining tokens to complex documents, but cap per document
            complex_docs = [doc_id for doc_id, info in doc_complexities.items() 
                          if info['complexity'] == 'complex']
            if complex_docs:
                bonus_per_doc = min(200, remaining // len(complex_docs))  # Cap bonus
                for doc_id in complex_docs:
                    current_allocation = budgets[doc_id].allocated_tokens
                    if current_allocation + bonus_per_doc <= self.max_tokens_per_doc:
                        budgets[doc_id].allocated_tokens += bonus_per_doc
                        budgets[doc_id].remaining_tokens += bonus_per_doc
        
        # Create synthesis budget
        self.synthesis_budget = TokenBudget(
            document_id="synthesis",
            allocated_tokens=self.synthesis_buffer
        )
        
        self.document_budgets = budgets
        
        # Log allocation summary
        logger.info(f"Conservative token allocation for {num_docs} documents:")
        for doc_id, budget in budgets.items():
            complexity = doc_complexities[doc_id]['complexity']
            logger.info(f"  {doc_id}: {budget.allocated_tokens} tokens ({complexity})")
        logger.info(f"  Synthesis: {self.synthesis_budget.allocated_tokens} tokens")
        logger.info(f"  Response buffer: {self.response_buffer} tokens")
        logger.info(f"  Total allocated: {total_allocated + self.synthesis_buffer + self.response_buffer}")
        
        return budgets
    
    def get_document_budget(self, doc_id: str) -> Optional[TokenBudget]:
        """Get budget for a specific document."""
        return self.document_budgets.get(doc_id)
    
    def get_synthesis_budget(self) -> Optional[TokenBudget]:
        """Get synthesis phase budget."""
        return self.synthesis_budget
    
    def use_document_tokens(self, doc_id: str, tokens: int) -> bool:
        """Use tokens from a document's budget."""
        budget = self.document_budgets.get(doc_id)
        if budget:
            return budget.use_tokens(tokens)
        return False
    
    def use_synthesis_tokens(self, tokens: int) -> bool:
        """Use tokens from synthesis budget."""
        if self.synthesis_budget:
            return self.synthesis_budget.use_tokens(tokens)
        return False
    
    def get_total_used_tokens(self) -> int:
        """Get total tokens used across all budgets."""
        total = sum(budget.used_tokens for budget in self.document_budgets.values())
        if self.synthesis_budget:
            total += self.synthesis_budget.used_tokens
        return total
    
    def check_token_safety(self, additional_tokens: int = 0) -> bool:
        """
        Check if we're within safe token limits with performance buffer.
        
        Args:
            additional_tokens: Additional tokens we plan to use
            
        Returns:
            True if safe, False if would exceed effective limits
        """
        current_used = self.get_total_used_tokens()
        projected_total = current_used + additional_tokens
        
        # Check against effective limit (with performance buffer)
        return projected_total <= self.effective_limit
    
    def get_recommended_k(self, doc_id: str, tokens_per_chunk: int = 200) -> int:
        """
        Get recommended k value for retrieval based on document budget.
        Conservative estimation for Llama 3.
        
        Args:
            doc_id: Document identifier
            tokens_per_chunk: Estimated tokens per chunk (conservative for Llama)
            
        Returns:
            Recommended k value for retrieval
        """
        budget = self.document_budgets.get(doc_id)
        if not budget:
            return 5  # Conservative default
        
        # Reserve tokens for query processing and overhead
        query_overhead = 200  # Conservative overhead estimate
        usable_tokens = budget.remaining_tokens - query_overhead
        
        if usable_tokens <= 0:
            return 3  # Minimum viable
        
        # Calculate maximum chunks we can afford
        max_k = usable_tokens // tokens_per_chunk
        
        # Apply conservative bounds for quality
        return max(3, min(max_k, 12))  # More conservative max
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive allocation summary with safety metrics."""
        doc_summaries = {}
        total_allocated = 0
        total_used = 0
        
        for doc_id, budget in self.document_budgets.items():
            doc_summaries[doc_id] = {
                'allocated': budget.allocated_tokens,
                'used': budget.used_tokens,
                'remaining': budget.remaining_tokens,
                'utilization': budget.get_utilization()
            }
            total_allocated += budget.allocated_tokens
            total_used += budget.used_tokens
        
        # Include synthesis budget
        synthesis_summary = None
        if self.synthesis_budget:
            synthesis_summary = {
                'allocated': self.synthesis_budget.allocated_tokens,
                'used': self.synthesis_budget.used_tokens,
                'remaining': self.synthesis_budget.remaining_tokens,
                'utilization': self.synthesis_budget.get_utilization()
            }
            total_allocated += self.synthesis_budget.allocated_tokens
            total_used += self.synthesis_budget.used_tokens
        
        # Safety metrics
        safety_buffer_used = self.max_context_tokens - self.effective_limit
        
        return {
            'max_context_tokens': self.max_context_tokens,
            'effective_limit': self.effective_limit,
            'performance_buffer': self.performance_buffer,
            'total_allocated': total_allocated,
            'total_used': total_used,
            'total_remaining': self.effective_limit - total_used,
            'safety_buffer': safety_buffer_used,
            'utilization_of_effective': (total_used / self.effective_limit) * 100,
            'utilization_of_max': (total_used / self.max_context_tokens) * 100,
            'is_safe': self.check_token_safety(),
            'documents': doc_summaries,
            'synthesis': synthesis_summary
        }