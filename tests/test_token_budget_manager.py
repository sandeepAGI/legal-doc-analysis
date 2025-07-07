# tests/test_token_budget_manager.py

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from backend.token_budget_manager import TokenBudgetManager, TokenBudget

class TestTokenBudget:
    """Test suite for TokenBudget class."""
    
    def test_token_budget_initialization(self):
        """Test TokenBudget initialization."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=1000)
        
        assert budget.document_id == "doc_1"
        assert budget.allocated_tokens == 1000
        assert budget.used_tokens == 0
        assert budget.remaining_tokens == 1000
    
    def test_token_budget_with_used_tokens(self):
        """Test TokenBudget with initial used tokens."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=1000, used_tokens=300)
        
        assert budget.remaining_tokens == 700
    
    def test_use_tokens_success(self):
        """Test successful token usage."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=1000)
        
        result = budget.use_tokens(300)
        
        assert result is True
        assert budget.used_tokens == 300
        assert budget.remaining_tokens == 700
    
    def test_use_tokens_insufficient(self):
        """Test token usage with insufficient budget."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=1000)
        budget.use_tokens(800)  # Use most tokens
        
        result = budget.use_tokens(300)  # Try to use more than remaining
        
        assert result is False
        assert budget.used_tokens == 800  # Should remain unchanged
        assert budget.remaining_tokens == 200
    
    def test_get_utilization(self):
        """Test utilization calculation."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=1000)
        budget.use_tokens(250)
        
        utilization = budget.get_utilization()
        
        assert utilization == 25.0
    
    def test_get_utilization_zero_allocation(self):
        """Test utilization with zero allocation."""
        budget = TokenBudget(document_id="doc_1", allocated_tokens=0)
        
        utilization = budget.get_utilization()
        
        assert utilization == 0.0

class TestTokenBudgetManager:
    """Test suite for TokenBudgetManager class."""
    
    @pytest.fixture
    def token_manager(self):
        """Create TokenBudgetManager instance for testing."""
        with patch('backend.token_budget_manager.AutoTokenizer'):
            return TokenBudgetManager(max_context_tokens=8192, performance_buffer=1500)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        docs = []
        for i in range(3):
            doc = Document(
                page_content=f"This is document {i} with some legal content and analysis. " * 50,  # ~150 words
                metadata={"filename": f"doc_{i}.pdf"}
            )
            docs.append(doc)
        return docs
    
    def test_initialization(self, token_manager):
        """Test TokenBudgetManager initialization."""
        assert token_manager.max_context_tokens == 8192
        assert token_manager.performance_buffer == 1500
        assert token_manager.effective_limit == 6692
        assert token_manager.synthesis_buffer == 1200
        assert token_manager.response_buffer == 800
        assert token_manager.available_for_docs == 4692
    
    def test_estimate_tokens_with_tokenizer(self):
        """Test token estimation with mock tokenizer."""
        with patch('backend.token_budget_manager.AutoTokenizer') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            manager = TokenBudgetManager()
            tokens = manager.estimate_tokens("Hello world")
            
            assert tokens == 5
    
    def test_estimate_tokens_fallback(self, token_manager):
        """Test token estimation fallback when tokenizer fails."""
        # Force tokenizer to None
        token_manager.tokenizer = None
        
        tokens = token_manager.estimate_tokens("Hello world test")  # 17 characters
        
        # Should use fallback: length // 3.5
        expected = 17 // 3.5
        assert abs(tokens - expected) < 1
    
    def test_estimate_document_complexity_simple(self, token_manager):
        """Test document complexity estimation for simple document."""
        doc = Document(page_content="This is a short legal document with basic content.")
        
        complexity = token_manager.estimate_document_complexity(doc)
        
        assert complexity == "simple"
    
    def test_estimate_document_complexity_medium(self, token_manager):
        """Test document complexity estimation for medium document."""
        content = "This document contains analysis and evaluation of legal precedents and court rulings. " * 50
        doc = Document(page_content=content)
        
        complexity = token_manager.estimate_document_complexity(doc)
        
        assert complexity in ["medium", "complex"]  # Could be either based on content length and complexity
    
    def test_estimate_document_complexity_complex(self, token_manager):
        """Test document complexity estimation for complex document."""
        content = ("This complex legal document contains extensive analysis, comparison, "
                  "evaluation, assessment of court rulings, with tables, figures, and exhibits. " * 100)
        doc = Document(page_content=content)
        
        complexity = token_manager.estimate_document_complexity(doc)
        
        assert complexity == "complex"
    
    def test_allocate_document_budgets_basic(self, token_manager, sample_documents):
        """Test basic document budget allocation."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        
        assert len(budgets) == 3
        assert "doc_0" in budgets
        assert "doc_1" in budgets
        assert "doc_2" in budgets
        
        # Check that each budget is within reasonable bounds
        for budget in budgets.values():
            assert isinstance(budget, TokenBudget)
            assert budget.allocated_tokens >= token_manager.min_tokens_per_doc
            assert budget.allocated_tokens <= token_manager.max_tokens_per_doc
    
    def test_allocate_document_budgets_empty_list(self, token_manager):
        """Test budget allocation with empty document list."""
        budgets = token_manager.allocate_document_budgets([])
        
        assert budgets == {}
    
    def test_allocate_document_budgets_insufficient_tokens(self, token_manager):
        """Test budget allocation when insufficient tokens available."""
        # Create many documents that would exceed available tokens
        many_docs = [Document(page_content="test") for _ in range(10)]
        
        budgets = token_manager.allocate_document_budgets(many_docs)
        
        # Should still allocate budgets, but may reduce synthesis buffer
        assert len(budgets) > 0
        assert len(budgets) <= len(many_docs)
    
    def test_get_document_budget(self, token_manager, sample_documents):
        """Test getting document budget."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        
        budget = token_manager.get_document_budget("doc_0")
        
        assert budget is not None
        assert budget.document_id == "doc_0"
    
    def test_get_document_budget_nonexistent(self, token_manager):
        """Test getting nonexistent document budget."""
        budget = token_manager.get_document_budget("nonexistent")
        
        assert budget is None
    
    def test_get_synthesis_budget(self, token_manager, sample_documents):
        """Test getting synthesis budget."""
        token_manager.allocate_document_budgets(sample_documents)
        
        synthesis_budget = token_manager.get_synthesis_budget()
        
        assert synthesis_budget is not None
        assert synthesis_budget.document_id == "synthesis"
        assert synthesis_budget.allocated_tokens == token_manager.synthesis_buffer
    
    def test_use_document_tokens(self, token_manager, sample_documents):
        """Test using document tokens."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        
        result = token_manager.use_document_tokens("doc_0", 300)
        
        assert result is True
        
        budget = token_manager.get_document_budget("doc_0")
        assert budget.used_tokens == 300
    
    def test_use_document_tokens_insufficient(self, token_manager, sample_documents):
        """Test using more document tokens than available."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        budget = token_manager.get_document_budget("doc_0")
        
        # Try to use more than allocated
        result = token_manager.use_document_tokens("doc_0", budget.allocated_tokens + 100)
        
        assert result is False
    
    def test_use_synthesis_tokens(self, token_manager, sample_documents):
        """Test using synthesis tokens."""
        token_manager.allocate_document_budgets(sample_documents)
        
        result = token_manager.use_synthesis_tokens(500)
        
        assert result is True
        
        synthesis_budget = token_manager.get_synthesis_budget()
        assert synthesis_budget.used_tokens == 500
    
    def test_get_total_used_tokens(self, token_manager, sample_documents):
        """Test getting total used tokens."""
        token_manager.allocate_document_budgets(sample_documents)
        
        # Use some tokens
        token_manager.use_document_tokens("doc_0", 200)
        token_manager.use_document_tokens("doc_1", 150)
        token_manager.use_synthesis_tokens(300)
        
        total = token_manager.get_total_used_tokens()
        
        assert total == 650  # 200 + 150 + 300
    
    def test_check_token_safety(self, token_manager, sample_documents):
        """Test token safety checking."""
        token_manager.allocate_document_budgets(sample_documents)
        
        # Should be safe initially
        assert token_manager.check_token_safety() is True
        
        # Should be safe with moderate additional tokens
        assert token_manager.check_token_safety(1000) is True
        
        # Should be unsafe with excessive additional tokens
        assert token_manager.check_token_safety(10000) is False
    
    def test_get_recommended_k(self, token_manager, sample_documents):
        """Test getting recommended k value."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        
        k = token_manager.get_recommended_k("doc_0")
        
        assert isinstance(k, int)
        assert k >= 3  # Minimum
        assert k <= 15  # Maximum
    
    def test_get_recommended_k_low_budget(self, token_manager, sample_documents):
        """Test getting recommended k with low token budget."""
        budgets = token_manager.allocate_document_budgets(sample_documents)
        
        # Use most of the budget
        budget = token_manager.get_document_budget("doc_0")
        token_manager.use_document_tokens("doc_0", budget.allocated_tokens - 400)
        
        k = token_manager.get_recommended_k("doc_0")
        
        assert k == 3  # Should return minimum
    
    def test_get_allocation_summary(self, token_manager, sample_documents):
        """Test getting allocation summary."""
        token_manager.allocate_document_budgets(sample_documents)
        
        # Use some tokens
        token_manager.use_document_tokens("doc_0", 200)
        token_manager.use_synthesis_tokens(300)
        
        summary = token_manager.get_allocation_summary()
        
        assert 'max_context_tokens' in summary
        assert 'effective_limit' in summary
        assert 'total_allocated' in summary
        assert 'total_used' in summary
        assert 'utilization_of_effective' in summary
        assert 'is_safe' in summary
        assert 'documents' in summary
        assert 'synthesis' in summary
        
        # Check document summaries
        assert 'doc_0' in summary['documents']
        assert summary['documents']['doc_0']['used'] == 200
        
        # Check synthesis summary
        assert summary['synthesis']['used'] == 300

class TestTokenBudgetManagerEdgeCases:
    """Test edge cases and error conditions for TokenBudgetManager."""
    
    def test_zero_max_tokens(self):
        """Test initialization with zero max tokens."""
        with patch('backend.token_budget_manager.AutoTokenizer'):
            manager = TokenBudgetManager(max_context_tokens=0, performance_buffer=0)
            
            assert manager.max_context_tokens == 0
            assert manager.effective_limit == 0
    
    def test_performance_buffer_larger_than_max(self):
        """Test performance buffer larger than max context."""
        with patch('backend.token_budget_manager.AutoTokenizer'):
            manager = TokenBudgetManager(max_context_tokens=1000, performance_buffer=1500)
            
            # Should handle gracefully
            assert manager.performance_buffer == 1500
            assert manager.effective_limit < 0  # Negative, but should be handled
    
    def test_single_document_allocation(self):
        """Test allocation with single document."""
        with patch('backend.token_budget_manager.AutoTokenizer'):
            manager = TokenBudgetManager()
            
            # Create a single test document
            doc = Document(page_content="This is a test document for allocation testing.")
            
            budgets = manager.allocate_document_budgets([doc])
            
            assert len(budgets) == 1
            assert "doc_0" in budgets
            
            # Should get a reasonable allocation
            budget = budgets["doc_0"]
            assert budget.allocated_tokens >= manager.min_tokens_per_doc

if __name__ == "__main__":
    pytest.main([__file__, "-v"])