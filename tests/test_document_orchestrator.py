# tests/test_document_orchestrator.py

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.documents import Document

from backend.document_orchestrator import (
    DocumentOrchestrator, DocumentProcessor, CrossDocumentSynthesizer,
    DocumentProcessingResult, OrchestratorResult
)
from backend.llm_query_classifier import QueryClassification
from backend.token_budget_manager import TokenBudget

class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def mock_smart_vs(self):
        """Mock Smart Vector Store."""
        mock_vs = Mock()
        mock_vectordb = Mock()
        mock_vs.get_or_create_vectorstore.return_value = mock_vectordb
        return mock_vs
    
    @pytest.fixture
    def document_processor(self, mock_smart_vs):
        """Create DocumentProcessor instance for testing."""
        with patch('backend.document_orchestrator.get_embedder') as mock_embedder:
            mock_embedder.return_value = Mock()
            processor = DocumentProcessor(mock_smart_vs, "bge-small-en")
            return processor
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return Document(
            page_content="This is a test legal document with important information about the case.",
            metadata={"filename": "test.pdf"}
        )
    
    @pytest.fixture
    def sample_classification(self):
        """Sample query classification."""
        return QueryClassification(
            query_type="SINGLE_DOCUMENT",
            confidence=0.9,
            processing_strategy="sequential",
            recommended_synthesis="simple",
            token_allocation_bias="balanced",
            reasoning="Test classification",
            cache_key="test123"
        )
    
    @patch('backend.document_orchestrator.semantic_chunk')
    @patch('backend.document_orchestrator.synthesize_answer_cached')
    async def test_process_document_success(self, mock_synthesize, mock_chunk, 
                                          document_processor, sample_document, sample_classification):
        """Test successful document processing."""
        # Mock dependencies
        mock_chunk.return_value = [("chunk1", {"page": 1}), ("chunk2", {"page": 1})]
        mock_synthesize.return_value = "This is the answer to the question."
        
        # Mock retrieval
        mock_vectordb = document_processor.smart_vs.get_or_create_vectorstore.return_value
        with patch('backend.document_orchestrator.AdaptiveRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever
            mock_retriever.adaptive_retrieve.return_value = (
                [(sample_document, 0.95)], 
                {"query_complexity": "simple", "final_results_count": 1}
            )
            
            # Process document
            result = await document_processor.process_document(
                sample_document, "doc_1", "test.pdf", "What is this about?", 1000, sample_classification
            )
        
        # Assertions
        assert isinstance(result, DocumentProcessingResult)
        assert result.document_id == "doc_1"
        assert result.document_name == "test.pdf"
        assert result.answer == "This is the answer to the question."
        assert result.error is None
        assert result.tokens_used > 0
        assert result.processing_time > 0
    
    @patch('backend.document_orchestrator.semantic_chunk')
    async def test_process_document_with_error(self, mock_chunk, document_processor, 
                                             sample_document, sample_classification):
        """Test document processing with error."""
        # Mock an error in chunking
        mock_chunk.side_effect = Exception("Chunking failed")
        
        result = await document_processor.process_document(
            sample_document, "doc_1", "test.pdf", "What is this about?", 1000, sample_classification
        )
        
        # Assertions
        assert isinstance(result, DocumentProcessingResult)
        assert result.error is not None
        assert "Chunking failed" in result.error
        assert result.tokens_used == 0
    
    def test_get_context_prefix(self, document_processor, sample_classification):
        """Test context prefix generation."""
        prefix = document_processor._get_context_prefix(sample_classification, "test.pdf")
        assert "test.pdf" in prefix
        assert "Document Analysis" in prefix

class TestCrossDocumentSynthesizer:
    """Test suite for CrossDocumentSynthesizer class."""
    
    @pytest.fixture
    def synthesizer(self):
        """Create CrossDocumentSynthesizer instance."""
        return CrossDocumentSynthesizer(token_budget=1200)
    
    @pytest.fixture
    def sample_results(self):
        """Sample document processing results."""
        results = []
        for i in range(3):
            result = DocumentProcessingResult(
                document_id=f"doc_{i}",
                document_name=f"document_{i}.pdf",
                query="Test query",
                answer=f"This is answer {i} with legal analysis and court ruling information.",
                retrieved_chunks=[],
                retrieval_metadata={},
                tokens_used=200,
                processing_time=1.0
            )
            results.append(result)
        return results
    
    @pytest.fixture
    def comparative_classification(self):
        """Comparative query classification."""
        return QueryClassification(
            query_type="COMPARATIVE",
            confidence=0.9,
            processing_strategy="parallel",
            recommended_synthesis="comparative",
            token_allocation_bias="synthesis_heavy",
            reasoning="Comparative analysis",
            cache_key="comp123"
        )
    
    def test_extract_cross_document_insights_basic(self, synthesizer, sample_results, comparative_classification):
        """Test basic cross-document insights extraction."""
        insights = synthesizer.extract_cross_document_insights(sample_results, comparative_classification)
        
        assert insights['document_count'] == 3
        assert insights['successful_documents'] == 3
        assert insights['failed_documents'] == 0
        assert insights['query_type'] == 'COMPARATIVE'
        assert insights['processing_confidence'] == 0.9
    
    def test_analyze_comparisons(self, synthesizer, sample_results):
        """Test comparison analysis."""
        analysis = synthesizer._analyze_comparisons(sample_results)
        
        assert 'comparison_pairs' in analysis
        assert 'key_differences' in analysis
        assert 'similarities' in analysis
        assert isinstance(analysis['comparison_pairs'], list)
    
    def test_analyze_patterns(self, synthesizer, sample_results):
        """Test pattern analysis."""
        analysis = synthesizer._analyze_patterns(sample_results)
        
        assert 'recurring_patterns' in analysis
        assert 'contradictions' in analysis
        assert 'consensus_points' in analysis
        assert isinstance(analysis['recurring_patterns'], list)
    
    def test_analyze_themes(self, synthesizer, sample_results):
        """Test theme analysis."""
        analysis = synthesizer._analyze_themes(sample_results)
        
        assert 'major_themes' in analysis
        assert 'theme_frequency' in analysis
        assert 'theme_distribution' in analysis
        assert isinstance(analysis['theme_frequency'], dict)
    
    def test_analyze_coverage(self, synthesizer, sample_results):
        """Test coverage analysis."""
        analysis = synthesizer._analyze_coverage(sample_results)
        
        assert 'coverage_completeness' in analysis
        assert 'information_density' in analysis
        assert 'average_response_length' in analysis
        assert analysis['coverage_completeness'] >= 0
    
    def test_synthesize_single_document(self, synthesizer, sample_results):
        """Test single document synthesis."""
        result = synthesizer._synthesize_single_document("Test query", sample_results[0])
        
        assert "document_0.pdf" in result
        assert sample_results[0].answer in result
    
    @patch('backend.document_orchestrator.synthesize_answer_cached')
    async def test_synthesize_comparative(self, mock_synthesize, synthesizer, 
                                        sample_results, comparative_classification):
        """Test comparative synthesis."""
        mock_synthesize.return_value = "Comparative analysis result"
        insights = {'comparison_pairs': [{'doc1': 'doc1', 'doc2': 'doc2', 'common_concepts': 5}]}
        
        result = await synthesizer._synthesize_comparative("Compare documents", sample_results, insights)
        
        assert "Comparative analysis result" in result
        mock_synthesize.assert_called_once()
    
    @patch('backend.document_orchestrator.synthesize_answer_cached')
    async def test_synthesize_with_llm_error(self, mock_synthesize, synthesizer, 
                                           sample_results, comparative_classification):
        """Test synthesis with LLM error (fallback)."""
        mock_synthesize.side_effect = Exception("LLM failed")
        
        result = await synthesizer._synthesize_comparative("Compare documents", sample_results, {})
        
        # Should fall back to simple concatenation
        assert "document_0.pdf" in result
        assert "document_1.pdf" in result
        assert "document_2.pdf" in result

class TestDocumentOrchestrator:
    """Test suite for DocumentOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create DocumentOrchestrator instance."""
        with patch('backend.document_orchestrator.TokenBudgetManager'), \
             patch('backend.document_orchestrator.LLMQueryClassifier'), \
             patch('backend.document_orchestrator.SmartVectorStore'):
            return DocumentOrchestrator()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        docs = []
        for i in range(3):
            doc = Document(
                page_content=f"This is document {i} with legal content and analysis.",
                metadata={"filename": f"doc_{i}.pdf"}
            )
            docs.append(doc)
        return docs
    
    @pytest.fixture
    def sample_classification(self):
        """Sample query classification."""
        return QueryClassification(
            query_type="COMPARATIVE",
            confidence=0.9,
            processing_strategy="parallel",
            recommended_synthesis="comparative",
            token_allocation_bias="balanced",
            reasoning="Test classification",
            cache_key="test123"
        )
    
    def test_allocate_tokens_based_on_classification(self, orchestrator, sample_documents, sample_classification):
        """Test token allocation based on classification."""
        # Mock the token manager
        mock_budgets = {
            "doc_0": Mock(allocated_tokens=1000),
            "doc_1": Mock(allocated_tokens=1000),
            "doc_2": Mock(allocated_tokens=1000)
        }
        orchestrator.token_manager.allocate_document_budgets.return_value = mock_budgets
        
        budgets = orchestrator._allocate_tokens_based_on_classification(sample_documents, sample_classification)
        
        assert len(budgets) == 3
        orchestrator.token_manager.allocate_document_budgets.assert_called()
    
    @patch('backend.document_orchestrator.DocumentProcessor')
    async def test_process_documents_parallel(self, mock_processor_class, orchestrator, 
                                            sample_documents, sample_classification):
        """Test parallel document processing."""
        # Mock document processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock processing results
        async def mock_process_document(*args, **kwargs):
            return DocumentProcessingResult(
                document_id="doc_0",
                document_name="doc_0.pdf",
                query="test",
                answer="test answer",
                retrieved_chunks=[],
                retrieval_metadata={},
                tokens_used=100,
                processing_time=1.0
            )
        
        mock_processor.process_document = mock_process_document
        
        # Mock budgets
        budgets = {
            "doc_0": Mock(allocated_tokens=1000),
            "doc_1": Mock(allocated_tokens=1000),
            "doc_2": Mock(allocated_tokens=1000)
        }
        
        results = await orchestrator._process_documents_parallel(
            sample_documents, ["doc_0.pdf", "doc_1.pdf", "doc_2.pdf"],
            "test query", "bge-small-en", budgets, sample_classification
        )
        
        assert len(results) == 3
        assert all(isinstance(r, DocumentProcessingResult) for r in results)
    
    @patch('backend.document_orchestrator.DocumentProcessor')
    async def test_process_documents_sequential(self, mock_processor_class, orchestrator, 
                                              sample_documents, sample_classification):
        """Test sequential document processing."""
        # Mock document processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock processing results
        async def mock_process_document(*args, **kwargs):
            return DocumentProcessingResult(
                document_id="doc_0",
                document_name="doc_0.pdf",
                query="test",
                answer="test answer",
                retrieved_chunks=[],
                retrieval_metadata={},
                tokens_used=100,
                processing_time=1.0
            )
        
        mock_processor.process_document = mock_process_document
        
        # Mock budgets
        budgets = {
            "doc_0": Mock(allocated_tokens=1000),
            "doc_1": Mock(allocated_tokens=1000),
            "doc_2": Mock(allocated_tokens=1000)
        }
        
        results = await orchestrator._process_documents_sequential(
            sample_documents, ["doc_0.pdf", "doc_1.pdf", "doc_2.pdf"],
            "test query", "bge-small-en", budgets, sample_classification
        )
        
        assert len(results) == 3
        assert all(isinstance(r, DocumentProcessingResult) for r in results)
    
    def test_perform_simple_synthesis_single_document(self, orchestrator):
        """Test simple synthesis for single document."""
        result = DocumentProcessingResult(
            document_id="doc_0",
            document_name="test.pdf",
            query="test",
            answer="test answer",
            retrieved_chunks=[],
            retrieval_metadata={},
            tokens_used=100,
            processing_time=1.0
        )
        
        synthesis, insights = orchestrator._perform_simple_synthesis([result])
        
        assert "test.pdf" in synthesis
        assert "test answer" in synthesis
        assert insights['synthesis_type'] == 'single_document'
        assert insights['document_count'] == 1
    
    def test_perform_simple_synthesis_multiple_documents(self, orchestrator):
        """Test simple synthesis for multiple documents."""
        results = []
        for i in range(3):
            result = DocumentProcessingResult(
                document_id=f"doc_{i}",
                document_name=f"doc_{i}.pdf",
                query="test",
                answer=f"answer {i}",
                retrieved_chunks=[],
                retrieval_metadata={},
                tokens_used=100,
                processing_time=1.0
            )
            results.append(result)
        
        synthesis, insights = orchestrator._perform_simple_synthesis(results)
        
        assert "Multi-Document Analysis" in synthesis
        assert "doc_0.pdf" in synthesis
        assert "doc_1.pdf" in synthesis
        assert "doc_2.pdf" in synthesis
        assert insights['synthesis_type'] == 'simple_aggregation'
        assert insights['document_count'] == 3
    
    async def test_process_documents_with_empty_input(self, orchestrator):
        """Test processing with empty document list."""
        result = await orchestrator.process_documents([], [], "test query")
        
        assert isinstance(result, OrchestratorResult)
        assert result.error is not None
        assert "No documents provided" in result.error
    
    async def test_process_documents_with_mismatched_inputs(self, orchestrator, sample_documents):
        """Test processing with mismatched document and name lists."""
        result = await orchestrator.process_documents(
            sample_documents, ["doc1.pdf"], "test query"  # Only one name for 3 docs
        )
        
        assert isinstance(result, OrchestratorResult)
        assert result.error is not None
        assert "must match" in result.error
    
    def test_check_document_feasibility(self, orchestrator):
        """Test document feasibility checking."""
        # Mock token manager check
        orchestrator.token_manager.min_tokens_per_doc = 600
        orchestrator.token_manager.available_for_docs = 3000
        
        feasibility = orchestrator.check_document_feasibility(4)
        
        assert 'feasible' in feasibility
        assert 'documents_requested' in feasibility
        assert 'max_feasible_documents' in feasibility
        assert feasibility['documents_requested'] == 4

class TestIntegration:
    """Integration tests for the complete orchestrator system."""
    
    @pytest.fixture
    def full_orchestrator(self):
        """Create a full orchestrator with real dependencies for integration testing."""
        return DocumentOrchestrator()
    
    def test_orchestrator_initialization(self, full_orchestrator):
        """Test that orchestrator initializes correctly with all components."""
        assert full_orchestrator.token_manager is not None
        assert full_orchestrator.query_classifier is not None
        assert full_orchestrator.smart_vs is not None
        assert full_orchestrator.max_concurrent_docs == 3
    
    def test_token_allocation_summary(self, full_orchestrator):
        """Test getting token allocation summary."""
        summary = full_orchestrator.get_token_allocation_summary()
        assert isinstance(summary, dict)
    
    def test_clear_classifier_cache(self, full_orchestrator):
        """Test clearing classifier cache."""
        # Should not raise any errors
        full_orchestrator.clear_classifier_cache()

# Test fixtures and utilities
@pytest.fixture
def sample_query_classification():
    """Standard query classification for testing."""
    return QueryClassification(
        query_type="COMPARATIVE",
        confidence=0.85,
        processing_strategy="parallel",
        recommended_synthesis="comparative",
        token_allocation_bias="balanced",
        reasoning="Test classification for comparison",
        cache_key="test_cache_key"
    )

@pytest.fixture
def mock_document_results():
    """Mock document processing results."""
    results = []
    for i in range(2):
        result = DocumentProcessingResult(
            document_id=f"doc_{i}",
            document_name=f"document_{i}.pdf",
            query="Compare these documents",
            answer=f"Analysis result {i} with legal precedent and court decision.",
            retrieved_chunks=[],
            retrieval_metadata={'query_complexity': 'medium', 'final_results_count': 5},
            tokens_used=150,
            processing_time=2.5
        )
        results.append(result)
    return results

if __name__ == "__main__":
    pytest.main([__file__, "-v"])