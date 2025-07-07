# backend/document_orchestrator.py

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from langchain_core.documents import Document
import logging

from .token_budget_manager import TokenBudgetManager
from .llm_query_classifier import LLMQueryClassifier, QueryClassification
from .smart_vectorstore import SmartVectorStore
from .adaptive_retrieval import AdaptiveRetriever
from .embedder import get_embedder
from .chunker import semantic_chunk
from .loader import load_document
from .llm_wrapper import synthesize_answer_cached, get_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessingResult:
    """Result from processing a single document."""
    document_id: str
    document_name: str
    query: str
    answer: str
    retrieved_chunks: List[Tuple[Document, float]]
    retrieval_metadata: Dict[str, Any]
    tokens_used: int
    processing_time: float
    error: Optional[str] = None

@dataclass
class OrchestratorResult:
    """Final result from multi-document orchestration."""
    query: str
    query_classification: QueryClassification
    individual_results: List[DocumentProcessingResult]
    synthesis_result: str
    total_documents: int
    total_tokens_used: int
    total_processing_time: float
    token_allocation_summary: Dict[str, Any]
    cross_document_insights: Dict[str, Any]
    processing_strategy_used: str
    error: Optional[str] = None

class DocumentProcessor:
    """Processes individual documents within allocated token budgets."""
    
    def __init__(self, smart_vs: SmartVectorStore, embedding_model: str):
        self.smart_vs = smart_vs
        self.embedding_model = embedding_model
        
        # Use the correct embedder interface
        if embedding_model == "nomic-embed-text":
            self.embedder = get_embedder(model_name=embedding_model)
        elif embedding_model in ["arctic-embed-33m", "all-minilm-l6-v2"]:
            self.embedder = get_embedder(model_name=embedding_model)
        else:
            # BGE models - use local path
            model_path = f"/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/{embedding_model}"
            self.embedder = get_embedder(local_model_path=model_path)
    
    async def process_document(self, document: Document, document_id: str, 
                             document_name: str, query: str, 
                             token_budget: int, classification: QueryClassification) -> DocumentProcessingResult:
        """
        Process a single document within its allocated token budget.
        
        Args:
            document: Document to process
            document_id: Unique document identifier
            document_name: Human-readable document name
            query: User query
            token_budget: Available tokens for this document
            classification: Query classification to guide processing
            
        Returns:
            DocumentProcessingResult with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing {document_name} with {token_budget} tokens ({classification.query_type})")
            
            # Chunk the document
            chunks = semantic_chunk(document.page_content, max_chunk_size=1000, overlap_size=200)
            
            # Create/get vector store for this document
            chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
            vectordb = self.smart_vs.get_or_create_vectorstore(
                embedder=self.embedder,
                chunks=chunks,
                document_content=document.page_content,
                embedding_model=self.embedding_model,
                chunk_params=chunk_params,
                document_info={"document_name": document_name, "document_id": document_id}
            )
            
            # Perform adaptive retrieval within token budget
            retriever = AdaptiveRetriever(embedding_model=self.embedding_model)
            
            # Calculate recommended k based on token budget and query type
            tokens_per_chunk = 200  # Conservative estimate for Llama
            overhead_tokens = 300   # Reserve for query processing
            usable_tokens = token_budget - overhead_tokens
            
            # Adjust k based on query classification
            if classification.query_type == 'SINGLE_DOCUMENT':
                # More detailed retrieval for single document queries
                max_k = min(15, usable_tokens // tokens_per_chunk)
            elif classification.query_type in ['COMPARATIVE', 'CROSS_DOCUMENT']:
                # Focused retrieval for comparison
                max_k = min(12, usable_tokens // tokens_per_chunk)
            else:
                # Balanced retrieval for thematic/aggregation
                max_k = min(10, usable_tokens // tokens_per_chunk)
            
            max_k = max(3, max_k)  # Ensure minimum viable retrieval
            
            # Get retrieval results
            results, metadata = retriever.adaptive_retrieve(vectordb, query)
            
            # Limit results to fit within token budget
            if len(results) > max_k:
                results = results[:max_k]
                metadata['final_results_count'] = len(results)
                metadata['budget_limited'] = True
            
            # Estimate tokens used for retrieval
            retrieval_tokens = len(results) * tokens_per_chunk
            
            # Prepare context for LLM based on query type
            context_prefix = self._get_context_prefix(classification, document_name)
            
            # Generate answer using cached LLM
            try:
                answer = await asyncio.to_thread(
                    synthesize_answer_cached, 
                    query, 
                    results
                )
            except Exception as e:
                logger.error(f"LLM synthesis failed for {document_name}: {e}")
                answer = f"Error generating answer for {document_name}: {str(e)}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentProcessingResult(
                document_id=document_id,
                document_name=document_name,
                query=query,
                answer=answer,
                retrieved_chunks=results,
                retrieval_metadata=metadata,
                tokens_used=retrieval_tokens,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing document {document_name}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentProcessingResult(
                document_id=document_id,
                document_name=document_name,
                query=query,
                answer="",
                retrieved_chunks=[],
                retrieval_metadata={},
                tokens_used=0,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _get_context_prefix(self, classification: QueryClassification, document_name: str) -> str:
        """Get context prefix based on query classification."""
        prefixes = {
            'SINGLE_DOCUMENT': f"Document Analysis: {document_name}",
            'COMPARATIVE': f"Document for Comparison: {document_name}",
            'CROSS_DOCUMENT': f"Document for Cross-Analysis: {document_name}",
            'THEMATIC': f"Document for Theme Analysis: {document_name}",
            'AGGREGATION': f"Document for Aggregation: {document_name}"
        }
        return prefixes.get(classification.query_type, f"Document: {document_name}")

class CrossDocumentSynthesizer:
    """Synthesizes insights across multiple documents based on query classification."""
    
    def __init__(self, token_budget: int = 1200):
        self.token_budget = token_budget
    
    def extract_cross_document_insights(self, results: List[DocumentProcessingResult], 
                                      classification: QueryClassification) -> Dict[str, Any]:
        """Extract insights that span multiple documents based on query type."""
        insights = {
            'document_count': len(results),
            'successful_documents': len([r for r in results if not r.error]),
            'failed_documents': len([r for r in results if r.error]),
            'total_chunks_retrieved': sum(len(r.retrieved_chunks) for r in results),
            'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
            'query_type': classification.query_type,
            'processing_confidence': classification.confidence
        }
        
        successful_results = [r for r in results if not r.error and r.answer]
        
        # Query-type specific analysis
        if classification.query_type == 'COMPARATIVE':
            insights.update(self._analyze_comparisons(successful_results))
        elif classification.query_type == 'CROSS_DOCUMENT':
            insights.update(self._analyze_patterns(successful_results))
        elif classification.query_type == 'THEMATIC':
            insights.update(self._analyze_themes(successful_results))
        elif classification.query_type == 'AGGREGATION':
            insights.update(self._analyze_coverage(successful_results))
        
        return insights
    
    def _analyze_comparisons(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Analyze results for comparative insights."""
        analysis = {'comparison_pairs': [], 'key_differences': [], 'similarities': []}
        
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                # Simple similarity analysis
                words1 = set(result1.answer.lower().split())
                words2 = set(result2.answer.lower().split())
                common_words = words1 & words2
                unique_words1 = words1 - words2
                unique_words2 = words2 - words1
                
                if len(common_words) > 5:
                    analysis['comparison_pairs'].append({
                        'doc1': result1.document_name,
                        'doc2': result2.document_name,
                        'common_concepts': len(common_words),
                        'unique_to_doc1': len(unique_words1),
                        'unique_to_doc2': len(unique_words2)
                    })
        
        return analysis
    
    def _analyze_patterns(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Analyze results for cross-document patterns."""
        analysis = {'recurring_patterns': [], 'contradictions': [], 'consensus_points': []}
        
        # Look for recurring keywords across documents
        all_words = []
        for result in results:
            all_words.extend(result.answer.lower().split())
        
        # Find frequently mentioned terms
        word_freq = {}
        for word in all_words:
            if len(word) > 4:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Identify patterns appearing in multiple documents
        recurring = [(word, freq) for word, freq in word_freq.items() if freq >= len(results) * 0.6]
        analysis['recurring_patterns'] = sorted(recurring, key=lambda x: x[1], reverse=True)[:10]
        
        return analysis
    
    def _analyze_themes(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Analyze results for thematic content."""
        analysis = {'major_themes': [], 'theme_frequency': {}, 'theme_distribution': []}
        
        # Simple theme extraction based on common terms
        theme_keywords = {
            'legal': ['court', 'law', 'legal', 'ruling', 'judge', 'case'],
            'financial': ['money', 'cost', 'financial', 'payment', 'economic'],
            'procedural': ['process', 'procedure', 'step', 'method', 'approach'],
            'evidence': ['evidence', 'proof', 'testimony', 'witness', 'document'],
            'analysis': ['analysis', 'evaluation', 'assessment', 'conclusion', 'finding']
        }
        
        for theme, keywords in theme_keywords.items():
            theme_count = 0
            for result in results:
                for keyword in keywords:
                    theme_count += result.answer.lower().count(keyword)
            
            if theme_count > 0:
                analysis['theme_frequency'][theme] = theme_count
        
        return analysis
    
    def _analyze_coverage(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Analyze results for aggregation coverage."""
        analysis = {'coverage_completeness': 0, 'information_density': [], 'key_points': []}
        
        # Estimate coverage completeness
        total_chunks = sum(len(r.retrieved_chunks) for r in results)
        avg_answer_length = sum(len(r.answer.split()) for r in results) / len(results) if results else 0
        
        analysis['coverage_completeness'] = min(100, (total_chunks / len(results)) * 10)
        analysis['average_response_length'] = avg_answer_length
        analysis['total_information_sources'] = total_chunks
        
        return analysis
    
    async def synthesize_cross_document_answer(self, query: str, 
                                             results: List[DocumentProcessingResult],
                                             insights: Dict[str, Any],
                                             classification: QueryClassification) -> str:
        """
        Synthesize a comprehensive answer across multiple documents based on query type.
        
        Args:
            query: Original user query
            results: Individual document processing results
            insights: Cross-document insights
            classification: Query classification
            
        Returns:
            Synthesized cross-document answer
        """
        successful_results = [r for r in results if not r.error and r.answer]
        
        if not successful_results:
            return "No successful document processing results to synthesize."
        
        # Generate synthesis based on query type
        if classification.query_type == 'SINGLE_DOCUMENT':
            return self._synthesize_single_document(query, successful_results[0])
        elif classification.query_type == 'COMPARATIVE':
            return await self._synthesize_comparative(query, successful_results, insights)
        elif classification.query_type == 'CROSS_DOCUMENT':
            return await self._synthesize_cross_document(query, successful_results, insights)
        elif classification.query_type == 'THEMATIC':
            return await self._synthesize_thematic(query, successful_results, insights)
        elif classification.query_type == 'AGGREGATION':
            return await self._synthesize_aggregation(query, successful_results, insights)
        else:
            return await self._synthesize_general(query, successful_results, insights)
    
    def _synthesize_single_document(self, query: str, result: DocumentProcessingResult) -> str:
        """Simple synthesis for single document."""
        return f"**Analysis of {result.document_name}:**\n\n{result.answer}"
    
    async def _synthesize_comparative(self, query: str, results: List[DocumentProcessingResult], 
                                    insights: Dict[str, Any]) -> str:
        """Synthesize comparative analysis."""
        synthesis_context = f"Comparative Analysis Query: {query}\n\n"
        synthesis_context += f"Documents analyzed: {len(results)}\n\n"
        
        # Include individual results with clear attribution
        for i, result in enumerate(results, 1):
            synthesis_context += f"**Document {i}: {result.document_name}**\n"
            synthesis_context += f"{result.answer}\n\n"
        
        # Add comparison insights
        if insights.get('comparison_pairs'):
            synthesis_context += "**Cross-Document Comparisons:**\n"
            for pair in insights['comparison_pairs'][:3]:
                synthesis_context += f"- {pair['doc1']} vs {pair['doc2']}: {pair['common_concepts']} shared concepts\n"
            synthesis_context += "\n"
        
        synthesis_prompt = f"""Based on the analysis above, provide a comprehensive comparative answer to: "{query}"

Focus on:
1. Key similarities between documents
2. Important differences and contrasts
3. Unique aspects of each document
4. Overall comparative insights

Comparative Analysis:"""
        
        return await self._call_llm_for_synthesis(synthesis_prompt, synthesis_context)
    
    async def _synthesize_cross_document(self, query: str, results: List[DocumentProcessingResult], 
                                       insights: Dict[str, Any]) -> str:
        """Synthesize cross-document pattern analysis."""
        synthesis_context = f"Cross-Document Analysis Query: {query}\n\n"
        
        # Include individual results
        for result in results:
            synthesis_context += f"**{result.document_name}:**\n{result.answer}\n\n"
        
        # Add pattern insights
        if insights.get('recurring_patterns'):
            synthesis_context += "**Recurring Patterns:**\n"
            for pattern, freq in insights['recurring_patterns'][:5]:
                synthesis_context += f"- '{pattern}' appears {freq} times across documents\n"
            synthesis_context += "\n"
        
        synthesis_prompt = f"""Based on the analysis above, identify patterns, contradictions, and relationships across documents for: "{query}"

Focus on:
1. Patterns that appear across multiple documents
2. Contradictions or conflicts between documents
3. Relationships and connections
4. Overarching insights

Cross-Document Analysis:"""
        
        return await self._call_llm_for_synthesis(synthesis_prompt, synthesis_context)
    
    async def _synthesize_thematic(self, query: str, results: List[DocumentProcessingResult], 
                                 insights: Dict[str, Any]) -> str:
        """Synthesize thematic analysis."""
        synthesis_context = f"Thematic Analysis Query: {query}\n\n"
        
        # Include individual results
        for result in results:
            synthesis_context += f"**{result.document_name}:**\n{result.answer}\n\n"
        
        # Add theme insights
        if insights.get('theme_frequency'):
            synthesis_context += "**Identified Themes:**\n"
            for theme, freq in sorted(insights['theme_frequency'].items(), key=lambda x: x[1], reverse=True):
                synthesis_context += f"- {theme.title()}: {freq} mentions\n"
            synthesis_context += "\n"
        
        synthesis_prompt = f"""Based on the analysis above, identify and analyze themes across documents for: "{query}"

Focus on:
1. Major themes that emerge across documents
2. How themes are developed in different documents
3. Theme relationships and connections
4. Thematic insights and conclusions

Thematic Analysis:"""
        
        return await self._call_llm_for_synthesis(synthesis_prompt, synthesis_context)
    
    async def _synthesize_aggregation(self, query: str, results: List[DocumentProcessingResult], 
                                    insights: Dict[str, Any]) -> str:
        """Synthesize aggregated information."""
        synthesis_context = f"Information Aggregation Query: {query}\n\n"
        synthesis_context += f"Aggregating information from {len(results)} documents:\n\n"
        
        # Include all results with source attribution
        for i, result in enumerate(results, 1):
            synthesis_context += f"**Source {i} ({result.document_name}):**\n"
            synthesis_context += f"{result.answer}\n\n"
        
        synthesis_prompt = f"""Based on all the information above, provide a comprehensive aggregated answer to: "{query}"

Focus on:
1. Combining information from all sources
2. Identifying the most important points
3. Providing complete coverage of the topic
4. Clear source attribution

Comprehensive Summary:"""
        
        return await self._call_llm_for_synthesis(synthesis_prompt, synthesis_context)
    
    async def _synthesize_general(self, query: str, results: List[DocumentProcessingResult], 
                                insights: Dict[str, Any]) -> str:
        """General synthesis for unclassified queries."""
        synthesis_context = f"Multi-Document Analysis Query: {query}\n\n"
        
        for result in results:
            synthesis_context += f"**{result.document_name}:**\n{result.answer}\n\n"
        
        synthesis_prompt = f"""Based on the analysis above, provide a comprehensive answer to: "{query}"

Multi-Document Analysis:"""
        
        return await self._call_llm_for_synthesis(synthesis_prompt, synthesis_context)
    
    async def _call_llm_for_synthesis(self, synthesis_prompt: str, synthesis_context: str) -> str:
        """Call LLM for synthesis within token budget."""
        try:
            # Create synthesis document
            full_context = synthesis_context + synthesis_prompt
            synthesis_doc = Document(page_content=full_context)
            synthesis_result = [(synthesis_doc, 1.0)]
            
            synthesis_answer = await asyncio.to_thread(
                synthesize_answer_cached,
                synthesis_prompt,
                synthesis_result
            )
            
            return synthesis_answer
            
        except Exception as e:
            logger.error(f"Cross-document synthesis failed: {e}")
            
            # Fallback: Simple concatenation with source attribution
            fallback_answer = f"Multi-document analysis results:\n\n"
            successful_results = [r for r in synthesis_result if hasattr(r, 'answer')]
            
            for i, result in enumerate(successful_results, 1):
                fallback_answer += f"**Document {i} ({result.document_name}):**\n"
                fallback_answer += f"{result.answer}\n\n"
            
            return fallback_answer

class DocumentOrchestrator:
    """
    Main orchestrator for multi-document processing with query classification and token awareness.
    Uses standalone query classification to optimize processing strategy.
    """
    
    def __init__(self, main_token_budget: int = 6692):
        """
        Initialize the document orchestrator with full token budget for processing.
        Query classification happens separately with its own budget.
        
        Args:
            main_token_budget: Full token budget for document processing (excluding classification)
        """
        self.token_manager = TokenBudgetManager(
            max_context_tokens=8192, 
            performance_buffer=1500
        )
        self.query_classifier = LLMQueryClassifier(dedicated_token_budget=300)
        self.smart_vs = SmartVectorStore()
        self.max_concurrent_docs = 3  # Limit concurrent processing for stability
        
        logger.info(f"DocumentOrchestrator initialized with {main_token_budget} token budget for processing")
    
    async def process_documents(self, documents: List[Document], document_names: List[str],
                              query: str, embedding_model: str = "bge-small-en") -> OrchestratorResult:
        """
        Process multiple documents with intelligent orchestration based on query classification.
        
        Args:
            documents: List of documents to process
            document_names: List of human-readable document names
            query: User query
            embedding_model: Embedding model to use
            
        Returns:
            OrchestratorResult with individual and synthesized results
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not documents:
                raise ValueError("No documents provided")
            
            if len(documents) != len(document_names):
                raise ValueError("Number of documents and names must match")
            
            logger.info(f"Starting orchestration: {len(documents)} documents, query: {query[:50]}...")
            
            # Step 1: Classify query (uses separate 300 token budget)
            classification = self.query_classifier.classify_query(query, len(documents))
            logger.info(f"Query classified as: {classification.query_type} (confidence: {classification.confidence:.2f})")
            
            # Step 2: Get processing recommendations
            recommendations = self.query_classifier.get_processing_recommendations(classification, len(documents))
            
            # Step 3: Allocate token budgets based on classification
            budgets = self._allocate_tokens_based_on_classification(documents, classification)
            
            # Step 4: Determine processing strategy
            if recommendations['use_parallel_processing'] and len(documents) > 1:
                individual_results = await self._process_documents_parallel(
                    documents, document_names, query, embedding_model, budgets, classification
                )
                strategy_used = "parallel"
            else:
                individual_results = await self._process_documents_sequential(
                    documents, document_names, query, embedding_model, budgets, classification
                )
                strategy_used = "sequential"
            
            # Step 5: Update token usage tracking
            for result in individual_results:
                if not result.error:
                    self.token_manager.use_document_tokens(result.document_id, result.tokens_used)
            
            # Step 6: Cross-document synthesis based on classification
            if len(individual_results) > 1 and recommendations['requires_cross_document_analysis']:
                synthesis_result, insights = await self._perform_intelligent_synthesis(
                    query, individual_results, classification
                )
            else:
                synthesis_result, insights = self._perform_simple_synthesis(individual_results)
            
            # Step 7: Calculate totals and return result
            total_tokens_used = self.token_manager.get_total_used_tokens()
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            return OrchestratorResult(
                query=query,
                query_classification=classification,
                individual_results=individual_results,
                synthesis_result=synthesis_result,
                total_documents=len(documents),
                total_tokens_used=total_tokens_used,
                total_processing_time=total_processing_time,
                token_allocation_summary=self.token_manager.get_allocation_summary(),
                cross_document_insights=insights,
                processing_strategy_used=strategy_used
            )
            
        except Exception as e:
            logger.error(f"Document orchestration failed: {e}")
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create fallback classification for error case
            fallback_classification = self.query_classifier._get_fallback_classification(query, len(documents) if documents else 0, "error")
            
            return OrchestratorResult(
                query=query,
                query_classification=fallback_classification,
                individual_results=[],
                synthesis_result="",
                total_documents=len(documents) if documents else 0,
                total_tokens_used=0,
                total_processing_time=total_processing_time,
                token_allocation_summary={},
                cross_document_insights={},
                processing_strategy_used="error",
                error=str(e)
            )
    
    def _allocate_tokens_based_on_classification(self, documents: List[Document], 
                                               classification: QueryClassification) -> Dict[str, Any]:
        """Allocate tokens based on query classification."""
        # Get base allocation
        budgets = self.token_manager.allocate_document_budgets(documents)
        
        # Adjust synthesis budget based on classification
        if classification.token_allocation_bias == 'synthesis_heavy':
            # Increase synthesis budget, reduce document budgets proportionally
            self.token_manager.synthesis_buffer = min(1500, self.token_manager.synthesis_buffer * 1.2)
        elif classification.token_allocation_bias == 'document_heavy':
            # Decrease synthesis budget, increase document budgets
            self.token_manager.synthesis_buffer = max(800, self.token_manager.synthesis_buffer * 0.8)
        
        # Re-allocate with adjusted synthesis budget
        budgets = self.token_manager.allocate_document_budgets(documents)
        
        return budgets
    
    async def _process_documents_parallel(self, documents: List[Document], document_names: List[str],
                                        query: str, embedding_model: str, budgets: Dict[str, Any],
                                        classification: QueryClassification) -> List[DocumentProcessingResult]:
        """Process documents in parallel."""
        logger.info(f"Processing {len(documents)} documents in parallel")
        
        # Create semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_docs)
        
        async def process_single_doc(doc, doc_name, doc_id):
            async with semaphore:
                processor = DocumentProcessor(self.smart_vs, embedding_model)
                budget = budgets[doc_id]
                return await processor.process_document(
                    doc, doc_id, doc_name, query, budget.allocated_tokens, classification
                )
        
        # Create and execute parallel tasks
        tasks = []
        for i, (doc, doc_name) in enumerate(zip(documents, document_names)):
            doc_id = f"doc_{i}"
            task = process_single_doc(doc, doc_name, doc_id)
            tasks.append(task)
        
        # Execute with error handling
        individual_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(individual_results):
            if isinstance(result, Exception):
                logger.error(f"Document {i} processing failed: {result}")
                processed_results.append(DocumentProcessingResult(
                    document_id=f"doc_{i}",
                    document_name=document_names[i],
                    query=query,
                    answer="",
                    retrieved_chunks=[],
                    retrieval_metadata={},
                    tokens_used=0,
                    processing_time=0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_documents_sequential(self, documents: List[Document], document_names: List[str],
                                          query: str, embedding_model: str, budgets: Dict[str, Any],
                                          classification: QueryClassification) -> List[DocumentProcessingResult]:
        """Process documents sequentially."""
        logger.info(f"Processing {len(documents)} documents sequentially")
        
        processor = DocumentProcessor(self.smart_vs, embedding_model)
        results = []
        
        for i, (doc, doc_name) in enumerate(zip(documents, document_names)):
            doc_id = f"doc_{i}"
            budget = budgets[doc_id]
            
            result = await processor.process_document(
                doc, doc_id, doc_name, query, budget.allocated_tokens, classification
            )
            results.append(result)
        
        return results
    
    async def _perform_intelligent_synthesis(self, query: str, results: List[DocumentProcessingResult],
                                           classification: QueryClassification) -> Tuple[str, Dict[str, Any]]:
        """Perform intelligent cross-document synthesis based on query classification."""
        synthesizer = CrossDocumentSynthesizer(
            self.token_manager.get_synthesis_budget().allocated_tokens
        )
        
        insights = synthesizer.extract_cross_document_insights(results, classification)
        synthesis_result = await synthesizer.synthesize_cross_document_answer(
            query, results, insights, classification
        )
        
        # Update synthesis token usage
        synthesis_tokens = len(synthesis_result) // 3  # Rough estimate
        self.token_manager.use_synthesis_tokens(synthesis_tokens)
        
        return synthesis_result, insights
    
    def _perform_simple_synthesis(self, results: List[DocumentProcessingResult]) -> Tuple[str, Dict[str, Any]]:
        """Perform simple synthesis for single document or basic aggregation."""
        if len(results) == 1:
            result = results[0]
            synthesis = f"**Analysis of {result.document_name}:**\n\n{result.answer}"
            insights = {'synthesis_type': 'single_document', 'document_count': 1}
        else:
            synthesis = "**Multi-Document Analysis:**\n\n"
            for i, result in enumerate(results, 1):
                if not result.error:
                    synthesis += f"**Document {i} ({result.document_name}):**\n"
                    synthesis += f"{result.answer}\n\n"
            
            insights = {
                'synthesis_type': 'simple_aggregation',
                'document_count': len(results),
                'successful_documents': len([r for r in results if not r.error])
            }
        
        return synthesis, insights
    
    def get_token_allocation_summary(self) -> Dict[str, Any]:
        """Get current token allocation summary."""
        return self.token_manager.get_allocation_summary()
    
    def check_document_feasibility(self, num_documents: int) -> Dict[str, Any]:
        """
        Check if processing the given number of documents is feasible.
        
        Args:
            num_documents: Number of documents to check
            
        Returns:
            Feasibility analysis with recommendations
        """
        min_tokens_needed = num_documents * self.token_manager.min_tokens_per_doc
        available_tokens = self.token_manager.available_for_docs
        
        return {
            'feasible': min_tokens_needed <= available_tokens,
            'documents_requested': num_documents,
            'min_tokens_needed': min_tokens_needed,
            'available_tokens': available_tokens,
            'max_feasible_documents': available_tokens // self.token_manager.min_tokens_per_doc,
            'recommended_limit': min(num_documents, 4),  # Conservative recommendation
            'performance_impact': 'high' if num_documents > 3 else 'medium' if num_documents > 1 else 'low'
        }
    
    def clear_classifier_cache(self):
        """Clear the query classifier cache."""
        self.query_classifier.clear_cache()
        logger.info("Query classifier cache cleared")