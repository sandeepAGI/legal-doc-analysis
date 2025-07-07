# tests/test_baseline_orchestrator.py

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_orchestrator import DocumentOrchestrator
from backend.llm_query_classifier import LLMQueryClassifier
from backend.loader import load_document
from langchain_core.documents import Document

# Test configuration
EMBEDDING_MODELS = [
    "arctic-embed-33m",
    "all-minilm-l6-v2", 
    "bge-small-en",
    "bge-base-en"
    # Skip nomic-embed-text for baseline to avoid Ollama dependency
]

# Multi-document test queries covering different classification types
MULTI_DOC_QUERIES = [
    # Comparative queries
    ("Compare the legal strategies used in these documents", "COMPARATIVE"),
    ("What are the main differences between these rulings?", "COMPARATIVE"),
    
    # Cross-document queries  
    ("Find contradictions or conflicts between these documents", "CROSS_DOCUMENT"),
    ("What patterns appear across all these legal documents?", "CROSS_DOCUMENT"),
    
    # Thematic queries
    ("What are the common themes across all documents?", "THEMATIC"),
    ("Identify recurring legal concepts in these documents", "THEMATIC"),
    
    # Aggregation queries
    ("Summarize the key findings from all documents", "AGGREGATION"),
    ("What are the most important points across all documents?", "AGGREGATION")
]

# Single document queries for comparison
SINGLE_DOC_QUERIES = [
    ("What is the main legal argument presented?", "SINGLE_DOCUMENT"),
    ("Who are the key parties involved?", "SINGLE_DOCUMENT"),
    ("What is the court's ruling?", "SINGLE_DOCUMENT")
]

def get_test_documents():
    """Load test documents from the data directory."""
    data_dir = Path("data")
    documents = []
    document_names = []
    
    # Look for PDF files in the data directory
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        print(f"Warning: Only {len(pdf_files)} PDF files found in data/. Multi-document tests need at least 2.")
        print("Please add more PDF files to the data/ directory for comprehensive testing.")
    
    # Load up to 3 documents for testing (to stay within token limits)
    for pdf_file in pdf_files[:3]:
        try:
            print(f"Loading document: {pdf_file.name}")
            full_text = load_document(str(pdf_file))
            doc = Document(page_content=full_text, metadata={"filename": pdf_file.name})
            documents.append(doc)
            document_names.append(pdf_file.name)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    return documents, document_names

async def test_orchestrator_with_documents(documents, document_names, queries, embedding_model):
    """Test the orchestrator with multiple documents and queries."""
    print(f"\n=== Testing {embedding_model} with {len(documents)} documents ===")
    
    orchestrator = DocumentOrchestrator()
    results = []
    
    for query, expected_type in queries:
        print(f"\nQuery: {query}")
        print(f"Expected Type: {expected_type}")
        
        start_time = time.time()
        
        try:
            # Process documents using orchestrator
            result = await orchestrator.process_documents(
                documents, document_names, query, embedding_model
            )
            
            processing_time = time.time() - start_time
            
            # Collect result metrics
            test_result = {
                'embedding_model': embedding_model,
                'query': query,
                'expected_type': expected_type,
                'actual_type': result.query_classification.query_type,
                'classification_confidence': result.query_classification.confidence,
                'processing_strategy': result.processing_strategy_used,
                'total_documents': result.total_documents,
                'successful_documents': len([r for r in result.individual_results if not r.error]),
                'total_tokens_used': result.total_tokens_used,
                'processing_time': processing_time,
                'synthesis_length': len(result.synthesis_result) if result.synthesis_result else 0,
                'error': result.error
            }
            
            results.append(test_result)
            
            # Print summary
            print(f"  Classified as: {result.query_classification.query_type} (confidence: {result.query_classification.confidence:.2f})")
            print(f"  Strategy: {result.processing_strategy_used}")
            print(f"  Tokens used: {result.total_tokens_used}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Success rate: {test_result['successful_documents']}/{test_result['total_documents']}")
            
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Synthesis length: {test_result['synthesis_length']} chars")
                
                # Show first 100 chars of synthesis
                synthesis_preview = result.synthesis_result[:100] + "..." if len(result.synthesis_result) > 100 else result.synthesis_result
                print(f"  Synthesis preview: {synthesis_preview}")
        
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({
                'embedding_model': embedding_model,
                'query': query,
                'expected_type': expected_type,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
    
    return results

async def test_query_classifier_standalone():
    """Test the standalone query classifier."""
    print("\n=== Testing Standalone Query Classifier ===")
    
    classifier = LLMQueryClassifier()
    
    all_queries = MULTI_DOC_QUERIES + SINGLE_DOC_QUERIES
    
    for query, expected_type in all_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected_type}")
        
        # Test with different document counts
        for num_docs in [1, 2, 3]:
            classification = classifier.classify_query(query, num_docs, use_cache=False)
            
            print(f"  {num_docs} docs -> {classification.query_type} (conf: {classification.confidence:.2f}, strategy: {classification.processing_strategy})")
            
            # Check if classification makes sense
            if num_docs == 1 and classification.query_type != "SINGLE_DOCUMENT":
                print(f"    WARNING: Single doc should be SINGLE_DOCUMENT, got {classification.query_type}")
            
            if num_docs > 1 and "compare" in query.lower() and classification.query_type != "COMPARATIVE":
                print(f"    WARNING: Comparative query should be COMPARATIVE, got {classification.query_type}")

def test_token_budget_feasibility():
    """Test token budget feasibility for different document counts."""
    print("\n=== Testing Token Budget Feasibility ===")
    
    orchestrator = DocumentOrchestrator()
    
    for num_docs in range(1, 6):
        feasibility = orchestrator.check_document_feasibility(num_docs)
        
        print(f"{num_docs} documents:")
        print(f"  Feasible: {feasibility['feasible']}")
        print(f"  Tokens needed: {feasibility['min_tokens_needed']}")
        print(f"  Tokens available: {feasibility['available_tokens']}")
        print(f"  Max feasible: {feasibility['max_feasible_documents']}")
        print(f"  Performance impact: {feasibility['performance_impact']}")

async def run_comprehensive_baseline():
    """Run comprehensive baseline tests for the orchestrator."""
    print("🚀 Starting Multi-Document Orchestrator Baseline Tests")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Test token budget feasibility
    test_token_budget_feasibility()
    
    # Test standalone query classifier
    await test_query_classifier_standalone()
    
    # Load test documents
    documents, document_names = get_test_documents()
    
    if len(documents) < 2:
        print("\n❌ Insufficient documents for multi-document testing")
        print("Please add at least 2 PDF files to the data/ directory")
        return
    
    print(f"\n📄 Loaded {len(documents)} documents: {', '.join(document_names)}")
    
    # Test with different numbers of documents
    all_results = []
    
    for embedding_model in EMBEDDING_MODELS:
        try:
            print(f"\n🔄 Testing embedding model: {embedding_model}")
            
            # Test single document
            if documents:
                single_results = await test_orchestrator_with_documents(
                    [documents[0]], [document_names[0]], SINGLE_DOC_QUERIES, embedding_model
                )
                all_results.extend(single_results)
            
            # Test multi-document (use first 2-3 documents)
            if len(documents) >= 2:
                multi_results = await test_orchestrator_with_documents(
                    documents[:3], document_names[:3], MULTI_DOC_QUERIES, embedding_model
                )
                all_results.extend(multi_results)
                
        except Exception as e:
            print(f"❌ Error testing {embedding_model}: {e}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📊 BASELINE TEST SUMMARY")
    print("=" * 80)
    
    # Group results by model
    for model in EMBEDDING_MODELS:
        model_results = [r for r in all_results if r.get('embedding_model') == model]
        if not model_results:
            continue
            
        print(f"\n{model}:")
        
        successful = [r for r in model_results if not r.get('error')]
        failed = [r for r in model_results if r.get('error')]
        
        print(f"  Success rate: {len(successful)}/{len(model_results)} ({len(successful)/len(model_results)*100:.1f}%)")
        
        if successful:
            avg_time = sum(r['processing_time'] for r in successful) / len(successful)
            avg_tokens = sum(r.get('total_tokens_used', 0) for r in successful) / len(successful)
            
            print(f"  Average processing time: {avg_time:.2f}s")
            print(f"  Average tokens used: {avg_tokens:.0f}")
            
            # Classification accuracy
            correct_classifications = [r for r in successful 
                                    if r.get('actual_type') == r.get('expected_type')]
            if successful:
                accuracy = len(correct_classifications) / len(successful) * 100
                print(f"  Classification accuracy: {accuracy:.1f}%")
        
        if failed:
            print(f"  Failed queries: {len(failed)}")
            for failure in failed[:3]:  # Show first 3 failures
                print(f"    - {failure.get('query', 'Unknown')[:50]}...")
    
    # Overall statistics
    successful_total = [r for r in all_results if not r.get('error')]
    
    if successful_total:
        print(f"\n📈 Overall Statistics:")
        print(f"Total successful tests: {len(successful_total)}")
        print(f"Average processing time: {sum(r['processing_time'] for r in successful_total) / len(successful_total):.2f}s")
        
        token_results = [r for r in successful_total if 'total_tokens_used' in r]
        if token_results:
            print(f"Average tokens used: {sum(r['total_tokens_used'] for r in token_results) / len(token_results):.0f}")
        
        # Query type distribution
        type_counts = {}
        for result in successful_total:
            actual_type = result.get('actual_type', 'Unknown')
            type_counts[actual_type] = type_counts.get(actual_type, 0) + 1
        
        print("\nQuery type distribution:")
        for query_type, count in sorted(type_counts.items()):
            print(f"  {query_type}: {count}")
    
    print(f"\n✅ Baseline testing completed at {datetime.now().isoformat()}")
    
    return all_results

if __name__ == "__main__":
    # Run the comprehensive baseline tests
    results = asyncio.run(run_comprehensive_baseline())
    
    # Save results to file
    import json
    
    # Prepare results for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            # Convert any non-serializable values
            if isinstance(value, (str, int, float, bool, type(None))):
                json_result[key] = value
            else:
                json_result[key] = str(value)
        json_results.append(json_result)
    
    results_file = f"results_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'multi_document_orchestrator_baseline',
            'embedding_models': EMBEDDING_MODELS,
            'total_tests': len(json_results),
            'results': json_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")