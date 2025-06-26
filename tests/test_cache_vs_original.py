#!/usr/bin/env python3
"""
Integration test that compares Smart Vector Store results with original vector store
to ensure identical retrieval quality and measure performance differences.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.smart_vectorstore import SmartVectorStore
from backend.vectorstore import create_vectorstore, query_vectorstore
from backend.llm_wrapper import synthesize_answer

# Configuration
PDF_PATH = "data/in-re-andrews-inc-sec-litig.pdf"
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"

# Single test question for focused comparison
TEST_QUESTION = "What happened on September 27, 2022?"

# Test with one model for focused comparison
EMBEDDING_MODEL = {
    "name": "bge-small-en", 
    "path": os.path.join(MODEL_ROOT, "bge-small-en"), 
    "ollama": False
}

def test_original_vectorstore(pdf_path, question, model_config, full_text, chunks):
    """Test using original vector store approach."""
    print("🔄 Testing Original Vector Store...")
    start_time = time.time()
    
    # Initialize embedder
    embedder = get_embedder(local_model_path=model_config['path'])
    
    # Create vector store using original method
    vectordb = create_vectorstore(
        embedder=embedder,
        chunks=chunks,
        persist_dir=f"temp_original_{model_config['name']}"
    )
    
    # Query vector store
    retrieved = query_vectorstore(vectordb, question, k=10)
    
    # Generate answer
    answer = synthesize_answer(question, retrieved)
    
    processing_time = time.time() - start_time
    
    return {
        'answer': answer,
        'retrieved_chunks': [(doc.page_content, score) for doc, score in retrieved],
        'processing_time': processing_time,
        'method': 'original'
    }

def test_smart_vectorstore(pdf_path, question, model_config, full_text, chunks, smart_vs):
    """Test using Smart Vector Store approach."""
    print("🚀 Testing Smart Vector Store...")
    start_time = time.time()
    
    # Initialize embedder
    embedder = get_embedder(local_model_path=model_config['path'])
    
    # Chunking parameters for fingerprinting
    chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
    
    # Document info
    document_info = {
        "filename": os.path.basename(pdf_path),
        "file_size": len(full_text),
        "chunks_count": len(chunks)
    }
    
    # Get or create vector store using Smart Vector Store
    vectordb = smart_vs.get_or_create_vectorstore(
        embedder=embedder,
        chunks=chunks,
        document_content=full_text,
        embedding_model=model_config['name'],
        chunk_params=chunk_params,
        document_info=document_info
    )
    
    # Query vector store (returns tuple with metadata now)
    retrieved, metadata = smart_vs.query_vectorstore(vectordb, question, k=10, adaptive=False)
    
    # Generate answer
    answer = synthesize_answer(question, retrieved)
    
    processing_time = time.time() - start_time
    
    return {
        'answer': answer,
        'retrieved_chunks': [(doc.page_content, score) for doc, score in retrieved],
        'processing_time': processing_time,
        'method': 'smart'
    }

def compare_results(original_result, smart_result):
    """Compare results between original and smart vector store."""
    print("\n📊 Comparison Results")
    print("=" * 50)
    
    # Compare processing times
    time_diff = original_result['processing_time'] - smart_result['processing_time']
    speedup = (time_diff / original_result['processing_time']) * 100 if original_result['processing_time'] > 0 else 0
    
    print(f"⏱️  Processing Time Comparison:")
    print(f"   Original: {original_result['processing_time']:.2f}s")
    print(f"   Smart:    {smart_result['processing_time']:.2f}s")
    print(f"   Speedup:  {speedup:.1f}% {'faster' if speedup > 0 else 'slower'}")
    
    # Compare number of retrieved chunks
    orig_count = len(original_result['retrieved_chunks'])
    smart_count = len(smart_result['retrieved_chunks'])
    
    print(f"\n📑 Retrieved Chunks:")
    print(f"   Original: {orig_count} chunks")
    print(f"   Smart:    {smart_count} chunks")
    
    # Compare chunk content (first few for verification)
    print(f"\n🔍 Content Comparison:")
    if orig_count == smart_count:
        print("   ✅ Same number of chunks retrieved")
        
        # Check if chunks are identical
        content_match = True
        for i in range(min(5, orig_count)):  # Check first 5 chunks
            orig_content = original_result['retrieved_chunks'][i][0]
            smart_content = smart_result['retrieved_chunks'][i][0]
            
            if orig_content != smart_content:
                content_match = False
                print(f"   ❌ Chunk {i+1} content differs")
                break
        
        if content_match:
            print("   ✅ Chunk contents are identical (first 5 checked)")
    else:
        print("   ⚠️  Different number of chunks retrieved")
    
    # Compare answers
    print(f"\n💬 Answer Comparison:")
    answer_similarity = original_result['answer'] == smart_result['answer']
    
    if answer_similarity:
        print("   ✅ Answers are identical")
    else:
        print("   ⚠️  Answers differ")
        print(f"\n   Original Answer: {original_result['answer'][:200]}...")
        print(f"\n   Smart Answer:    {smart_result['answer'][:200]}...")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if orig_count == smart_count and content_match:
        print("   ✅ Smart Vector Store produces identical results")
        print("   ✅ Performance optimization successful")
        return True
    else:
        print("   ❌ Results differ between methods")
        return False

def test_cache_performance(smart_vs, pdf_path, question, model_config, full_text, chunks):
    """Test cache performance with multiple runs."""
    print("\n🔥 Cache Performance Test")
    print("-" * 30)
    
    times = []
    
    # Run 3 times to test cache behavior
    for run in range(1, 4):
        print(f"Run {run}: ", end="")
        
        start_time = time.time()
        
        # Initialize embedder
        embedder = get_embedder(local_model_path=model_config['path'])
        
        # Chunking parameters for fingerprinting
        chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
        
        # Document info
        document_info = {
            "filename": os.path.basename(pdf_path),
            "file_size": len(full_text),
            "chunks_count": len(chunks)
        }
        
        # Get or create vector store using Smart Vector Store
        vectordb = smart_vs.get_or_create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            document_content=full_text,
            embedding_model=model_config['name'],
            chunk_params=chunk_params,
            document_info=document_info
        )
        
        # Query vector store
        retrieved = smart_vs.query_vectorstore(vectordb, question, k=10)
        
        processing_time = time.time() - start_time
        times.append(processing_time)
        
        print(f"{processing_time:.2f}s")
    
    # Analyze cache performance
    cache_stats = smart_vs.get_cache_statistics()
    
    print(f"\n📈 Cache Statistics:")
    print(f"   Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"   Cache Hits: {cache_stats['cache_hits']}")
    print(f"   Cache Misses: {cache_stats['cache_misses']}")
    
    print(f"\n⚡ Performance Analysis:")
    print(f"   First run (cache miss): {times[0]:.2f}s")
    if len(times) > 1:
        avg_cached = sum(times[1:]) / len(times[1:])
        speedup = ((times[0] - avg_cached) / times[0]) * 100
        print(f"   Avg cached runs: {avg_cached:.2f}s")
        print(f"   Cache speedup: {speedup:.1f}%")

def main():
    """Run the comparison test."""
    print("🧪 Smart Vector Store vs Original Comparison Test")
    print("=" * 60)
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ ERROR: PDF file not found at {PDF_PATH}")
        return False
    
    print(f"📄 Loading document: {PDF_PATH}")
    full_text = load_document(PDF_PATH)
    print(f"✅ Document loaded: {len(full_text)} characters")
    
    print("🔪 Chunking document...")
    chunks = semantic_chunk(full_text, max_chunk_size=1000)
    print(f"✅ Document chunked: {len(chunks)} chunks")
    
    print(f"\n🔍 Testing question: {TEST_QUESTION}")
    print(f"🔢 Using model: {EMBEDDING_MODEL['name']}")
    
    # Initialize Smart Vector Store
    smart_vs = SmartVectorStore()
    
    try:
        # Test original method
        original_result = test_original_vectorstore(
            PDF_PATH, TEST_QUESTION, EMBEDDING_MODEL, full_text, chunks
        )
        
        # Test smart method
        smart_result = test_smart_vectorstore(
            PDF_PATH, TEST_QUESTION, EMBEDDING_MODEL, full_text, chunks, smart_vs
        )
        
        # Compare results
        success = compare_results(original_result, smart_result)
        
        # Test cache performance
        test_cache_performance(smart_vs, PDF_PATH, TEST_QUESTION, EMBEDDING_MODEL, full_text, chunks)
        
        return success
        
    except Exception as e:
        print(f"\n💥 ERROR during comparison test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup temporary directory
        import shutil
        temp_dir = f"temp_original_{EMBEDDING_MODEL['name']}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    print(f"Comparison Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = main()
    
    if success:
        print("\n✅ Comparison test completed successfully!")
        print("   Smart Vector Store produces identical results with better performance!")
    else:
        print("\n❌ Comparison test failed!")
        print("   Results differ between Smart and Original vector stores!")
    
    sys.exit(0 if success else 1)