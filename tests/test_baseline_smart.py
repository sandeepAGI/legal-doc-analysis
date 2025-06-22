#!/usr/bin/env python3
"""
Smart Vector Store Baseline test script that runs 8 questions on the PDF file using all embedding models
with intelligent caching for dramatically faster repeated runs.
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
from backend.llm_wrapper import synthesize_answer

# Configuration
PDF_PATH = "data/in-re-andrews-inc-sec-litig.pdf"
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"
RESULTS_FILE = "results_smart.md"

# The 8 test questions
QUESTIONS = [
    "What was the core allegation made by the plaintiff?",
    "Who were the defendants in this case?",
    "What issues did the plaintiff raise about the ARR metric?",
    "How did UiPath allegedly mislead investors during the IPO?",
    "What were the court's reasons for dismissing the Section 11 claims?",
    "What does the opinion say about Rule 10b-5 violations?",
    "When was the lawsuit filed and who was appointed lead plaintiff?",
    "What happened on September 27, 2022?"
]

# Embedding models to test
EMBEDDING_MODELS = [
    {"name": "bge-small-en", "path": os.path.join(MODEL_ROOT, "bge-small-en"), "ollama": False},
    {"name": "bge-base-en", "path": os.path.join(MODEL_ROOT, "bge-base-en"), "ollama": False},
    {"name": "nomic-embed-text", "path": None, "ollama": True}
]

def run_qa_pipeline_smart(pdf_path, question, model_config, smart_vs, full_text, chunks):
    """Run the QA pipeline using Smart Vector Store for caching."""
    print(f"Processing question with {model_config['name']}: {question[:50]}...")
    
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
    
    # Get or create vector store using Smart Vector Store (with caching)
    vectordb = smart_vs.get_or_create_vectorstore(
        embedder=embedder,
        chunks=chunks,
        document_content=full_text,
        embedding_model=model_config['name'],
        chunk_params=chunk_params,
        document_info=document_info
    )
    
    # Query vector store with k=10 to match main app behavior
    retrieved = smart_vs.query_vectorstore(vectordb, question, k=10)
    
    # Generate answer
    answer = synthesize_answer(question, retrieved)
    
    processing_time = time.time() - start_time
    
    return {
        'question': question,
        'answer': answer,
        'retrieved_chunks': len(retrieved),
        'processing_time': processing_time,
        'model': model_config['name']
    }

def append_results_to_file(results, cache_stats):
    """Append results to the results file."""
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n## Smart Vector Store Baseline Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Cache performance summary
        f.write("### 📊 Cache Performance Summary\n\n")
        f.write(f"- **Cache Hit Rate**: {cache_stats['hit_rate_percent']:.1f}%\n")
        f.write(f"- **Total Requests**: {cache_stats['total_requests']}\n")
        f.write(f"- **Cache Hits**: {cache_stats['cache_hits']}\n")
        f.write(f"- **Cache Misses**: {cache_stats['cache_misses']}\n")
        f.write(f"- **Storage Usage**: {cache_stats['storage_usage_gb']:.3f} GB\n\n")
        
        # Results by model
        current_model = None
        total_time_by_model = {}
        
        for result in results:
            if result['model'] != current_model:
                if current_model is not None:
                    f.write(f"\n**Total processing time for {current_model}: {total_time_by_model[current_model]:.2f}s**\n\n")
                
                current_model = result['model']
                total_time_by_model[current_model] = 0
                f.write(f"### Model: {current_model}\n\n")
            
            total_time_by_model[current_model] += result['processing_time']
            
            f.write(f"**Q: {result['question']}**\n")
            f.write(f"*Processing time: {result['processing_time']:.2f}s | Retrieved chunks: {result['retrieved_chunks']}*\n\n")
            f.write(f"A: {result['answer']}\n\n")
            f.write("---\n\n")
        
        # Final model total
        if current_model:
            f.write(f"**Total processing time for {current_model}: {total_time_by_model[current_model]:.2f}s**\n\n")

def main():
    """Run the complete Smart Vector Store baseline test."""
    print("🧪 Smart Vector Store Baseline Test")
    print("=" * 60)
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ ERROR: PDF file not found at {PDF_PATH}")
        return False
    
    # Initialize Smart Vector Store
    smart_vs = SmartVectorStore()
    
    print(f"📄 Loading document: {PDF_PATH}")
    full_text = load_document(PDF_PATH)
    print(f"✅ Document loaded: {len(full_text)} characters")
    
    print("🔪 Chunking document...")
    chunks = semantic_chunk(full_text, max_chunk_size=1000)
    print(f"✅ Document chunked: {len(chunks)} chunks")
    
    # Clear results file
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("# Smart Vector Store Baseline Test Results\n\n")
        f.write("This file contains results from baseline testing using Smart Vector Store with intelligent caching.\n\n")
    
    results = []
    total_start_time = time.time()
    
    # Run tests for each model and question
    for model_config in EMBEDDING_MODELS:
        print(f"\n🔢 Testing with model: {model_config['name']}")
        model_start_time = time.time()
        
        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n--- Question {i}/{len(QUESTIONS)} ---")
            
            try:
                result = run_qa_pipeline_smart(
                    PDF_PATH, question, model_config, smart_vs, full_text, chunks
                )
                results.append(result)
                
                print(f"✅ Completed in {result['processing_time']:.2f}s")
                
            except Exception as e:
                print(f"❌ ERROR processing question: {str(e)}")
                # Continue with next question
                continue
        
        model_total_time = time.time() - model_start_time
        print(f"\n✅ Model {model_config['name']} completed in {model_total_time:.2f}s")
    
    total_time = time.time() - total_start_time
    
    # Get final cache statistics
    final_cache_stats = smart_vs.get_cache_statistics()
    
    print(f"\n🎉 All tests completed in {total_time:.2f}s")
    print(f"📊 Cache Performance:")
    print(f"   Hit Rate: {final_cache_stats['hit_rate_percent']:.1f}%")
    print(f"   Total Requests: {final_cache_stats['total_requests']}")
    print(f"   Storage Usage: {final_cache_stats['storage_usage_gb']:.3f} GB")
    
    # Save results
    append_results_to_file(results, final_cache_stats)
    print(f"\n💾 Results saved to {RESULTS_FILE}")
    
    # Performance comparison info
    estimated_original_time = total_time / (final_cache_stats['hit_rate_percent'] / 100) if final_cache_stats['hit_rate_percent'] > 0 else total_time
    if final_cache_stats['cache_hits'] > 0:
        print(f"🚀 Performance: Smart Vector Store saved ~{estimated_original_time - total_time:.1f}s compared to no caching")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Smart Vector Store baseline test completed successfully!")
    else:
        print("\n❌ Smart Vector Store baseline test failed!")
    
    sys.exit(0 if success else 1)