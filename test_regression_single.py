#!/usr/bin/env python3
"""
Single question regression test for semantic chunking improvements.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.loader import load_document
from backend.chunker import semantic_chunk, semantic_chunk_legacy
from backend.embedder import get_embedder
from backend.vectorstore import create_vectorstore, query_vectorstore
from backend.llm_wrapper import synthesize_answer

# Configuration
PDF_PATH = "data/in-re-andrews-inc-sec-litig.pdf"
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"
TEST_QUESTION = "What was the core allegation made by the plaintiff?"
MODEL_CONFIG = {"name": "bge-small-en", "path": os.path.join(MODEL_ROOT, "bge-small-en"), "ollama": False}

def run_chunker_comparison():
    """Compare old vs new chunking approach on a single question."""
    print("Loading document...")
    document_text = load_document(PDF_PATH)
    print(f"Document loaded: {len(document_text)} characters")
    
    print("\n=== Testing Legacy Chunker ===")
    legacy_chunks = semantic_chunk_legacy(document_text, max_chunk_size=1000)
    print(f"Legacy chunker produced {len(legacy_chunks)} chunks")
    
    print("\n=== Testing New Chunker ===")
    new_chunks = semantic_chunk(document_text, max_chunk_size=1000, overlap_size=200)
    print(f"New chunker produced {len(new_chunks)} chunks")
    
    # Test with embeddings and vectorstore
    print(f"\n=== Running QA Pipeline with {MODEL_CONFIG['name']} ===")
    
    try:
        # Create vectorstore with new chunks
        embedder = get_embedder(MODEL_CONFIG["path"])
        vectorstore_new = create_vectorstore(embedder, new_chunks, f"test_new_{MODEL_CONFIG['name'].replace('-', '_')}")
        
        # Query vectorstore
        relevant_chunks = query_vectorstore(vectorstore_new, TEST_QUESTION, k=5)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Generate answer
        answer = synthesize_answer(TEST_QUESTION, relevant_chunks)
        print(f"\nQuestion: {TEST_QUESTION}")
        print(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
        
        # Show chunk information
        print(f"\n=== Chunk Analysis ===")
        print(f"Average chunk size (new): {sum(len(c.page_content) for c in new_chunks) / len(new_chunks):.1f}")
        print(f"Average chunk size (legacy): {sum(len(c.page_content) for c in legacy_chunks) / len(legacy_chunks):.1f}")
        
        # Show metadata from first few chunks
        print(f"\n=== Sample Chunk Metadata (New) ===")
        for i, chunk in enumerate(new_chunks[:3]):
            print(f"Chunk {i+1}: {chunk.metadata}")
            
        print("\n✅ Regression test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during regression test: {e}")
        return False

if __name__ == "__main__":
    success = run_chunker_comparison()
    sys.exit(0 if success else 1)