#!/usr/bin/env python3
"""
Baseline test script that runs 8 questions on the PDF file using all embedding models
and stores results in a markdown file.
"""

import os
import uuid
import shutil
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.vectorstore import create_vectorstore, query_vectorstore
from backend.llm_wrapper import synthesize_answer

# Configuration
PDF_PATH = "data/in-re-andrews-inc-sec-litig.pdf"
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"
RESULTS_FILE = "results.md"

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

def run_qa_pipeline(pdf_path, question, model_config):
    """Run the complete QA pipeline for a single question and model."""
    print(f"Processing question with {model_config['name']}: {question[:50]}...")
    
    # Load and chunk document
    full_text = load_document(pdf_path)
    chunks = semantic_chunk(full_text, max_chunk_size=1000)
    
    # Initialize embedder
    embedder = get_embedder(local_model_path=model_config['path'])
    
    # Create unique persist directory for each model to avoid dimension conflicts
    persist_dir = f"chroma_store_{model_config['name']}"
    
    # Clean up existing store to avoid dimension mismatches
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    # Generate unique collection name to avoid stale results (following app.py pattern)
    collection_name = f"doc_chunks_{uuid.uuid4().hex[:8]}"
    
    # Create vector store
    vectordb = create_vectorstore(
        embedder=embedder,
        chunks=chunks,
        persist_dir=persist_dir
    )
    
    # Query vector store with k=10 to match main app behavior
    retrieved = query_vectorstore(vectordb, question, k=10)
    
    # Synthesize answer
    answer = synthesize_answer(question, retrieved)
    
    return answer, retrieved

def write_results_header(file_handle, timestamp):
    """Write the results header to the markdown file."""
    file_handle.write(f"# Baseline Test Results - {timestamp}\n\n")
    file_handle.write(f"**Test Run Date:** {timestamp}\n")
    file_handle.write(f"**PDF File:** {PDF_PATH}\n")
    file_handle.write(f"**Number of Questions:** {len(QUESTIONS)}\n")
    file_handle.write(f"**Embedding Models Tested:** {len(EMBEDDING_MODELS)}\n\n")

def write_model_results(file_handle, model_name, questions_and_answers):
    """Write results for a specific model to the markdown file."""
    file_handle.write(f"## Results for {model_name}\n\n")
    
    for i, (question, answer, retrieved_count) in enumerate(questions_and_answers, 1):
        file_handle.write(f"### Question {i}\n")
        file_handle.write(f"**Q:** {question}\n\n")
        file_handle.write(f"**A:** {answer}\n\n")
        file_handle.write(f"*Retrieved chunks: {retrieved_count}*\n\n")
        file_handle.write("---\n\n")

def main():
    """Main test execution function."""
    print(f"Starting baseline test with {len(EMBEDDING_MODELS)} embedding models...")
    print(f"PDF file: {PDF_PATH}")
    print(f"Questions: {len(QUESTIONS)}")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Rotate results file if it gets too large (>1MB)
    MAX_SIZE_MB = 1
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > MAX_SIZE_MB * 1024 * 1024:
        backup_name = f"results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        shutil.move(RESULTS_FILE, backup_name)
        print(f"📦 Rotated large results file to: {backup_name}")
    
    # Open results file for writing (append mode)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        # Write separator if file already exists and has content
        if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
            f.write("\n\n" + "="*80 + "\n\n")
        
        write_results_header(f, timestamp)
        
        # Test each embedding model
        for model_config in EMBEDDING_MODELS:
            print(f"\n{'='*60}")
            print(f"Testing with {model_config['name']} embedding model")
            print(f"{'='*60}")
            
            questions_and_answers = []
            
            # Run each question
            for i, question in enumerate(QUESTIONS, 1):
                print(f"\nQuestion {i}/{len(QUESTIONS)}: {question}")
                
                try:
                    answer, retrieved = run_qa_pipeline(PDF_PATH, question, model_config)
                    questions_and_answers.append((question, answer, len(retrieved)))
                    print(f"✓ Answer generated ({len(retrieved)} chunks retrieved)")
                    
                except Exception as e:
                    error_msg = f"Error processing question: {str(e)}"
                    print(f"✗ {error_msg}")
                    questions_and_answers.append((question, error_msg, 0))
            
            # Write results for this model
            write_model_results(f, model_config['name'], questions_and_answers)
    
    print(f"\n{'='*60}")
    print(f"Baseline test completed!")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()