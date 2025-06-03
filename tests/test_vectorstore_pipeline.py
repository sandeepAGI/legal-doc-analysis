import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.vectorstore import create_vectorstore, add_chunks_to_store, query_vectorstore

def run_vectorstore_pipeline(pdf_path, persist_dir="chroma_store"):
    print("Loading document...")
    full_text = load_document(pdf_path)

    print("Chunking document...")
    chunks = semantic_chunk(full_text, max_chunk_size=1000)
    print(f"Generated {len(chunks)} chunks.")

    print("Initializing embedder...")
    embedder = get_embedder()

    print("Creating vector store...")
    vectordb = create_vectorstore(persist_dir, embedder)

    print("Adding chunks to vector store...")
    add_chunks_to_store(chunks, vectordb)

    print("Running test query...")
    query = "Who were the defendants?"
    results = query_vectorstore(query, vectordb, k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)

if __name__ == "__main__":
    run_vectorstore_pipeline("data/in-re-andrews-inc-sec-litig.pdf")