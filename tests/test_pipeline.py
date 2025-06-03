import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.loader import load_document
from backend.chunker import semantic_chunk

def test_document_pipeline(file_path):
    print("Loading document...")
    full_text = load_document(file_path)
    print(f"Total characters extracted: {len(full_text)}")

    print("\n--- Sample Extracted Text ---")
    print(full_text[:1000])  # Print a sample from the beginning

    print("\nChunking document...")
    chunks = semantic_chunk(full_text, max_chunk_size=1000)
    print(f"Total chunks generated: {len(chunks)}")

    print("\n--- First 3 Chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

if __name__ == "__main__":
    test_document_pipeline("data/in-re-andrews-inc-sec-litig.pdf")