import streamlit as st
import uuid
import os
from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.vectorstore import create_vectorstore, query_vectorstore
from backend.llm_wrapper import synthesize_answer

st.set_page_config(page_title="Document Analyzer", layout="wide")
st.title("📄 Legal Document Q&A (Local LLM)")

# Base model path
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"

# Embedding model selection
embed_model_option = st.selectbox(
    "Select embedding model:",
    ["bge-small-en", "bge-base-en", "nomic-embed-text (Ollama)"]
)

use_ollama = embed_model_option == "nomic-embed-text (Ollama)"
model_path = None if use_ollama else os.path.join(MODEL_ROOT, embed_model_option)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file and query:
    with st.spinner("Processing..."):
        # Save uploaded file
        pdf_path = f"data/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and chunk document
        full_text = load_document(pdf_path)
        chunks = semantic_chunk(full_text, max_chunk_size=1000)

        # Initialize embedder
        embedder = get_embedder(local_model_path=model_path)

        # Create model-specific persist directory to avoid dimension conflicts
        model_name = embed_model_option.replace(" (Ollama)", "").replace("-", "_")
        persist_dir = f"chroma_store_{model_name}"

        # Clean up existing store to avoid stale results (like test_baseline.py)
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        # Create and populate vector store
        vectordb = create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            persist_dir=persist_dir
        )
        # Query vector store
        retrieved = query_vectorstore(vectordb, query, k=10)

        # Synthesize answer using LLM
        answer = synthesize_answer(query, retrieved)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show Retrieved Chunks"):
            for i, (doc, score) in enumerate(retrieved, start=1):
                content = doc.page_content  # ✅ direct access to just the content
                section = doc.metadata.get("section", "Unknown")
                score_fmt = f"{score:.2f}" if score is not None else "N/A"

                st.markdown(f"**Chunk {i} (Score: {score_fmt}, Section: {section})**\n\n{content}")