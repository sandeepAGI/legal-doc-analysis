import streamlit as st
import uuid
import os
from backend.loader import load_document
from backend.chunker import semantic_chunk
from backend.embedder import get_embedder
from backend.smart_vectorstore import SmartVectorStore
from backend.llm_wrapper import synthesize_answer, synthesize_answer_stream

st.set_page_config(page_title="Document Analyzer", layout="wide")
st.title("📄 Legal Document Q&A (Local LLM)")

# Base model path
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"

# Embedding model selection
embed_model_option = st.selectbox(
    "Select embedding model:",
    ["bge-small-en", "bge-base-en", "nomic-embed-text (Ollama)"]
)

# Streaming toggle
enable_streaming = st.checkbox("Enable Streaming Response", value=True, help="Stream LLM response in real-time for faster perceived response")

use_ollama = embed_model_option == "nomic-embed-text (Ollama)"
model_path = None if use_ollama else os.path.join(MODEL_ROOT, embed_model_option)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document:")

# Initialize Smart Vector Store
@st.cache_resource
def get_smart_vectorstore():
    return SmartVectorStore()

smart_vs = get_smart_vectorstore()

# Display cache statistics in sidebar
with st.sidebar:
    st.subheader("📊 Cache Statistics")
    cache_stats = smart_vs.get_cache_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cache Hits", cache_stats["cache_hits"])
        st.metric("Cache Misses", cache_stats["cache_misses"])
    with col2:
        st.metric("Hit Rate", f"{cache_stats['hit_rate_percent']:.1f}%")
        st.metric("Storage", f"{cache_stats['storage_usage_gb']:.2f} GB")
    
    if st.button("🧹 Force Cleanup"):
        smart_vs.force_cleanup()
        st.success("Cache cleanup completed!")

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

        # Get embedding model name for fingerprinting
        embedding_model = embed_model_option.replace(" (Ollama)", "")
        
        # Chunking parameters for fingerprinting
        chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
        
        # Document info for metadata
        document_info = {
            "filename": uploaded_file.name,
            "file_size": len(uploaded_file.getvalue()),
            "chunks_count": len(chunks)
        }

        # Get or create vector store using Smart Vector Store
        vectordb = smart_vs.get_or_create_vectorstore(
            embedder=embedder,
            chunks=chunks,
            document_content=full_text,
            embedding_model=embedding_model,
            chunk_params=chunk_params,
            document_info=document_info
        )
        
        # Query vector store with adaptive retrieval
        retrieved, retrieval_metadata = smart_vs.query_vectorstore(
            vectordb, query, embedding_model=embedding_model, adaptive=True
        )

        # Synthesize answer using LLM
        st.subheader("Answer")
        
        if enable_streaming:
            # Streaming response
            answer_container = st.empty()
            full_answer = ""
            
            for chunk in synthesize_answer_stream(query, retrieved):
                full_answer += chunk
                answer_container.markdown(full_answer + "▊")  # Show cursor while streaming
            
            # Remove cursor after completion
            answer_container.markdown(full_answer)
        else:
            # Non-streaming response (original behavior)
            answer = synthesize_answer(query, retrieved)
            st.write(answer)
        
        # Show retrieval strategy info
        if retrieval_metadata:
            with st.expander("🎯 Retrieval Strategy"):
                from backend.adaptive_retrieval import AdaptiveRetriever
                retriever = AdaptiveRetriever(embedding_model)
                explanation = retriever.get_retrieval_explanation(retrieval_metadata)
                st.info(explanation)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Type", retrieval_metadata['query_complexity'].title())
                    st.metric("Results Retrieved", retrieval_metadata['final_results_count'])
                with col2:
                    st.metric("Search Pool", retrieval_metadata['raw_results_count'])
                    st.metric("After Filtering", retrieval_metadata['filtered_results_count'])
                with col3:
                    st.metric("Est. Tokens Used", retrieval_metadata['estimated_tokens'])
                    threshold = f"{retrieval_metadata['quality_threshold']:.3f}"
                    st.metric("Quality Threshold", threshold)

        with st.expander("Show Retrieved Chunks"):
            for i, (doc, score) in enumerate(retrieved, start=1):
                content = doc.page_content  # ✅ direct access to just the content
                section = doc.metadata.get("section", "Unknown")
                score_fmt = f"{score:.2f}" if score is not None else "N/A"

                st.markdown(f"**Chunk {i} (Score: {score_fmt}, Section: {section})**\n\n{content}")
        
        # Show cache statistics after processing
        with st.expander("📈 Performance Stats"):
            updated_stats = smart_vs.get_cache_statistics()
            st.json(updated_stats)