import streamlit as st
import asyncio
import os
from typing import List
from langchain_core.documents import Document

from backend.loader import load_document
from backend.document_orchestrator import DocumentOrchestrator
from backend.llm_query_classifier import LLMQueryClassifier
from backend.smart_vectorstore import SmartVectorStore
from backend.llm_wrapper import get_query_cache_stats, cleanup_query_cache

st.set_page_config(page_title="Multi-Document Analyzer", layout="wide")
st.title("📄 Multi-Document Legal Analysis (Local LLM)")

# Base model path
MODEL_ROOT = "/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models"

# Processing mode selection
processing_mode = st.radio(
    "Select processing mode:",
    ["Single Document", "Multi-Document Analysis"],
    help="Choose single document for traditional processing or multi-document for comparative analysis"
)

# Embedding model selection
embed_model_option = st.selectbox(
    "Select embedding model:",
    [
        "arctic-embed-33m (Snowflake)",
        "all-minilm-l6-v2 (Fast)",
        "bge-small-en", 
        "bge-base-en", 
        "nomic-embed-text (Ollama)"
    ]
)

# Parse model selection
if embed_model_option == "nomic-embed-text (Ollama)":
    model_name = "nomic-embed-text"
elif embed_model_option == "arctic-embed-33m (Snowflake)":
    model_name = "arctic-embed-33m"
elif embed_model_option == "all-minilm-l6-v2 (Fast)":
    model_name = "all-minilm-l6-v2"
else:
    # BGE models (local)
    model_name = embed_model_option.split(" ")[0]  # Remove any descriptive text

# Advanced options
with st.expander("⚙️ Advanced Options"):
    enable_streaming = st.checkbox("Enable Streaming Response", value=True, 
                                 help="Stream LLM response in real-time for faster perceived response")
    
    if processing_mode == "Multi-Document Analysis":
        max_documents = st.slider("Maximum documents to process", 1, 5, 3,
                                help="Limit concurrent processing for performance")
        
        show_token_allocation = st.checkbox("Show token allocation details", value=False,
                                          help="Display detailed token budget information")

# Initialize components
@st.cache_resource
def get_orchestrator():
    return DocumentOrchestrator()

@st.cache_resource
def get_smart_vectorstore():
    return SmartVectorStore()

@st.cache_resource 
def get_query_classifier():
    return LLMQueryClassifier()

orchestrator = get_orchestrator()
smart_vs = get_smart_vectorstore()
query_classifier = get_query_classifier()

# Document upload section
if processing_mode == "Single Document":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload PDF files (up to 5)", type=["pdf"], 
                                    accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) > max_documents:
        st.warning(f"Only the first {max_documents} files will be processed for performance reasons.")
        uploaded_files = uploaded_files[:max_documents]

# Query input
query = st.text_input("Ask a question about the document(s):", 
                     placeholder="e.g., Compare the legal strategies in these documents")

# Display sidebar statistics
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
    
    # Query cache statistics
    st.subheader("🧠 Query Cache")
    query_cache_stats = get_query_cache_stats()
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Query Hits", query_cache_stats["hits"])
        st.metric("Query Misses", query_cache_stats["misses"])
    with col4:
        st.metric("Query Hit Rate", f"{query_cache_stats['hit_rate']:.1%}")
        st.metric("Total Queries", query_cache_stats["total_queries"])
    
    if st.button("🧹 Cleanup Query Cache"):
        cleanup_result = cleanup_query_cache()
        if cleanup_result['total_removed'] > 0:
            st.success(f"Removed {cleanup_result['total_removed']} cached queries")
        else:
            st.info("No cleanup needed - cache is healthy!")
    
    # Query classifier cache
    if processing_mode == "Multi-Document Analysis":
        st.subheader("🔍 Query Classification")
        classifier_stats = query_classifier.get_cache_stats()
        st.metric("Cached Classifications", classifier_stats["cached_classifications"])
        
        if st.button("🧹 Clear Classifier Cache"):
            query_classifier.clear_cache()
            st.success("Query classifier cache cleared!")

# Main processing logic
if uploaded_files and query:
    
    # Pre-processing: Query classification for multi-document mode
    if processing_mode == "Multi-Document Analysis" and len(uploaded_files) > 1:
        with st.expander("🔍 Query Analysis", expanded=True):
            with st.spinner("Analyzing query type..."):
                classification = query_classifier.classify_query(query, len(uploaded_files))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query Type", classification.query_type.replace('_', ' ').title())
                st.metric("Confidence", f"{classification.confidence:.1%}")
            with col2:
                st.metric("Processing Strategy", classification.processing_strategy.title())
                st.metric("Synthesis Type", classification.recommended_synthesis.title())
            with col3:
                st.metric("Token Allocation", classification.token_allocation_bias.replace('_', ' ').title())
            
            st.info(f"**Analysis:** {query_classifier.explain_classification(classification)}")
            
            # Get processing recommendations
            recommendations = query_classifier.get_processing_recommendations(classification, len(uploaded_files))
            
            if recommendations['requires_cross_document_analysis']:
                st.success("✅ Multi-document analysis will be performed")
            else:
                st.info("ℹ️ Individual document analysis will be performed")
    
    # Document processing
    with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
        
        # Save uploaded files and load documents
        documents = []
        document_names = []
        
        for uploaded_file in uploaded_files:
            # Save file
            pdf_path = f"data/{uploaded_file.name}"
            os.makedirs("data", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load document
            full_text = load_document(pdf_path)
            doc = Document(page_content=full_text, metadata={"filename": uploaded_file.name})
            documents.append(doc)
            document_names.append(uploaded_file.name)
        
        # Process documents using orchestrator
        if processing_mode == "Multi-Document Analysis" and len(documents) > 1:
            # Multi-document processing
            async def process_multi_documents():
                return await orchestrator.process_documents(
                    documents, document_names, query, model_name
                )
            
            # Run async processing
            result = asyncio.run(process_multi_documents())
            
            if result.error:
                st.error(f"Processing failed: {result.error}")
            else:
                # Display query classification
                st.subheader("🔍 Query Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Query Type:** {result.query_classification.query_type.replace('_', ' ').title()}")
                    st.write(f"**Processing Strategy:** {result.processing_strategy_used.title()}")
                with col2:
                    st.write(f"**Confidence:** {result.query_classification.confidence:.1%}")
                    st.write(f"**Documents Processed:** {result.total_documents}")
                
                # Display main synthesis result
                st.subheader("📋 Comprehensive Analysis")
                st.markdown(result.synthesis_result)
                
                # Display individual document results
                st.subheader("📄 Individual Document Analysis")
                for i, doc_result in enumerate(result.individual_results):
                    if doc_result.error:
                        with st.expander(f"❌ {doc_result.document_name} (Error)", expanded=False):
                            st.error(f"Error: {doc_result.error}")
                    else:
                        with st.expander(f"📄 {doc_result.document_name}", expanded=False):
                            st.markdown(doc_result.answer)
                            
                            # Show retrieval details
                            if doc_result.retrieval_metadata:
                                st.write(f"**Chunks Retrieved:** {len(doc_result.retrieved_chunks)}")
                                st.write(f"**Tokens Used:** {doc_result.tokens_used}")
                                st.write(f"**Processing Time:** {doc_result.processing_time:.2f}s")
                
                # Show token allocation if requested
                if show_token_allocation:
                    with st.expander("🧮 Token Allocation Details"):
                        st.json(result.token_allocation_summary)
                
                # Show cross-document insights
                if result.cross_document_insights:
                    with st.expander("🔗 Cross-Document Insights"):
                        insights = result.cross_document_insights
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Documents", insights.get('document_count', 0))
                            st.metric("Successful", insights.get('successful_documents', 0))
                        with col2:
                            st.metric("Total Chunks", insights.get('total_chunks_retrieved', 0))
                            st.metric("Avg Processing Time", f"{insights.get('average_processing_time', 0):.2f}s")
                        with col3:
                            st.metric("Query Type", insights.get('query_type', 'Unknown'))
                            st.metric("Processing Confidence", f"{insights.get('processing_confidence', 0):.1%}")
                        
                        # Show specific insights based on query type
                        if 'comparison_pairs' in insights and insights['comparison_pairs']:
                            st.subheader("🔄 Document Comparisons")
                            for pair in insights['comparison_pairs']:
                                st.write(f"• **{pair['doc1']}** vs **{pair['doc2']}**: {pair['common_concepts']} shared concepts")
                        
                        if 'recurring_patterns' in insights and insights['recurring_patterns']:
                            st.subheader("🔄 Recurring Patterns")
                            for pattern, freq in insights['recurring_patterns'][:5]:
                                st.write(f"• **{pattern}**: appears {freq} times")
                        
                        if 'theme_frequency' in insights and insights['theme_frequency']:
                            st.subheader("🎯 Theme Analysis")
                            for theme, freq in insights['theme_frequency'].items():
                                st.write(f"• **{theme.title()}**: {freq} mentions")
                
                # Performance summary
                with st.expander("📈 Performance Summary"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processing Time", f"{result.total_processing_time:.2f}s")
                        st.metric("Total Tokens Used", result.total_tokens_used)
                    with col2:
                        st.metric("Strategy Used", result.processing_strategy_used.title())
                        avg_time = result.total_processing_time / len(result.individual_results) if result.individual_results else 0
                        st.metric("Avg Time per Doc", f"{avg_time:.2f}s")
                    with col3:
                        successful_docs = len([r for r in result.individual_results if not r.error])
                        st.metric("Success Rate", f"{(successful_docs/len(result.individual_results)*100):.1f}%")
                        
                        if result.token_allocation_summary:
                            st.metric("Token Efficiency", f"{result.token_allocation_summary.get('utilization_of_effective', 0):.1f}%")
        
        else:
            # Single document processing (fallback to original logic)
            from backend.chunker import semantic_chunk
            from backend.embedder import get_embedder
            from backend.llm_wrapper import synthesize_answer_cached, synthesize_answer_stream_cached
            from backend.adaptive_retrieval import AdaptiveRetriever
            
            doc = documents[0]
            doc_name = document_names[0]
            
            # Load and chunk document
            chunks = semantic_chunk(doc.page_content, max_chunk_size=1000)
            
            # Initialize embedder
            embedder = get_embedder(
                use_ollama=(model_name == "nomic-embed-text"),
                model_path=None,
                model_name=model_name
            )
            
            # Chunking parameters for fingerprinting
            chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
            
            # Document info for metadata
            document_info = {
                "filename": doc_name,
                "file_size": len(uploaded_files[0].getvalue()),
                "chunks_count": len(chunks)
            }
            
            # Get or create vector store using Smart Vector Store
            vectordb = smart_vs.get_or_create_vectorstore(
                embedder=embedder,
                chunks=chunks,
                document_content=doc.page_content,
                embedding_model=model_name,
                chunk_params=chunk_params,
                document_info=document_info
            )
            
            # Query vector store with adaptive retrieval
            retrieved, retrieval_metadata = smart_vs.query_vectorstore(
                vectordb, query, embedding_model=model_name, adaptive=True
            )
            
            # Synthesize answer using LLM
            st.subheader("Answer")
            
            if enable_streaming:
                # Streaming response
                answer_container = st.empty()
                full_answer = ""
                
                for chunk in synthesize_answer_stream_cached(query, retrieved):
                    full_answer += chunk
                    answer_container.markdown(full_answer + "▊")  # Show cursor while streaming
                
                # Remove cursor after completion
                answer_container.markdown(full_answer)
            else:
                # Non-streaming response (original behavior)
                answer = synthesize_answer_cached(query, retrieved)
                st.write(answer)
            
            # Show retrieval strategy info
            if retrieval_metadata:
                with st.expander("🎯 Retrieval Strategy"):
                    retriever = AdaptiveRetriever(model_name)
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
                    content = doc.page_content
                    section = doc.metadata.get("section", "Unknown")
                    score_fmt = f"{score:.2f}" if score is not None else "N/A"
                    
                    st.markdown(f"**Chunk {i} (Score: {score_fmt}, Section: {section})**\n\n{content}")
            
            # Show cache statistics after processing
            with st.expander("📈 Performance Stats"):
                updated_stats = smart_vs.get_cache_statistics()
                st.json(updated_stats)

# Usage instructions
if not uploaded_files or not query:
    st.markdown("""
    ## 🚀 How to Use Multi-Document Analysis
    
    ### Single Document Mode
    - Upload one PDF file
    - Ask any question about the document
    - Get detailed analysis with source attribution
    
    ### Multi-Document Analysis Mode
    - Upload 2-5 PDF files for comparison
    - Ask comparative questions like:
      - *"Compare the legal strategies in these documents"*
      - *"What are the main differences between these rulings?"*
      - *"Find contradictions between these expert opinions"*
      - *"What common themes appear across all documents?"*
      - *"Summarize the key findings from all documents"*
    
    ### Features
    - **Intelligent Query Classification**: Automatically detects query type and optimizes processing
    - **Token-Aware Processing**: Manages token budgets for optimal performance within Llama 3 8K context
    - **Parallel Processing**: Processes multiple documents simultaneously when beneficial
    - **Smart Synthesis**: Provides cross-document insights based on query type
    - **Source Attribution**: Clear identification of which document provided each insight
    - **Performance Optimization**: Caching and adaptive retrieval for fast responses
    
    ### Supported Query Types
    - **Comparative**: Compare elements between documents
    - **Cross-Document**: Find patterns and contradictions across documents  
    - **Thematic**: Identify themes and topics across all documents
    - **Aggregation**: Summarize and combine information from all documents
    - **Single Document**: Traditional single document analysis
    """)