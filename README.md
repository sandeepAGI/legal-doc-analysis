# Doc Analysis App

A local document analysis tool that uses LangChain, Ollama, and Streamlit to ingest and query long legal documents (30–150 pages). The app supports multiple embedding models and local LLM inference, providing a fast and private environment for deep document understanding.

---

## 📁 Project Structure

```
doc-analysis/
├── app.py                         # Streamlit app for interaction
├── backend/
│   ├── chunker.py                 # Semantic chunking with page & section metadata
│   ├── embedder.py               # Embedding loader with model toggle
│   ├── loader.py                 # PDF loader using unstructured
│   ├── llm_wrapper.py            # LLM prompt and response handler
│   ├── prompts.py                # Central prompt templates (static)
│   └── vectorstore.py            # Chroma-based vector DB interface
├── download_hf_bbg.py            # Script to fetch BGE models from Hugging Face
├── requirements.txt              # Full dependency list
└── README.md                     # Project documentation
```

---

## 🚀 Features Implemented

- ✅ Streamlit UI to load PDF, select embedding model, and ask questions
- ✅ Local embedding support for:
  - `bge-small-en`
  - `bge-base-en`
  - `nomic-embed-text` (via Ollama)
- ✅ Semantic chunking with page/section detection
- ✅ Metadata-aware display of chunks in response view:
  - Chunk #
  - Similarity Score
  - Section
- ✅ Clean toggle and storage separation for vector stores
- ✅ Error-handling and diagnostics for model compatibility and Chroma issues

---

## ⚠️ Critical Retrieval Issues & Next Steps

### 🔴 High Priority - Retrieval Robustness
1. ❌ **Persistent vector store management**  
   Currently creates new UUID collection each run (app.py:43), preventing reuse and wasting computation. Need document-based collection naming and reuse logic.

2. ❌ **Adaptive retrieval parameters**  
   Fixed k=10 retrieval (app.py:52) regardless of query complexity. Need dynamic k based on query type and score thresholds for quality filtering.

3. ❌ **Query preprocessing pipeline**  
   No spell checking, query expansion, or semantic preprocessing. Raw user queries often miss relevant content due to vocabulary mismatches.

4. ❌ **Semantic chunking improvements**  
   Hard character limits (chunker.py:33) break semantic units. Need overlap, sentence boundary awareness, and context preservation.

5. ❌ **Retrieval quality validation**  
   No similarity score thresholds or result filtering. Poor matches get passed to LLM, degrading answer quality.

### 🟡 Medium Priority - Advanced Features
6. ❌ **Hybrid search implementation**  
   Only vector similarity search. Adding BM25/keyword fusion would improve recall for specific terms and names.

7. ❌ **Reranking pipeline**  
   Retrieved chunks need reranking by query-specific relevance, especially for complex multi-part questions.

8. ❌ **Prompt flexibility (double-check mode)**  
   While `prompts.py` includes a double-check prompt, the toggle and logic for switching between prompts is not wired into `llm_wrapper.py` or `app.py`.

9. ❌ **Metadata-aware retrieval**  
   Page/section metadata exists but isn't used for filtering or boosting relevant document sections.

### 🟢 Low Priority - Infrastructure
10. ❌ **Support for different document types**  
    Current assumptions lean heavily on legal rulings. Chunk labeling logic may not generalize to motions, filings, or expert opinions.

11. ❌ **Query logging / batch testing**  
    No functionality yet for saving and comparing model answers across test suites.

12. ❌ **Persistent vector store pruning**  
    Vectorstore directories accumulate; a background cleaner or UI option for stale store removal is pending.

---

## 🧪 Usage Instructions

```bash
# 1. Set up environment
conda activate your_env_name

# 2. Install dependencies
pip install -r requirements.txt

# 3. (One-time) Download local BGE models
python download_hf_bbg.py

# 4. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧭 Implementation Roadmap

### Phase 1: Core Retrieval Fixes (Week 1-2)
**Priority: Fix fundamental retrieval issues**

#### 1.1 Persistent Vector Store Management
```python
# In app.py, replace UUID-based naming with document-based:
doc_hash = hashlib.md5(uploaded_file.read()).hexdigest()[:8]
collection_name = f"doc_{doc_hash}_{embed_model_option.replace('-', '_')}"
# Check if collection exists before recreating
```

#### 1.2 Adaptive Retrieval Parameters
```python
# In vectorstore.py, add smart k selection and score filtering:
def query_vectorstore(vectordb, query, min_score=0.7, max_k=15):
    results = vectordb.similarity_search_with_score(query, k=max_k)
    return [(doc, score) for doc, score in results if score >= min_score]
```

#### 1.3 Semantic Chunking Improvements
```python
# In chunker.py, add overlap and sentence boundaries:
def semantic_chunk(text, max_chunk_size=1000, overlap=200):
    # Use sentence tokenizer, preserve paragraph boundaries
    # Add sliding window overlap for context preservation
```

### Phase 2: Advanced Retrieval (Week 3-4)
**Priority: Enhance retrieval quality**

#### 2.1 Query Preprocessing
- Add spell correction using `pyspellchecker`
- Implement query expansion with synonyms
- Add legal domain-specific term normalization

#### 2.2 Hybrid Search Implementation
- Integrate BM25 keyword search alongside vector similarity
- Implement score fusion (e.g., RRF - Reciprocal Rank Fusion)
- Add metadata-based filtering and boosting

#### 2.3 Reranking Pipeline
- Add cross-encoder reranking for top-k results
- Implement query-chunk relevance scoring
- Add diversity-aware result selection

### Phase 3: User Experience (Week 5-6)
**Priority: Polish and usability**

#### 3.1 Prompt Flexibility Implementation
- Add Streamlit toggle for double-check mode
- Wire `DOUBLE_CHECK_PROMPT` into `llm_wrapper.py`
- Add confidence scoring and uncertainty handling

#### 3.2 Advanced UI Features
- Show retrieval quality metrics to users
- Add retrieval parameter controls (k, score thresholds)
- Implement query suggestion and auto-completion

#### 3.3 Evaluation and Testing
- Build automated testing suite with ground truth Q&A pairs
- Add retrieval quality metrics (MRR, NDCG, recall@k)
- Implement A/B testing framework for retrieval strategies
---

## 👤 Author Notes

Built for fast, iterative experimentation on local LLMs. Designed with legal and analytical documents in mind but flexible enough for general enterprise use.
