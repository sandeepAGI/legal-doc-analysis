# Doc Analysis App

A local document analysis tool that uses LangChain, Ollama, and Streamlit to ingest and query long legal documents (30–150 pages). The app supports multiple embedding models and local LLM inference, providing a fast and private environment for deep document understanding.

This code has been developed using Claude Code and is part of my quest to understand what is possible using LLMs today

Disclaimer:  I am not a professional programmer and only had rudimentary knowledge of Python before I started this journey - everything here has been learnt through experiential learing.

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
│   ├── smart_vectorstore.py      # Smart Vector Store with caching & fingerprinting
│   └── vectorstore.py            # Original Chroma-based vector DB interface
├── tests/
│   ├── test_baseline.py          # Baseline test suite for embedding model comparison
│   ├── test_baseline_smart.py    # Smart Vector Store baseline with caching
│   ├── test_cache_vs_original.py # Smart vs Original comparison test
│   ├── test_chunker.py           # Comprehensive chunking unit tests
│   ├── test_pipeline.py          # Pipeline integration tests
│   ├── test_smart_vectorstore.py # Smart Vector Store unit tests
│   └── test_vectorstore_pipeline.py # Vector store specific tests
├── download_hf_bbg.py            # Script to fetch BGE models from Hugging Face
├── requirements.txt              # Full dependency list
├── results.md                    # Test results from baseline comparisons
└── README.md                     # Project documentation
```

---

## 🚀 Features Implemented

- ✅ Streamlit UI to load PDF, select embedding model, and ask questions
- ✅ Local embedding support for:
  - `bge-small-en`
  - `bge-base-en`
  - `nomic-embed-text` (via Ollama)
- ✅ **Enhanced semantic chunking** with sentence boundaries, sliding window overlap, and page/section detection
- ✅ Metadata-aware display of chunks in response view:
  - Chunk #
  - Similarity Score
  - Section
- ✅ Model-specific vector store separation to prevent dimension conflicts
- ✅ **Smart Vector Store Management** with intelligent caching and storage optimization:
  - Document fingerprinting (MD5 hash of content + model + params)
  - Automatic cache hits for unchanged documents (~90% faster processing)
  - LRU-based storage management with configurable limits
  - Real-time cache statistics in Streamlit sidebar
  - Automatic cleanup and storage monitoring
- ✅ Error-handling and diagnostics for model compatibility and Chroma issues

---

## ⚠️ Critical Retrieval Issues & Implementation Solutions

### 🔴 High Priority - Retrieval Robustness

#### 1. **Vector Store Stale Data Issue** ✅
**Status**: **FIXED** (June 19th, 2025)

**Issue**: Streamlit app was reusing stale vector stores from previous document uploads, causing inconsistent retrieval results compared to fresh baseline tests.

**Root Cause**: `app.py` persisted vector stores across sessions while `test_baseline.py` created fresh ones each time, leading to quality differences.

**Solution Implemented**:
- ✅ **Vector Store Cleanup**: Added automatic cleanup of existing vector stores before creating new ones
- ✅ **Consistent Behavior**: Streamlit app now matches baseline test logic exactly
- ✅ **Fresh Embeddings**: Each document upload creates a clean vector store, preventing stale retrieval

**Implementation**:
```python
# In app.py lines 46-49
# Clean up existing store to avoid stale results (like test_baseline.py)
import shutil
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
```

**Performance Impact**: Slightly longer processing time per document, but ensures retrieval accuracy and consistency with baseline tests.

**Validation**: Q8 test case now correctly finds "September 27, 2022" information in both Streamlit and baseline tests.

#### 2. **Smart Vector Store Management** ✅
**Status**: **IMPLEMENTED** (June 22nd, 2025)

**Implementation Completed**:
- ✅ **Document Fingerprinting**: MD5 hash of (document_content + embedding_model + chunk_params) for unique identification
- ✅ **Intelligent Caching**: Automatic reuse of existing vector stores when document unchanged (~90% faster processing)
- ✅ **Storage Management**: LRU-based cleanup with configurable limits (1GB total, 10 collections per model, 30-day TTL)
- ✅ **Real-time Statistics**: Cache hit/miss rates and storage usage displayed in Streamlit sidebar
- ✅ **Future-Ready Design**: Infrastructure supports document comparison capabilities

**Key Benefits**:
```python
# Smart Vector Store automatically handles caching
smart_vs = SmartVectorStore()
vectordb = smart_vs.get_or_create_vectorstore(
    embedder=embedder,
    chunks=chunks,
    document_content=full_text,
    embedding_model=model_name,
    chunk_params={"max_chunk_size": 1000, "overlap_size": 200}
)
```

**Performance Results**:
- **Cache Hit Speed**: ~90% faster processing for unchanged documents
- **Storage Management**: Automatic LRU cleanup prevents unlimited growth
- **Memory Efficiency**: Load only required collections into memory
- **Test Coverage**: ✅ 10 unit tests + regression test confirms functionality

**Implementation Files**:
- `backend/smart_vectorstore.py`: Core Smart Vector Store implementation
- `app.py`: Updated to use Smart Vector Store with cache statistics UI
- `tests/test_smart_vectorstore.py`: Comprehensive unit test suite
- `test_regression_smart_vectorstore.py`: Regression testing script

#### 3. **Adaptive Retrieval Parameters**
**Issue**: Fixed k=10 retrieval regardless of query complexity.

**Solution Proposal**:
- **Query Analysis**: Classify queries by type (factual, analytical, comparative) to determine optimal k
- **Dynamic Scaling**: Start with k=5 for simple queries, scale to k=20 for complex multi-part questions
- **Score Thresholding**: Filter results below similarity threshold (0.65-0.75 range)
- **Fallback Strategy**: Gradually lower thresholds if insufficient results found

**Implementation**:
```python
# In vectorstore.py
def adaptive_retrieve(query, vectordb):
    query_complexity = analyze_query_complexity(query)
    base_k = 5 if query_complexity == 'simple' else 15
    min_score = 0.7 if query_complexity == 'simple' else 0.65
    
    results = vectordb.similarity_search_with_score(query, k=base_k*2)
    filtered = [(doc, score) for doc, score in results if score >= min_score]
    return filtered[:base_k]
```

**Estimated Effort**: 6-8 hours
**Dependencies**: NLP analysis library, query classification logic

#### 4. **Query Preprocessing Pipeline**
**Issue**: No spell checking, query expansion, or semantic preprocessing.

**Solution Proposal**:
- **Spell Correction**: Use `pyspellchecker` with legal domain dictionary
- **Query Expansion**: Add synonyms and legal terminology variants
- **Normalization**: Standardize legal abbreviations and citations
- **Context Enhancement**: Extract key entities and expand with related terms

**Implementation**:
```python
# New module: backend/query_processor.py
class QueryProcessor:
    def __init__(self):
        self.spellchecker = SpellChecker()
        self.legal_synonyms = load_legal_dictionary()
    
    def preprocess(self, query):
        corrected = self.spell_correct(query)
        expanded = self.expand_legal_terms(corrected)
        normalized = self.normalize_citations(expanded)
        return normalized
```

**Estimated Effort**: 8-10 hours
**Dependencies**: `pyspellchecker`, legal terminology database

#### 5. **Semantic Chunking Improvements** ✅
**Status**: **IMPLEMENTED** (June 19th, 2025)

**Issue**: Hard character limits break semantic units, causing information loss and poor retrieval quality.

**Solution Implemented**:
- ✅ **Sentence Boundary Preservation**: Implemented NLTK sentence tokenizer to maintain semantic integrity
- ✅ **Sliding Window Overlap**: 200-character configurable overlap between chunks to maintain context
- ✅ **Hierarchical Chunking**: Respects paragraph and section boundaries with forced breaks at section headers
- ✅ **Enhanced Metadata**: Added chunk_size and sentence_count to metadata for better analysis
- ✅ **Backward Compatibility**: Legacy chunking function preserved as `semantic_chunk_legacy()`

**Key Improvements**:
```python
def semantic_chunk(text, max_chunk_size=1000, overlap_size=200, min_chunk_size=100):
    """
    Improved semantic chunking with sentence boundaries and sliding window overlap.
    - Preserves sentence boundaries using NLTK tokenization
    - Implements configurable sliding window overlap
    - Respects section boundaries with metadata tracking
    - Enhanced metadata including chunk_size and sentence_count
    """
```

**Performance Results**:
- **Chunk Quality**: Better semantic coherence with sentence-level boundaries
- **Overlap Effectiveness**: 200-character overlap maintains context between chunks
- **Metadata Enhancement**: Rich metadata enables better retrieval analysis
- **Regression Testing**: ✅ Passes all existing functionality tests

**Test Coverage**:
- ✅ **Unit Tests**: 13 comprehensive tests in `tests/test_chunker.py`
- ✅ **Integration Tests**: Pipeline integration verified in `tests/test_pipeline.py`
- ✅ **Regression Tests**: Single-question baseline test confirms compatibility

**Implementation Files**:
- `backend/chunker.py`: Enhanced semantic chunking with overlap and sentence boundaries
- `tests/test_chunker.py`: Comprehensive unit test suite
- `test_regression_single.py`: Regression testing script

**Dependencies**: NLTK (already in requirements.txt)

#### 6. **Retrieval Quality Validation**
**Issue**: No similarity score thresholds or result filtering.

**Solution Proposal**:
- **Multi-Tier Scoring**: Implement confidence bands (high/medium/low) based on similarity scores
- **Result Validation**: Check for semantic coherence between query and retrieved chunks
- **Quality Metrics**: Track retrieval precision and provide user feedback
- **Fallback Handling**: Graceful degradation when no high-quality matches found

**Implementation**:
```python
# In vectorstore.py
def validate_retrieval_quality(query, results):
    quality_bands = {
        'high': results with score > 0.8,
        'medium': results with 0.65 < score <= 0.8,
        'low': results with 0.5 < score <= 0.65
    }
    
    if not quality_bands['high'] and not quality_bands['medium']:
        return expand_search_or_rephrase_suggestion(query)
    
    return filtered_results_with_confidence_scores
```

**Estimated Effort**: 5-7 hours
**Dependencies**: Score analysis and user feedback mechanisms

### 🟡 Medium Priority - Advanced Features

#### 7. **Hybrid Search Implementation**
**Issue**: Only vector similarity search, missing keyword-based retrieval.

**Solution Proposal**:
- **BM25 Integration**: Use `rank_bm25` for keyword-based search
- **Score Fusion**: Implement Reciprocal Rank Fusion (RRF) to combine vector and keyword scores
- **Weighted Combination**: Allow user to adjust vector vs. keyword search balance
- **Query Analysis**: Auto-detect when to favor keyword vs. semantic search

**Implementation**:
```python
# New module: backend/hybrid_search.py
class HybridSearcher:
    def search(self, query, vector_db, bm25_index):
        vector_results = vector_db.similarity_search_with_score(query, k=20)
        bm25_results = bm25_index.get_top_n(query.split(), corpus, n=20)
        
        combined = self.reciprocal_rank_fusion(vector_results, bm25_results)
        return combined[:10]
```

**Estimated Effort**: 10-12 hours
**Dependencies**: `rank_bm25`, score fusion algorithms

#### 8. **Reranking Pipeline**
**Issue**: Retrieved chunks need reranking by query-specific relevance.

**Solution Proposal**:
- **Cross-Encoder Reranking**: Use sentence-transformers cross-encoder for precise relevance scoring
- **Query-Chunk Alignment**: Score how well each chunk answers the specific question
- **Diversity Injection**: Ensure variety in retrieved content to avoid redundancy
- **Multi-Criteria Scoring**: Balance relevance, completeness, and authority

**Implementation**:
```python
# New module: backend/reranker.py
class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query, candidates):
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.model.predict(pairs)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

**Estimated Effort**: 8-10 hours
**Dependencies**: `sentence-transformers`, cross-encoder models

#### 9. **Prompt Flexibility (Double-Check Mode)**
**Issue**: Double-check prompt exists but isn't wired into the application.

**Solution Proposal**:
- **UI Toggle**: Add Streamlit checkbox for "Double-Check Mode"
- **Prompt Router**: Dynamically select prompt template based on user preference
- **Confidence Integration**: Use double-check mode automatically for low-confidence retrievals
- **Response Comparison**: Show differences between normal and double-check responses

**Implementation**:
```python
# In app.py
double_check_mode = st.checkbox("Enable Double-Check Mode", 
                                help="More thorough analysis with verification steps")

# In llm_wrapper.py
def select_prompt(query, double_check=False, confidence_score=None):
    if double_check or (confidence_score and confidence_score < 0.7):
        return DOUBLE_CHECK_PROMPT
    return DEFAULT_PROMPT
```

**Estimated Effort**: 4-6 hours
**Dependencies**: UI updates, prompt routing logic

#### 10. **Metadata-Aware Retrieval**
**Issue**: Page/section metadata exists but isn't used for filtering or boosting.

**Solution Proposal**:
- **Section-Based Filtering**: Allow users to search within specific document sections
- **Page Range Queries**: Enable searches like "find information from pages 10-20"
- **Metadata Boosting**: Prioritize results from relevant sections (e.g., "Findings" for factual queries)
- **Hierarchical Context**: Show document structure in results

**Implementation**:
```python
# In vectorstore.py
def metadata_aware_search(query, vectordb, section_filter=None, page_range=None):
    base_results = vectordb.similarity_search_with_score(query, k=20)
    
    if section_filter:
        filtered = [r for r in base_results if section_filter in r[0].metadata.get('section', '')]
    
    if page_range:
        filtered = [r for r in base_results if page_range[0] <= r[0].metadata.get('page', 0) <= page_range[1]]
    
    return apply_metadata_boosting(filtered, query)
```

**Estimated Effort**: 6-8 hours
**Dependencies**: Metadata filtering logic, UI controls

### 🟢 Low Priority - Infrastructure

#### 11. **Support for Different Document Types**
**Issue**: Current assumptions lean heavily on legal rulings.

**Solution Proposal**:
- **Document Type Detection**: Auto-classify document types (rulings, motions, contracts, etc.)
- **Type-Specific Chunking**: Adapt chunking strategies per document type
- **Specialized Prompts**: Use document-type-aware prompt templates
- **Metadata Enrichment**: Extract type-specific metadata (case numbers, parties, dates)

**Estimated Effort**: 12-15 hours

#### 12. **Query Logging / Batch Testing** ✅
**Status**: Already implemented with comprehensive baseline testing suite

**Current Test Coverage**:
- ✅ **Baseline Regression Tests**: `test_baseline.py` runs 8 comprehensive questions across 3 embedding models (24 test scenarios)
- ✅ **Pipeline Integration Tests**: `test_pipeline.py` validates document loading and chunking pipeline
- ✅ **Vectorstore Tests**: `test_vectorstore_pipeline.py` tests complete vectorstore creation and querying
- ✅ **All Tests Pass**: Tests have been reviewed and updated to work with current codebase (January 2025)
- ✅ **Auto-Rotation**: Results file automatically rotates when exceeding 1MB to keep repo lightweight

**Test Execution**:
```bash
# Run individual tests
python tests/test_pipeline.py
python tests/test_vectorstore_pipeline.py

# Run full baseline regression suite (takes 5+ minutes)
python tests/test_baseline.py
```

**Test Results**: 
- Saved to `results.md` with detailed model comparisons and answer quality analysis
- Auto-rotates to `results_backup_YYYYMMDD_HHMMSS.md` when file exceeds 1MB
- Backup files excluded from git tracking to maintain repo size

#### 13. **Persistent Vector Store Pruning**
**Issue**: Vector store directories accumulate without cleanup.

**Solution Proposal**:
- **Automatic Cleanup**: Remove unused collections after 30 days
- **Storage Monitoring**: Track disk usage and warn users
- **UI Management**: Add collection browser with delete options
- **Compression**: Archive old collections to reduce storage footprint

**Implementation**:
```python
# New module: backend/storage_manager.py
class StorageManager:
    def cleanup_old_collections(self, max_age_days=30):
        # Scan collection directories
        # Check last access time
        # Remove stale collections
    
    def get_storage_stats(self):
        # Return disk usage, collection count, etc.
```

**Estimated Effort**: 6-8 hours

---

## 🧪 Usage Instructions

### Option A: VS Code Dev Container (Recommended for New Users)
**Prerequisites**: VS Code + Docker Desktop installed

1. **Get the code** (ZIP file, USB, or git clone)
2. **Open in VS Code**: `code doc-analysis/`
3. **Click "Reopen in Container"** when VS Code prompts
4. **Wait 5-10 minutes** for automatic setup (downloads ~4GB of models)
5. **Run**: `streamlit run app.py` in VS Code terminal
6. **Open**: http://localhost:8501

**What gets installed automatically:**
- ✅ Python environment & dependencies
- ✅ BGE embedding models (bge-small-en, bge-base-en)
- ✅ Ollama + nomic-embed-text embedding
- ✅ Llama3 LLM model
- ✅ All configured and ready to use!

### Option B: Manual Setup
```bash
# 1. Set up environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (One-time) Download local BGE models
python download_hf_bbg.py

# 4. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧭 Implementation Roadmap & Prioritization

### Phase 1: Foundation Fixes (Week 1-2) - **Total: 25-35 hours**
**Priority: Fix core retrieval issues that impact all queries**

**Quick Wins (4-6 hours each):**
1. ✅ **Smart Vector Store Management** - COMPLETED: Document fingerprinting, caching, and storage management
2. **Double-Check Mode Integration** - Wire existing prompt into UI toggle
3. **Retrieval Quality Validation** - Add score thresholds and confidence bands

**Medium Effort (6-8 hours each):**
4. **Adaptive Retrieval Parameters** - Dynamic k selection based on query complexity  
5. **Semantic Chunking Improvements** - Sentence boundaries and sliding window overlap
6. **Metadata-Aware Retrieval** - Leverage existing page/section metadata for filtering

### Phase 2: Advanced Retrieval (Week 3-4) - **Total: 26-30 hours**
**Priority: Enhance retrieval accuracy and recall**

**High Impact Features:**
1. **Query Preprocessing Pipeline** (8-10 hours) - Spell correction, legal term expansion
2. **Hybrid Search Implementation** (10-12 hours) - BM25 + vector fusion with RRF
3. **Cross-Encoder Reranking** (8-10 hours) - Precise relevance scoring for top results

### Phase 3: Infrastructure & Polish (Week 5-6) - **Total: 18-23 hours**
**Priority: Scalability and user experience**

**Infrastructure:**
1. **Storage Management** (6-8 hours) - Automatic cleanup and disk monitoring
2. **Document Type Support** (12-15 hours) - Multi-format chunking and metadata extraction

**Estimated Total Effort: 69-88 hours (9-11 weeks part-time)**

### Implementation Priority Matrix

| Feature | Impact | Effort | Priority | Dependencies |
|---------|--------|--------|----------|--------------|
| ✅ Smart Vector Store | High | Low | **DONE** | None |
| Retrieval Quality Validation | High | Medium | **P0** | None |
| Adaptive Retrieval Parameters | High | Medium | **P1** | Query analysis |
| Semantic Chunking | High | Medium | **P1** | NLTK/spaCy |
| Query Preprocessing | High | High | **P1** | Legal dictionary |
| Hybrid Search | Medium | High | **P2** | BM25, score fusion |
| Cross-Encoder Reranking | Medium | High | **P2** | sentence-transformers |
| Double-Check Mode | Low | Low | **P2** | UI updates |
| Metadata-Aware Search | Medium | Medium | **P2** | UI controls |
| Storage Management | Low | Medium | **P3** | Background services |
| Document Type Support | Medium | High | **P3** | ML classification |

### Success Metrics & Validation

**Phase 1 Success Criteria:**
- [x] ✅ Vector stores persist across sessions (measure: storage reuse rate) - ACHIEVED: 50% cache hit rate in testing
- [ ] Retrieval quality scores visible to users (measure: user confidence)
- [ ] Adaptive k selection reduces irrelevant results by 20%

**Phase 2 Success Criteria:**
- [ ] Query preprocessing improves retrieval accuracy by 15% (measured via test suite)
- [ ] Hybrid search increases recall for specific legal terms by 25%
- [ ] Cross-encoder reranking improves top-3 relevance by 30%

**Phase 3 Success Criteria:**
- [ ] Storage usage optimization (measure: disk space reduction)
- [ ] Support for 3+ document types with type-specific handling
- [ ] User satisfaction scores improve by 20% (via feedback integration)
---

## 👤 Author Notes

Built for fast, iterative experimentation on local LLMs. Designed with legal and analytical documents in mind but flexible enough for general enterprise use.
