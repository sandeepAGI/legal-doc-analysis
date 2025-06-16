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
├── tests/
│   ├── test_baseline.py          # Baseline test suite for embedding model comparison
│   ├── test_pipeline.py          # Pipeline integration tests
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
- ✅ Semantic chunking with page/section detection
- ✅ Metadata-aware display of chunks in response view:
  - Chunk #
  - Similarity Score
  - Section
- ✅ Model-specific vector store separation to prevent dimension conflicts
- ✅ Error-handling and diagnostics for model compatibility and Chroma issues

---

## ⚠️ Critical Retrieval Issues & Implementation Solutions

### 🔴 High Priority - Retrieval Robustness

#### 1. **Persistent Vector Store Management**
**Issue**: Creates new vector store each run, preventing reuse and wasting computation.

**Solution Proposal**:
- **Document Fingerprinting**: Use MD5 hash of document content + embedding model to create unique collection names
- **Collection Reuse Logic**: Check if collection exists before recreating, with file modification time validation
- **Storage Structure**: Organize collections by document hash with metadata files for quick lookup

**Implementation**:
```python
# In app.py and vectorstore.py
def get_collection_name(file_content, embedding_model):
    doc_hash = hashlib.md5(file_content).hexdigest()[:12]
    model_suffix = embedding_model.replace('-', '_').replace('/', '_')
    return f"doc_{doc_hash}_{model_suffix}"

def collection_exists(collection_name):
    # Check Chroma DB for existing collection
    # Validate freshness with metadata timestamp
```

**Estimated Effort**: 4-6 hours
**Dependencies**: Modify `app.py`, `vectorstore.py`

#### 2. **Adaptive Retrieval Parameters**
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

#### 3. **Query Preprocessing Pipeline**
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

#### 4. **Semantic Chunking Improvements**
**Issue**: Hard character limits break semantic units.

**Solution Proposal**:
- **Sentence Boundary Preservation**: Use NLTK/spaCy sentence tokenizer
- **Sliding Window Overlap**: 150-200 character overlap between chunks
- **Hierarchical Chunking**: Respect paragraph and section boundaries
- **Dynamic Sizing**: Adjust chunk size based on content density

**Implementation**:
```python
# In chunker.py
def improved_semantic_chunk(text, target_size=800, overlap=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        if current_size + len(sentence) > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Create overlap
            overlap_sentences = current_chunk[-overlap_sentence_count:]
            current_chunk = overlap_sentences + [sentence]
            current_size = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_size += len(sentence)
    
    return chunks
```

**Estimated Effort**: 6-8 hours
**Dependencies**: NLTK or spaCy for sentence tokenization

#### 5. **Retrieval Quality Validation**
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

#### 6. **Hybrid Search Implementation**
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

#### 7. **Reranking Pipeline**
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

#### 8. **Prompt Flexibility (Double-Check Mode)**
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

#### 9. **Metadata-Aware Retrieval**
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

#### 10. **Support for Different Document Types**
**Issue**: Current assumptions lean heavily on legal rulings.

**Solution Proposal**:
- **Document Type Detection**: Auto-classify document types (rulings, motions, contracts, etc.)
- **Type-Specific Chunking**: Adapt chunking strategies per document type
- **Specialized Prompts**: Use document-type-aware prompt templates
- **Metadata Enrichment**: Extract type-specific metadata (case numbers, parties, dates)

**Estimated Effort**: 12-15 hours

#### 11. **Query Logging / Batch Testing** ✅
**Status**: Already implemented with baseline testing suite

#### 12. **Persistent Vector Store Pruning**
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

## 🧭 Implementation Roadmap & Prioritization

### Phase 1: Foundation Fixes (Week 1-2) - **Total: 25-35 hours**
**Priority: Fix core retrieval issues that impact all queries**

**Quick Wins (4-6 hours each):**
1. **Persistent Vector Store Management** - Implement document fingerprinting and collection reuse
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
| Persistent Vector Store | High | Low | **P0** | None |
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
- [ ] Vector stores persist across sessions (measure: storage reuse rate)
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
