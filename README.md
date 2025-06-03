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
  - Page
  - Section
- ✅ Clean toggle and storage separation for vector stores
- ✅ Error-handling and diagnostics for model compatibility and Chroma issues

---

## ⚠️ Features NOT Yet Implemented (Next Steps)

1. ❌ **Prompt flexibility (double-check mode)**  
   While `prompts.py` includes a double-check prompt, the toggle and logic for switching between the default and verification prompts is not yet wired into `llm_wrapper.py` or `app.py`.

2. ❌ **Support for different document types**  
   Current assumptions lean heavily on legal rulings (e.g., court opinions). Chunk labeling logic may not generalize to motions, filings, or expert opinions.

3. ❌ **Persistent vector store pruning**  
   Vectorstore directories accumulate; a background cleaner or UI option for stale store removal is pending.

4. ❌ **Query logging / batch testing**  
   No functionality yet for saving and comparing model answers across test suites.

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

## 🧭 Next Steps

### 1. Implement Prompt Flexibility

- Add a checkbox in Streamlit for “Double-check Mode”
- In `llm_wrapper.py`, modify `synthesize_answer` to select between `QNA_PROMPT_TEMPLATE` and `QNA_DOUBLECHECK_PROMPT_TEMPLATE`
- Use top-k context aggregation to feed multiple chunks to the LLM

### 2. Refine Chunk Labels

- Add more resilient section/page detection in `chunker.py`
- Use regex or headings for better “section” parsing

### 3. Add Vectorstore Cleanup Utility

- Add a script to prune unused or duplicate Chroma stores

### 4. Expand Evaluation Suite

- Automate running a batch of predefined queries across all three models
- Store results and chunk provenance for side-by-side comparison

---

## 👤 Author Notes

Built for fast, iterative experimentation on local LLMs. Designed with legal and analytical documents in mind but flexible enough for general enterprise use.
