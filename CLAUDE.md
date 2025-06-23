Testing using test_baseline_smart.py, I got the following error (see excerpt)
🔢 Testing with model: bge-small-en

--- Question 1/8 ---
Processing question with bge-small-en: What was the core allegation made by the plaintiff...
🔍 Using Hugging Face embedder from: /Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/bge-small-en
Cache HIT: Reusing existing vector store for doc_7afbf3442d26_bge...
Cache HIT: Loaded collection with 71 documents
❌ ERROR processing question: too many values to unpack (expected 2)

--- Question 2/8 ---
Processing question with bge-small-en: Who were the defendants in this case?...
🔍 Using Hugging Face embedder from: /Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/bge-small-en
Cache HIT: Reusing existing vector store for doc_7afbf3442d26_bge...
Cache HIT: Loaded collection with 71 documents
❌ ERROR processing question: too many values to unpack (expected 2)
Please review the code and see what is the issue
