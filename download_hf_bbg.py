from sentence_transformers import SentenceTransformer

# Downloads and saves the model locally
model = SentenceTransformer("BAAI/bge-small-en")
model.save("/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/bge-small-en")

# Full path to save the model
model = SentenceTransformer("BAAI/bge-base-en")
model.save("/Users/sandeepmangaraj/myworkspace/Utilities/doc-analysis/models/bge-base-en")